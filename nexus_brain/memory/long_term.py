"""LONG-TERM MEMORY - hippocampus + neocortex + basal ganglia.

Three stores behind one facade:

* **Episodic** - timestamped records of specific interactions/events
  ("On 2026-09-05 the user asked me to book a dentist; the calendar tool
  failed once then succeeded").
* **Semantic** - general facts and learned concepts, including user
  preferences ("User's name is Priya", "User prefers 24h time").
* **Procedural** - skills/workflows with a success record ("schedule_meeting:
  check calendar -> create event -> confirm").

Retrieval
    score = w_rel * cosine(query, item)
          + w_rec * exp(-age / half_life)
          + w_imp * importance
    (weights are versioned config; the learning loop can nudge them within
    bounds). Access bumps ``access_count`` and ``last_accessed`` so
    frequently-useful memories become "stronger" - a crude analogue of
    long-term potentiation. Items also **decay**: importance drifts down when
    not accessed, and the ``forget()`` sweep can prune very weak items.

Communication
    ``retrieve()`` publishes ``memory.retrieved``; every write publishes
    ``memory.written``. Skills are read by the reasoning engine and their
    stats are updated by the learning loop.

Improvement
    Reflection writes new semantic facts and lessons, promotes recurring
    episodic patterns to semantic memory (consolidation), adjusts importance,
    and updates skill success rates.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import RetrievalWeights
from nexus_brain.core.schemas import MemoryItem, MemoryKind, RetrievedMemory, Skill, utcnow
from nexus_brain.memory.store import InMemoryVectorStore, VectorStore
from nexus_brain.perception.embedder import Embedder, HashingEmbedder, text_overlap


class LongTermMemory:
    def __init__(
        self,
        bus: EventBus,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        weights: Optional[RetrievalWeights] = None,
        skills_path: Optional[Path | str] = None,
    ) -> None:
        self.bus = bus
        self.embedder = embedder or HashingEmbedder()
        self.store = store if store is not None else InMemoryVectorStore()
        self.weights = weights or RetrievalWeights()
        self.skills: dict[str, Skill] = {}
        self.skills_path = Path(skills_path) if skills_path else None
        if self.skills_path and self.skills_path.exists():
            self._load_skills()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def remember(
        self,
        content: str,
        kind: MemoryKind,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        source: str = "system",
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
        dedupe: bool = True,
    ) -> MemoryItem:
        content = content.strip()
        if dedupe:
            existing = self._find_duplicate(content, kind)
            if existing is not None:
                existing.importance = max(existing.importance, importance)
                existing.access_count += 1
                existing.last_accessed = utcnow()
                self.store.update(existing)
                return existing
        item = MemoryItem(
            kind=kind,
            content=content,
            embedding=self.embedder.embed(content),
            importance=max(0.0, min(1.0, importance)),
            confidence=confidence,
            tags=tags or [],
            source=source,
            metadata=metadata or {},
        )
        self.store.add(item)
        self.bus.publish(
            Topics.MEMORY_WRITTEN,
            {"id": item.id, "kind": kind.value, "importance": item.importance, "source": source},
            source="long_term_memory",
        )
        return item

    def remember_episode(self, content: str, importance: float = 0.5, **kw) -> MemoryItem:
        return self.remember(content, MemoryKind.EPISODIC, importance, dedupe=False, **kw)

    def remember_fact(self, content: str, importance: float = 0.6, **kw) -> MemoryItem:
        return self.remember(content, MemoryKind.SEMANTIC, importance, **kw)

    def _find_duplicate(self, content: str, kind: MemoryKind) -> Optional[MemoryItem]:
        emb = self.embedder.embed(content)
        for item, sim in self.store.search(emb, k=3, kinds=[kind]):
            if sim > 0.92 or item.content.lower() == content.lower():
                return item
        return None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        k: int = 5,
        kinds: Optional[Iterable[MemoryKind]] = None,
        query_embedding: Optional[list[float]] = None,
        min_score: float = 0.15,
        now: Optional[datetime] = None,
    ) -> list[RetrievedMemory]:
        now = now or utcnow()
        w = self.weights.normalised()
        emb = query_embedding or self.embedder.embed(query)
        candidates = self.store.search(emb, k=max(k * 4, 20), kinds=kinds)
        scored: list[RetrievedMemory] = []
        for item, sim in candidates:
            # Blend vector similarity with lexical overlap so the hashing
            # embedder's quirks do not dominate; a true semantic embedder makes
            # the overlap term nearly redundant but harmless.
            relevance = max(0.0, 0.7 * sim + 0.3 * text_overlap(query, item.content))
            age_h = max(0.0, (now - item.created_at).total_seconds() / 3600.0)
            recency = math.exp(-math.log(2) * age_h / max(1e-6, w.recency_half_life_hours))
            score = w.relevance * relevance + w.recency * recency + w.importance * item.importance
            if relevance < 0.05:
                # Never surface something purely because it is recent/important.
                continue
            scored.append(RetrievedMemory(item=item, relevance=relevance, recency=recency, importance=item.importance, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        top = [r for r in scored if r.score >= min_score][:k]
        for r in top:
            r.item.access_count += 1
            r.item.last_accessed = now
            self.store.update(r.item)
        self.bus.publish(
            Topics.MEMORY_RETRIEVED,
            {"query": query[:80], "count": len(top), "top_score": top[0].score if top else 0.0},
            source="long_term_memory",
        )
        return top

    # ------------------------------------------------------------------ #
    # Maintenance ("sleep")
    # ------------------------------------------------------------------ #
    def decay(self, rate: float = 0.02, now: Optional[datetime] = None) -> int:
        """Reduce importance of memories that have not been accessed recently."""
        now = now or utcnow()
        touched = 0
        for item in self.store.all():
            idle_days = (now - item.last_accessed).total_seconds() / 86400.0
            if idle_days > 1 and item.source != "user":  # user-stated facts decay slower
                item.importance = max(0.05, item.importance - rate * idle_days)
                self.store.update(item)
                touched += 1
        return touched

    def forget(self, importance_below: float = 0.08, min_age_days: float = 7.0, now: Optional[datetime] = None) -> int:
        now = now or utcnow()
        removed = 0
        for item in list(self.store.all()):
            age_days = (now - item.created_at).total_seconds() / 86400.0
            if item.importance < importance_below and age_days >= min_age_days and item.access_count <= 1:
                self.store.delete(item.id)
                removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        return {
            "episodic": len(self.store.all([MemoryKind.EPISODIC])),
            "semantic": len(self.store.all([MemoryKind.SEMANTIC])),
            "procedural": len(self.skills),
        }

    # ------------------------------------------------------------------ #
    # Procedural memory
    # ------------------------------------------------------------------ #
    def add_skill(self, skill: Skill) -> Skill:
        self.skills[skill.name] = skill
        self.remember(
            f"skill:{skill.name} - {skill.description}. triggers: {', '.join(skill.triggers)}",
            MemoryKind.PROCEDURAL,
            importance=0.7,
            tags=["skill", skill.name],
            metadata={"skill": skill.name},
        )
        self._save_skills()
        return skill

    def find_skills(self, text: str, intents: Iterable[str] = ()) -> list[tuple[Skill, float]]:
        """Return skills whose triggers match the text, best first."""
        lowered = text.lower()
        intents = set(intents)
        out: list[tuple[Skill, float]] = []
        for skill in self.skills.values():
            hits = sum(1 for t in skill.triggers if t.lower() in lowered)
            intent_hit = 1 if any(t in intents for t in skill.triggers) else 0
            if hits + intent_hit == 0:
                continue
            score = min(1.0, 0.4 * intent_hit + 0.3 * hits)
            out.append((skill, score))
        out.sort(key=lambda s: (s[1], s[0].success_rate), reverse=True)
        return out

    def record_skill_outcome(self, name: str, success: bool) -> Optional[Skill]:
        skill = self.skills.get(name)
        if not skill:
            return None
        if success:
            skill.successes += 1
        else:
            skill.failures += 1
        self._save_skills()
        return skill

    def _save_skills(self) -> None:
        if not self.skills_path:
            return
        self.skills_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills_path.write_text(json.dumps([s.model_dump(mode="json") for s in self.skills.values()], indent=2))

    def _load_skills(self) -> None:
        assert self.skills_path is not None
        try:
            for raw in json.loads(self.skills_path.read_text() or "[]"):
                s = Skill.model_validate(raw)
                self.skills[s.name] = s
        except json.JSONDecodeError:
            pass
