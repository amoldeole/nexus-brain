"""LEARNING & SELF-IMPROVEMENT LOOP - sleep-time consolidation.

Purpose
    Log the outcome of every cognitive cycle, periodically review the log to
    extract lessons (what worked / what didn't), and turn lessons into
    *bounded* updates: memory writes, skill statistics, small retrieval-weight
    nudges. Anything bigger (prompt changes, config changes, model swaps) is
    emitted as a ``Proposal`` for the UpgradeManager - never applied here.

Technique
    Deterministic pattern mining over the outcome log:
      * per-skill success rate -> skill stats + a semantic "lesson" memory
      * per-tool failure rate -> lesson + (if persistent) a CONFIG_CHANGE
        proposal (e.g. raise retries, mark tool unreliable)
      * retrieval quality vs. success correlation -> RETRIEVAL_WEIGHT nudge
        within configured bounds
      * negative user feedback clusters -> PROMPT_UPDATE proposal
      * recurring facts in episodic memory -> promote to semantic memory
    In production an LLM writes the natural-language lesson text and drafts
    the proposal diff; the *gating* (what may auto-apply) remains code.

Communication
    ``log_outcome`` is called by the orchestrator at the end of each cycle;
    ``reflect`` runs every N cycles or on demand. Publishes
    ``learning.outcome_logged``, ``learning.reflection_complete``,
    ``learning.proposal_created``.

Data
    Reads/writes the outcome log (JSONL), long-term memory, skill stats,
    retrieval weights (bounded). Hands proposals to the UpgradeManager.

Improvement
    This *is* the improvement mechanism - and it is deliberately not allowed
    to modify code, governance rules, or its own gating thresholds.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import LearningConfig, RetrievalWeights
from nexus_brain.core.schemas import Lesson, MemoryKind, Outcome, Proposal, ProposalKind
from nexus_brain.governance.audit import AuditLog
from nexus_brain.memory.long_term import LongTermMemory

AUTO_APPLY_KINDS = {ProposalKind.MEMORY_WRITE, ProposalKind.SKILL_STATS, ProposalKind.RETRIEVAL_WEIGHT}


class LearningLoop:
    def __init__(
        self,
        bus: EventBus,
        memory: LongTermMemory,
        audit: AuditLog,
        config: Optional[LearningConfig] = None,
        log_path: Optional[Path | str] = None,
        on_proposal: Optional[Callable[[Proposal], None]] = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.audit = audit
        self.config = config or LearningConfig()
        self.log_path = Path(log_path) if log_path else None
        self.on_proposal = on_proposal
        self.outcomes: list[Outcome] = []
        self.lessons: list[Lesson] = []
        self.proposals: list[Proposal] = []
        self._since_reflection = 0
        self._reflected_upto = 0
        if self.log_path and self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                if line.strip():
                    self.outcomes.append(Outcome.model_validate_json(line))
            self._reflected_upto = len(self.outcomes)

    # ------------------------------------------------------------------ #
    def log_outcome(self, outcome: Outcome) -> Optional[list[Lesson]]:
        self.outcomes.append(outcome)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(outcome.model_dump_json() + "\n")
        # immediate, bounded update: skill statistics
        if outcome.skill_name and outcome.success is not None and outcome.skill_name in self.memory.skills:
            self.memory.record_skill_outcome(outcome.skill_name, outcome.success)
        self.bus.publish(Topics.OUTCOME_LOGGED, {"cycle_id": outcome.cycle_id, "success": outcome.success, "mode": outcome.mode.value}, source="learning")
        self._since_reflection += 1
        if self._since_reflection >= self.config.reflect_every_n_cycles:
            return self.reflect()
        return None

    # ------------------------------------------------------------------ #
    def reflect(self, window: Optional[int] = None) -> list[Lesson]:
        """Review recent outcomes, write lessons, emit proposals."""
        batch = self.outcomes[self._reflected_upto :] if window is None else self.outcomes[-window:]
        self._since_reflection = 0
        self._reflected_upto = len(self.outcomes)
        if not batch:
            return []
        lessons: list[Lesson] = []
        lessons += self._skill_lessons(batch)
        lessons += self._tool_lessons(batch)
        lessons += self._retrieval_lessons(batch)
        lessons += self._feedback_lessons(batch)
        lessons += self._consolidate_episodes()

        for lesson in lessons:
            self.lessons.append(lesson)
            for proposal in lesson.proposals:
                self._route(proposal)
        self.audit.record("learning", "reflection", f"{len(batch)} outcomes -> {len(lessons)} lesson(s)", lessons=[l.text for l in lessons])
        self.bus.publish(Topics.REFLECTION_COMPLETE, {"outcomes": len(batch), "lessons": len(lessons)}, source="learning")
        return lessons

    # ------------------------------------------------------------------ #
    def _route(self, proposal: Proposal) -> None:
        self.proposals.append(proposal)
        if proposal.kind in AUTO_APPLY_KINDS and not proposal.requires_approval:
            self._auto_apply(proposal)
        else:
            proposal.status = "pending_eval"
            if self.on_proposal:
                self.on_proposal(proposal)
        self.bus.publish(Topics.PROPOSAL_CREATED, {"id": proposal.id, "kind": proposal.kind.value, "status": proposal.status, "title": proposal.title}, source="learning")

    def _auto_apply(self, proposal: Proposal) -> None:
        if proposal.kind == ProposalKind.MEMORY_WRITE:
            self.memory.remember(
                proposal.payload["content"], MemoryKind(proposal.payload.get("kind", "semantic")),
                importance=float(proposal.payload.get("importance", 0.6)), tags=proposal.payload.get("tags", ["lesson"]), source="reflection",
                confidence=proposal.confidence,
            )
        elif proposal.kind == ProposalKind.RETRIEVAL_WEIGHT:
            w = self.memory.weights
            lo, hi = self.config.retrieval_weight_bounds
            field = proposal.payload["field"]
            delta = float(proposal.payload["delta"])
            current = getattr(w, field)
            setattr(w, field, max(lo, min(hi, current + delta)))
            proposal.payload["applied_value"] = getattr(w, field)
        elif proposal.kind == ProposalKind.SKILL_STATS:
            pass  # already applied at log time; proposal exists for the record
        proposal.status = "applied"
        self.audit.record("learning", "auto_applied", proposal.title, kind=proposal.kind.value, payload=proposal.payload)

    # ------------------------------------------------------------------ #
    # pattern miners
    # ------------------------------------------------------------------ #
    def _skill_lessons(self, batch: list[Outcome]) -> list[Lesson]:
        per_skill: dict[str, list[Outcome]] = defaultdict(list)
        for o in batch:
            if o.skill_name and o.success is not None and o.skill_name in self.memory.skills:
                per_skill[o.skill_name].append(o)
        out: list[Lesson] = []
        for name, outs in per_skill.items():
            if len(outs) < self.config.min_evidence_for_lesson:
                continue
            rate = mean(1.0 if o.success else 0.0 for o in outs)
            evidence = [o.cycle_id for o in outs]
            if rate >= 0.8:
                text = f"Skill '{name}' is reliable ({rate:.0%} over {len(outs)} recent uses); safe for the System 1 fast path."
            elif rate <= 0.4:
                text = f"Skill '{name}' is under-performing ({rate:.0%} over {len(outs)} recent uses); prefer deliberate planning or revise its steps."
            else:
                continue
            lesson = Lesson(text=text, evidence=evidence, confidence=min(0.95, 0.5 + 0.1 * len(outs)))
            lesson.proposals.append(Proposal(kind=ProposalKind.MEMORY_WRITE, title=f"lesson about skill {name}", payload={"content": text, "kind": "semantic", "importance": 0.7, "tags": ["lesson", "skill", name]}, evidence=evidence, confidence=lesson.confidence, requires_approval=False))
            if rate <= 0.4:
                lesson.proposals.append(Proposal(kind=ProposalKind.CONFIG_CHANGE, title=f"raise System 1 threshold after '{name}' failures", payload={"path": "reasoning.system1_min_skill_success_rate", "delta": +0.05}, evidence=evidence, confidence=lesson.confidence, requires_approval=True))
            out.append(lesson)
        return out

    def _tool_lessons(self, batch: list[Outcome]) -> list[Lesson]:
        fails: Counter[str] = Counter()
        uses: Counter[str] = Counter()
        evidence: dict[str, list[str]] = defaultdict(list)
        for o in batch:
            for t in o.tools_used:
                uses[t] += 1
            for t in o.metadata.get("failed_tools", []):
                fails[t] += 1
                evidence[t].append(o.cycle_id)
        out: list[Lesson] = []
        for tool, n_fail in fails.items():
            if n_fail < self.config.min_evidence_for_lesson:
                continue
            rate = n_fail / max(1, uses[tool])
            text = f"Tool '{tool}' failed {n_fail}/{uses[tool]} times recently ({rate:.0%}); expect retries and keep a fallback ready."
            lesson = Lesson(text=text, evidence=evidence[tool], confidence=min(0.9, 0.4 + 0.1 * n_fail))
            lesson.proposals.append(Proposal(kind=ProposalKind.MEMORY_WRITE, title=f"lesson about tool {tool}", payload={"content": text, "kind": "semantic", "importance": 0.65, "tags": ["lesson", "tool", tool]}, evidence=evidence[tool], confidence=lesson.confidence, requires_approval=False))
            if rate >= 0.5:
                lesson.proposals.append(Proposal(kind=ProposalKind.CONFIG_CHANGE, title=f"increase retries because '{tool}' is flaky", payload={"path": "execution.max_retries", "delta": +1}, evidence=evidence[tool], confidence=lesson.confidence, requires_approval=True))
            out.append(lesson)
        return out

    def _retrieval_lessons(self, batch: list[Outcome]) -> list[Lesson]:
        judged = [o for o in batch if o.success is not None]
        if len(judged) < self.config.min_evidence_for_lesson * 2:
            return []
        good = [o.top_retrieval_score for o in judged if o.success]
        bad = [o.top_retrieval_score for o in judged if not o.success]
        if not good or not bad:
            return []
        gap = mean(good) - mean(bad)
        if abs(gap) < 0.1:
            return []
        step = self.config.retrieval_weight_step if gap > 0 else -self.config.retrieval_weight_step
        text = (
            f"Successful cycles had {'higher' if gap > 0 else 'lower'} retrieval relevance (Δ={gap:+.2f}); "
            f"{'increase' if gap > 0 else 'decrease'} the relevance weight slightly."
        )
        lesson = Lesson(text=text, evidence=[o.cycle_id for o in judged], confidence=min(0.85, 0.4 + abs(gap)))
        lesson.proposals.append(Proposal(kind=ProposalKind.RETRIEVAL_WEIGHT, title="nudge retrieval relevance weight", payload={"field": "relevance", "delta": step}, evidence=lesson.evidence, confidence=lesson.confidence, requires_approval=False))
        return [lesson]

    def _feedback_lessons(self, batch: list[Outcome]) -> list[Lesson]:
        negative = [o for o in batch if (o.user_feedback is not None and o.user_feedback < 0)]
        if len(negative) < self.config.min_evidence_for_lesson:
            return []
        tones = Counter(str(o.metadata.get("tone", "?")) for o in negative)
        worst_tone, n = tones.most_common(1)[0]
        text = f"{len(negative)} negative feedback events recently; most ({n}) under tone '{worst_tone}'. Review that style template."
        lesson = Lesson(text=text, evidence=[o.cycle_id for o in negative], confidence=0.5 + 0.05 * len(negative))
        lesson.proposals.append(Proposal(kind=ProposalKind.PROMPT_UPDATE, title=f"revise style template '{worst_tone}'", payload={"prompt": f"style.{worst_tone}", "suggestion": "Acknowledge the user's point explicitly before answering; keep sentences short."}, evidence=lesson.evidence, confidence=lesson.confidence, requires_approval=True))
        lesson.proposals.append(Proposal(kind=ProposalKind.MEMORY_WRITE, title="lesson about feedback", payload={"content": text, "kind": "semantic", "importance": 0.6, "tags": ["lesson", "feedback"]}, evidence=lesson.evidence, confidence=lesson.confidence, requires_approval=False))
        return [lesson]

    def _consolidate_episodes(self) -> list[Lesson]:
        """Promote facts that recur across episodes into semantic memory."""
        episodes = self.memory.store.all([MemoryKind.EPISODIC])
        facts: Counter[str] = Counter()
        for e in episodes:
            for f in e.metadata.get("facts", []):
                facts[f] += 1
        out: list[Lesson] = []
        for fact, n in facts.items():
            if n < 2:
                continue
            if self.memory._find_duplicate(fact, MemoryKind.SEMANTIC):
                continue
            lesson = Lesson(text=f"Recurring fact promoted to semantic memory: {fact}", confidence=min(0.9, 0.5 + 0.1 * n))
            lesson.proposals.append(Proposal(kind=ProposalKind.MEMORY_WRITE, title="consolidate recurring fact", payload={"content": fact, "kind": "semantic", "importance": 0.7, "tags": ["consolidated"]}, confidence=lesson.confidence, requires_approval=False))
            out.append(lesson)
        return out

    # ------------------------------------------------------------------ #
    def metrics(self, last_n: Optional[int] = None) -> dict[str, float]:
        outs = self.outcomes[-last_n:] if last_n else self.outcomes
        judged = [o for o in outs if o.success is not None]
        return {
            "cycles": len(outs),
            "success_rate": mean(1.0 if o.success else 0.0 for o in judged) if judged else 0.0,
            "system1_share": mean(1.0 if o.mode.value == "system1" else 0.0 for o in outs) if outs else 0.0,
            "avg_replans": mean(o.replans for o in outs) if outs else 0.0,
            "avg_latency_ms": mean(o.latency_ms for o in outs) if outs else 0.0,
            "lessons": len(self.lessons),
            "proposals": len(self.proposals),
        }
