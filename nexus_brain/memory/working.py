"""WORKING MEMORY - the prefrontal scratchpad.

Purpose
    Hold the short-term context for the current task/conversation: recent
    turns, the active goal, the retrieved long-term memories in use, and
    scratch notes from reasoning.

Technique
    A token-budgeted buffer with a three-way policy as it grows:
      * KEEP    - the last ``keep_last_turns`` turns are always verbatim.
      * SUMMARIZE - older turns are compressed into a rolling summary
                    (extractive here; LLM-abstractive in production).
      * DISCARD - low-salience turns (greetings, acknowledgements) are dropped
                  first, before anything is summarised.
    Before discarding, anything with lasting value is offered to long-term
    memory ("consolidation") - like hippocampal replay during sleep.

Communication
    Read by reasoning/language every cycle; writes to LongTermMemory only via
    the consolidation callback; publishes ``working_memory.compressed``.

Data
    Volatile per-session state; nothing durable of its own.

Improvement
    ``token_budget``, ``keep_last_turns`` and ``summary_ratio`` are versioned
    config knobs that the upgrade manager can tune with evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import WorkingMemoryConfig
from nexus_brain.core.schemas import RetrievedMemory
from nexus_brain.perception.embedder import tokenize

_LOW_SALIENCE = {"ok", "okay", "thanks", "thank you", "hi", "hello", "hey", "sure", "yes", "no", "cool", "great"}


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


@dataclass
class Turn:
    role: str  # user | assistant | system | tool
    content: str
    salience: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tokens(self) -> int:
        return approx_tokens(self.content)


class WorkingMemory:
    def __init__(
        self,
        bus: EventBus,
        config: Optional[WorkingMemoryConfig] = None,
        consolidate: Optional[Callable[[Turn], None]] = None,
    ) -> None:
        self.bus = bus
        self.config = config or WorkingMemoryConfig()
        self._consolidate = consolidate
        self.turns: list[Turn] = []
        self.summary: str = ""
        self.active_goal: Optional[str] = None
        self.retrieved: list[RetrievedMemory] = []
        self.scratch: dict[str, str] = {}
        self.compressions = 0

    # ------------------------------------------------------------------ #
    def add_turn(self, role: str, content: str, salience: Optional[float] = None) -> Turn:
        if salience is None:
            salience = self._estimate_salience(role, content)
        turn = Turn(role=role, content=content, salience=salience)
        self.turns.append(turn)
        self._enforce_budget()
        return turn

    def set_goal(self, goal: str) -> None:
        self.active_goal = goal

    def set_retrieved(self, items: list[RetrievedMemory]) -> None:
        self.retrieved = items

    def note(self, key: str, value: str) -> None:
        self.scratch[key] = value

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.active_goal = None
        self.retrieved.clear()
        self.scratch.clear()

    # ------------------------------------------------------------------ #
    @property
    def used_tokens(self) -> int:
        return sum(t.tokens for t in self.turns) + approx_tokens(self.summary)

    def recent(self, n: Optional[int] = None) -> list[Turn]:
        n = n or self.config.keep_last_turns
        return self.turns[-n:]

    def context_window(self) -> str:
        """Render the buffer the way it would be handed to an LLM."""
        parts: list[str] = []
        if self.active_goal:
            parts.append(f"[goal] {self.active_goal}")
        if self.summary:
            parts.append(f"[earlier conversation summary] {self.summary}")
        if self.retrieved:
            mem_lines = [f"- ({m.item.kind.value}, score={m.score:.2f}) {m.item.content}" for m in self.retrieved[:5]]
            parts.append("[relevant memories]\n" + "\n".join(mem_lines))
        for t in self.turns:
            parts.append(f"{t.role}: {t.content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    def _estimate_salience(self, role: str, content: str) -> float:
        lowered = content.strip().lower().rstrip("!.")
        if lowered in _LOW_SALIENCE or len(tokenize(content)) <= 1:
            return 0.1
        score = 0.5
        if role == "user":
            score += 0.1
        if any(k in lowered for k in ("remember", "prefer", "my name", "always", "never", "deadline", "important")):
            score += 0.3
        if "?" in content:
            score += 0.05
        return min(1.0, score)

    def _enforce_budget(self) -> None:
        budget = self.config.token_budget
        if self.used_tokens <= budget:
            return
        keep_n = self.config.keep_last_turns
        protected = self.turns[-keep_n:]
        candidates = self.turns[:-keep_n]
        if not candidates:
            return

        # 1) discard low-salience old turns first
        discarded = [t for t in candidates if t.salience < 0.2]
        survivors = [t for t in candidates if t.salience >= 0.2]

        # 2) summarise the remaining old turns (offer to LTM first)
        summarised: list[Turn] = []
        while survivors and (sum(t.tokens for t in survivors + protected) + approx_tokens(self.summary)) > budget:
            summarised.append(survivors.pop(0))
        if summarised:
            for t in summarised:
                if self._consolidate and t.salience >= 0.6:
                    self._consolidate(t)
            self.summary = self._merge_summary(self.summary, summarised)

        self.turns = survivors + protected
        self.compressions += 1
        self.bus.publish(
            Topics.WORKING_MEMORY_COMPRESSED,
            {"discarded": len(discarded), "summarised": len(summarised), "used_tokens": self.used_tokens},
            source="working_memory",
        )

    def _merge_summary(self, existing: str, turns: list[Turn]) -> str:
        """Extractive summary: keep the most salient sentence of each turn.

        In production this call is replaced by an LLM abstractive summary with
        an instruction to preserve names, numbers, commitments and open questions.
        """
        picks: list[str] = []
        for t in turns:
            sentences = [s.strip() for s in t.content.replace("!", ".").replace("?", "?.").split(".") if s.strip()]
            if not sentences:
                continue
            best = max(sentences, key=lambda s: len(tokenize(s)) * (1.0 + t.salience))
            picks.append(f"{t.role}: {best}")
        combined = (existing + " " if existing else "") + " ".join(picks)
        max_tokens = int(self.config.token_budget * self.config.summary_ratio)
        words = combined.split()
        if len(words) > max_tokens:
            combined = " ".join(words[-max_tokens:])
        return combined.strip()
