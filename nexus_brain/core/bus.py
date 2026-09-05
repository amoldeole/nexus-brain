"""In-process event bus - the "thalamus" that relays signals between modules.

Modules do not import each other to send notifications; they publish typed
events and subscribers react. This keeps the faculties loosely coupled and
gives one place to tap for tracing, auditing and metrics.

The bus is deliberately synchronous and single-process. In v2/v3 the same
interface is backed by Redis Streams / NATS (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)

Handler = Callable[["Event"], None]


@dataclass
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Minimal topic-based pub/sub with wildcard subscription ("*")."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.history: list[Event] = []
        self.max_history = 5000

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        self._subs[topic].append(handler)

        def _unsubscribe() -> None:
            if handler in self._subs[topic]:
                self._subs[topic].remove(handler)

        return _unsubscribe

    def publish(self, topic: str, payload: dict[str, Any] | None = None, source: str = "unknown") -> Event:
        event = Event(topic=topic, payload=payload or {}, source=source)
        self.history.append(event)
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]
        for handler in list(self._subs.get(topic, [])) + list(self._subs.get("*", [])):
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - a bad subscriber must not break cognition
                log.exception("event handler failed for topic %s", topic)
        return event

    def clear_history(self) -> None:
        self.history.clear()


# Canonical topic names (kept in one place so typos are caught by grep, not prod).
class Topics:
    PERCEPT_READY = "perception.percept_ready"
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_WRITTEN = "memory.written"
    WORKING_MEMORY_COMPRESSED = "working_memory.compressed"
    AFFECT_UPDATED = "affect.updated"
    DECISION_MADE = "reasoning.decision_made"
    PLAN_CREATED = "planning.plan_created"
    ACTION_STARTED = "action.started"
    ACTION_FINISHED = "action.finished"
    ACTION_FAILED = "action.failed"
    REPLAN = "action.replan"
    APPROVAL_REQUESTED = "governance.approval_requested"
    APPROVAL_DECIDED = "governance.approval_decided"
    GOVERNANCE_DENIED = "governance.denied"
    RESPONSE_GENERATED = "language.response_generated"
    OUTCOME_LOGGED = "learning.outcome_logged"
    REFLECTION_COMPLETE = "learning.reflection_complete"
    PROPOSAL_CREATED = "learning.proposal_created"
    UPGRADE_PROPOSED = "upgrade.proposed"
    UPGRADE_EVALUATED = "upgrade.evaluated"
    UPGRADE_PROMOTED = "upgrade.promoted"
    UPGRADE_ROLLED_BACK = "upgrade.rolled_back"
    CYCLE_COMPLETE = "brain.cycle_complete"
