"""EMOTION-STATE SIMULATION - simulated affect for behavioural tuning.

>>> IMPORTANT: THIS IS NOT A CLAIM OF FEELING. <<<
The four variables below are scalar control signals, exactly like a
thermostat's setpoint error. They exist because a *useful assistant* should
answer an urgent user faster and shorter, hedge when it is unsure, slow down
and ask before acting when the interaction is going badly, and be warmer
with a user it has good rapport with. Nothing here experiences anything.
See docs/HONESTY.md.

Purpose
    Track: confidence, urgency, frustration_proxy, rapport (all in [0,1]).
    Expose them as biases for: tone/register (language), pacing/verbosity
    (language), risk tolerance and approval thresholds (reasoning/governance),
    and System1-vs-System2 selection (reasoning).

Technique
    Leaky integrator: each cycle the state decays toward a configurable
    baseline, then is nudged by evidence (perception cues, retrieval quality,
    action outcomes, explicit user feedback). Purely deterministic and fully
    inspectable - every update is published on the bus with its cause.

Communication
    Subscribes to nothing directly (the orchestrator calls ``update``); publishes
    ``affect.updated`` with the delta and reason for the audit log.

Data
    Volatile per-session; an ``AffectSnapshot`` is attached to every logged
    Outcome so reflection can study "how did state correlate with success".

Improvement
    Baselines and decay live in versioned config; the learning loop may
    propose (not silently apply) changes when evidence shows, e.g., that high
    urgency correlated with more failures.
"""
from __future__ import annotations

from typing import Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import AffectConfig
from nexus_brain.core.schemas import AffectSnapshot, Percept


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class AffectState:
    def __init__(self, bus: EventBus, config: Optional[AffectConfig] = None) -> None:
        self.bus = bus
        self.config = config or AffectConfig()
        self.confidence = self.config.baseline_confidence
        self.urgency = self.config.baseline_urgency
        self.frustration_proxy = self.config.baseline_frustration
        self.rapport = self.config.baseline_rapport
        self.history: list[AffectSnapshot] = []

    # ------------------------------------------------------------------ #
    def snapshot(self) -> AffectSnapshot:
        return AffectSnapshot(
            confidence=round(self.confidence, 3),
            urgency=round(self.urgency, 3),
            frustration_proxy=round(self.frustration_proxy, 3),
            rapport=round(self.rapport, 3),
        )

    def reset(self) -> None:
        c = self.config
        self.confidence, self.urgency = c.baseline_confidence, c.baseline_urgency
        self.frustration_proxy, self.rapport = c.baseline_frustration, c.baseline_rapport

    def _decay(self) -> None:
        d = self.config.decay
        c = self.config
        self.confidence += (c.baseline_confidence - self.confidence) * d
        self.urgency += (c.baseline_urgency - self.urgency) * d
        self.frustration_proxy += (c.baseline_frustration - self.frustration_proxy) * d
        self.rapport += (c.baseline_rapport - self.rapport) * d

    def _publish(self, reason: str, before: AffectSnapshot) -> AffectSnapshot:
        after = self.snapshot()
        self.history.append(after)
        self.bus.publish(
            Topics.AFFECT_UPDATED,
            {
                "reason": reason,
                "before": before.model_dump(mode="json", exclude={"timestamp"}),
                "after": after.model_dump(mode="json", exclude={"timestamp"}),
            },
            source="affect",
        )
        return after

    # ------------------------------------------------------------------ #
    # Update hooks called by the orchestrator at defined points of a cycle
    # ------------------------------------------------------------------ #
    def on_percept(self, percept: Percept) -> AffectSnapshot:
        before = self.snapshot()
        self._decay()
        self.urgency = _clamp(max(self.urgency, percept.urgency) if percept.urgency > 0 else self.urgency)
        if percept.sentiment < 0:
            self.frustration_proxy = _clamp(self.frustration_proxy + 0.25 * -percept.sentiment)
            self.rapport = _clamp(self.rapport - 0.08 * -percept.sentiment)
        elif percept.sentiment > 0:
            self.frustration_proxy = _clamp(self.frustration_proxy - 0.15 * percept.sentiment)
            self.rapport = _clamp(self.rapport + 0.08 * percept.sentiment)
        if "feedback_negative" in percept.intents:
            self.frustration_proxy = _clamp(self.frustration_proxy + 0.2)
            self.confidence = _clamp(self.confidence - 0.15)
        if "feedback_positive" in percept.intents:
            self.rapport = _clamp(self.rapport + 0.1)
            self.confidence = _clamp(self.confidence + 0.05)
        if percept.confidence < 0.7:  # noisy STT / OCR
            self.confidence = _clamp(self.confidence - 0.1)
        return self._publish("percept", before)

    def on_retrieval(self, top_score: float, count: int) -> AffectSnapshot:
        before = self.snapshot()
        if count == 0:
            self.confidence = _clamp(self.confidence - 0.05)
        else:
            self.confidence = _clamp(self.confidence + 0.15 * top_score)
        return self._publish("retrieval", before)

    def on_action_result(self, ok: bool, replanned: bool = False) -> AffectSnapshot:
        before = self.snapshot()
        if ok:
            self.confidence = _clamp(self.confidence + 0.08)
            self.frustration_proxy = _clamp(self.frustration_proxy - 0.05)
        else:
            self.confidence = _clamp(self.confidence - 0.12)
            self.frustration_proxy = _clamp(self.frustration_proxy + (0.15 if replanned else 0.08))
        return self._publish("action_result", before)

    def on_cycle_end(self, success: Optional[bool]) -> AffectSnapshot:
        before = self.snapshot()
        if success is True:
            self.rapport = _clamp(self.rapport + 0.03)
            self.urgency = _clamp(self.urgency * 0.5)
        elif success is False:
            self.rapport = _clamp(self.rapport - 0.03)
        return self._publish("cycle_end", before)

    # ------------------------------------------------------------------ #
    # Biases exposed to other modules (pure functions of the state)
    # ------------------------------------------------------------------ #
    def risk_tolerance(self) -> float:
        """0..1. Lower when frustrated or unsure -> reasoning prefers safer options."""
        return _clamp(0.5 + 0.4 * (self.confidence - 0.5) - self.config.approval_threshold_boost * self.frustration_proxy * 2)

    def prefer_fast_path(self) -> bool:
        """System 1 is acceptable when urgent and we are confident; never when frustrated."""
        return self.urgency > 0.5 and self.confidence > 0.5 and self.frustration_proxy < 0.5

    def style(self) -> dict[str, str | float]:
        """Communication biases consumed by the language module."""
        if self.frustration_proxy > 0.55:
            tone = "calm_and_concise"
        elif self.rapport > 0.7:
            tone = "warm"
        elif self.rapport < 0.3:
            tone = "neutral_professional"
        else:
            tone = "friendly"
        verbosity = "brief" if self.urgency > 0.5 else ("normal" if self.urgency > 0.2 else "thorough")
        hedging = "high" if self.confidence < 0.4 else ("some" if self.confidence < 0.65 else "low")
        return {"tone": tone, "verbosity": verbosity, "hedging": hedging, "rapport": round(self.rapport, 2)}
