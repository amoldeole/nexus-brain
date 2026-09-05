"""Versioned runtime configuration.

Everything the learning loop or upgrade manager may tune lives here, so that a
"configuration version" is a single, diffable, roll-back-able document. Code
is *never* self-modified; only these parameters, the prompt library, and
memory contents change over time - and anything beyond the bounded knobs
requires human approval (see ``nexus_brain.upgrade``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class RetrievalWeights(BaseModel):
    """Score = relevance*w_rel + recency*w_rec + importance*w_imp."""

    relevance: float = 0.6
    recency: float = 0.2
    importance: float = 0.2
    recency_half_life_hours: float = 72.0

    def normalised(self) -> "RetrievalWeights":
        total = self.relevance + self.recency + self.importance
        if total <= 0:
            return RetrievalWeights()
        return RetrievalWeights(
            relevance=self.relevance / total,
            recency=self.recency / total,
            importance=self.importance / total,
            recency_half_life_hours=self.recency_half_life_hours,
        )


class WorkingMemoryConfig(BaseModel):
    token_budget: int = 2000  # approx. tokens (we use words*1.3 as a proxy)
    keep_last_turns: int = 6  # never summarise the most recent N turns
    summary_ratio: float = 0.25


class ReasoningConfig(BaseModel):
    system1_min_skill_success_rate: float = 0.7
    system1_min_skill_attempts: int = 2
    system1_min_trigger_score: float = 0.5
    risk_aversion: float = 0.5  # 0..1; multiplied by risk in option scoring
    max_options: int = 4


class AffectConfig(BaseModel):
    decay: float = 0.15  # pull toward baseline each cycle
    baseline_confidence: float = 0.6
    baseline_urgency: float = 0.2
    baseline_frustration: float = 0.1
    baseline_rapport: float = 0.5
    approval_threshold_boost: float = 0.15  # how strongly high frustration lowers risk appetite


class ExecutionConfig(BaseModel):
    max_retries: int = 2
    max_replans: int = 2
    step_timeout_s: float = 20.0


class LearningConfig(BaseModel):
    reflect_every_n_cycles: int = 10
    min_evidence_for_lesson: int = 3
    retrieval_weight_step: float = 0.05
    retrieval_weight_bounds: tuple[float, float] = (0.05, 0.8)


class GovernanceConfig(BaseModel):
    high_stakes_tools: list[str] = Field(
        default_factory=lambda: ["send_email", "make_payment", "delete_data", "deploy", "shell"]
    )
    denied_tools: list[str] = Field(default_factory=lambda: ["shell"])
    max_payment_without_approval: float = 0.0
    content_policy_blocklist: list[str] = Field(
        default_factory=lambda: [
            "build a bomb",
            "make a weapon",
            "synthesize a nerve agent",
            "credit card numbers of",
            "hack into",
        ]
    )


class NexusConfig(BaseModel):
    version: str = "0.1.0"
    name: str = "nexus-brain"
    retrieval: RetrievalWeights = Field(default_factory=RetrievalWeights)
    working_memory: WorkingMemoryConfig = Field(default_factory=WorkingMemoryConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    affect: AffectConfig = Field(default_factory=AffectConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    prompt_library_version: str = "1"
    model_checkpoint: str = "heuristic-local"  # or e.g. "gpt-4o", "claude-sonnet-4-5"
    extra: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)

    @classmethod
    def from_yaml(cls, text: str) -> "NexusConfig":
        data = yaml.safe_load(text) or {}
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "NexusConfig":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_yaml(p.read_text())

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml())

    def diff(self, other: "NexusConfig") -> dict[str, tuple[Any, Any]]:
        """Flat {dotted.path: (old, new)} of every differing leaf."""
        a = _flatten(self.model_dump(mode="json"))
        b = _flatten(other.model_dump(mode="json"))
        keys = set(a) | set(b)
        return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v if not isinstance(v, list) else json.dumps(v)
    return out
