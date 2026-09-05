"""Shared internal representations used by every Nexus Brain module.

The schemas are the "lingua franca" between faculties: perception produces
``Percept``s, memory stores ``MemoryItem``s, reasoning produces ``Decision``s
and ``Plan``s, the executor produces ``ActionResult``s, and the learning loop
consumes ``Outcome``s and produces ``Lesson``s / ``Proposal``s.

Everything here is a plain pydantic model so it can be serialised to JSON for
persistence, audit logs, and the API.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def new_id(prefix: str = "") -> str:
    """Return a short, unique identifier with an optional readable prefix."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Perception
# --------------------------------------------------------------------------- #
class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    STRUCTURED = "structured"


class Percept(BaseModel):
    """Normalised representation of one unit of input, regardless of modality.

    ``text`` is the canonical textual rendering that downstream faculties
    reason over; ``embedding`` is the vector used for retrieval; the cue
    fields (``sentiment``, ``urgency``) are *shallow* signals consumed by the
    affect module, not deep understanding.
    """

    id: str = Field(default_factory=lambda: new_id("pct"))
    modality: Modality
    text: str
    embedding: Optional[list[float]] = None
    entities: list[str] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    sentiment: float = 0.0  # -1 .. 1
    urgency: float = 0.0  # 0 .. 1
    confidence: float = 1.0  # perception confidence (e.g. STT / OCR quality)
    source: str = "user"
    session_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class MemoryKind(str, Enum):
    EPISODIC = "episodic"  # timestamped events / interactions
    SEMANTIC = "semantic"  # facts, concepts, user preferences
    PROCEDURAL = "procedural"  # skills / workflows ("how to do X")


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    kind: MemoryKind
    content: str
    embedding: Optional[list[float]] = None
    importance: float = 0.5  # 0..1, set at write time, adjusted by reflection
    confidence: float = 1.0  # for semantic facts: how sure are we
    created_at: datetime = Field(default_factory=utcnow)
    last_accessed: datetime = Field(default_factory=utcnow)
    access_count: int = 0
    tags: list[str] = Field(default_factory=list)
    source: str = "system"  # provenance: user / reflection / tool / seed
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    item: MemoryItem
    relevance: float
    recency: float
    importance: float
    score: float


class ActionSpec(BaseModel):
    """A single tool invocation the executor knows how to run."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    def __init__(self, tool: str | None = None, args: dict[str, Any] | None = None, description: str | None = None, **data: Any) -> None:
        # Allow the terse positional form ActionSpec("tool", {...}, "desc").
        if tool is not None:
            data["tool"] = tool
        if args is not None:
            data["args"] = args
        if description is not None:
            data["description"] = description
        super().__init__(**data)


class Skill(BaseModel):
    """Procedural memory entry: a reusable workflow with a track record."""

    name: str
    description: str
    triggers: list[str]  # keywords / phrases that suggest this skill applies
    steps: list[ActionSpec]
    successes: int = 0
    failures: int = 0
    risk_level: str = "low"  # low | medium | high
    version: int = 1

    @property
    def attempts(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        # Laplace-smoothed so brand-new skills are neither trusted nor damned.
        return (self.successes + 1) / (self.attempts + 2)


# --------------------------------------------------------------------------- #
# Affect (simulated, see docs/HONESTY.md)
# --------------------------------------------------------------------------- #
class AffectSnapshot(BaseModel):
    """Four scalar control variables in [0, 1].

    These are *behavioural tuning knobs*, not feelings. They bias tone,
    pacing, and risk tolerance. See ``nexus_brain.affect``.
    """

    confidence: float = 0.6
    urgency: float = 0.2
    frustration_proxy: float = 0.1
    rapport: float = 0.5
    timestamp: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Reasoning / planning
# --------------------------------------------------------------------------- #
class ReasoningMode(str, Enum):
    SYSTEM1 = "system1"  # fast, heuristic, skill-driven
    SYSTEM2 = "system2"  # slow, deliberate, option-weighing


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"
    DENIED = "denied"


class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    action: ActionSpec
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None


class Plan(BaseModel):
    id: str = Field(default_factory=lambda: new_id("plan"))
    goal: str
    sub_goals: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str = ""
    mode: ReasoningMode = ReasoningMode.SYSTEM2
    skill_name: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class Option(BaseModel):
    """A candidate course of action considered by System 2."""

    id: str = Field(default_factory=lambda: new_id("opt"))
    description: str
    plan: Plan
    expected_value: float = 0.5  # 0..1 how well it satisfies the goal
    risk: float = 0.1  # 0..1 predicted harm / failure probability
    feasibility: float = 1.0  # 0..1 can we actually do it with current tools
    past_outcome_prior: float = 0.5  # 0..1 from episodic memory of similar attempts
    score: float = 0.0
    requires_approval: bool = False  # contains a step a human must approve
    blocked_by: Optional[str] = None  # governance rule id, if filtered out


class Decision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dec"))
    goal: str
    mode: ReasoningMode
    options: list[Option] = Field(default_factory=list)
    chosen: Option
    rationale: str
    confidence: float
    constraints_applied: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Action execution
# --------------------------------------------------------------------------- #
class ActionResult(BaseModel):
    step_id: str
    tool: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    attempts: int = 1
    duration_ms: float = 0.0
    governance: Optional[str] = None  # allow | require_approval | deny


class ExecutionReport(BaseModel):
    plan_id: str
    results: list[ActionResult] = Field(default_factory=list)
    replans: int = 0
    completed: bool = False
    awaiting_approval: list[str] = Field(default_factory=list)  # approval ticket ids
    summary: str = ""

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok and r.governance != "require_approval")


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class GovernanceVerdict(BaseModel):
    verdict: Verdict
    rule_id: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW


class ApprovalTicket(BaseModel):
    id: str = Field(default_factory=lambda: new_id("apr"))
    action: ActionSpec
    rule_id: str
    reason: str
    requested_at: datetime = Field(default_factory=utcnow)
    status: str = "pending"  # pending | approved | rejected
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    cycle_id: Optional[str] = None


class AuditEntry(BaseModel):
    seq: int
    timestamp: datetime
    actor: str  # module name
    event: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str
    hash: str


# --------------------------------------------------------------------------- #
# Learning loop
# --------------------------------------------------------------------------- #
class Outcome(BaseModel):
    """One logged cognitive cycle, the raw material for reflection."""

    cycle_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    request: str
    intents: list[str] = Field(default_factory=list)
    mode: ReasoningMode
    skill_name: Optional[str] = None
    tools_used: list[str] = Field(default_factory=list)
    actions_ok: int = 0
    actions_failed: int = 0
    replans: int = 0
    response: str = ""
    affect: AffectSnapshot = Field(default_factory=AffectSnapshot)
    memories_retrieved: int = 0
    top_retrieval_score: float = 0.0
    user_feedback: Optional[float] = None  # -1..1, explicit or inferred
    success: Optional[bool] = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposalKind(str, Enum):
    MEMORY_WRITE = "memory_write"  # low risk, bounded, auto-applied
    SKILL_STATS = "skill_stats"  # low risk, bounded, auto-applied
    RETRIEVAL_WEIGHT = "retrieval_weight"  # low risk within bounds, auto-applied
    PROMPT_UPDATE = "prompt_update"  # requires human approval via UpgradeManager
    CONFIG_CHANGE = "config_change"  # requires human approval via UpgradeManager
    MODEL_CHECKPOINT = "model_checkpoint"  # requires human approval via UpgradeManager


class Proposal(BaseModel):
    id: str = Field(default_factory=lambda: new_id("prop"))
    kind: ProposalKind
    title: str
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)  # cycle ids
    confidence: float = 0.5
    requires_approval: bool = True
    status: str = "proposed"  # proposed | applied | pending_eval | pending_approval | promoted | rejected
    created_at: datetime = Field(default_factory=utcnow)


class Lesson(BaseModel):
    id: str = Field(default_factory=lambda: new_id("les"))
    text: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    proposals: list[Proposal] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Tracing (the "day in the life" record of a cycle)
# --------------------------------------------------------------------------- #
class TraceStage(BaseModel):
    stage: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float = 0.0


class CycleResult(BaseModel):
    cycle_id: str
    response: str
    decision: Optional[Decision] = None
    execution: Optional[ExecutionReport] = None
    affect: AffectSnapshot
    percept: Percept
    trace: list[TraceStage] = Field(default_factory=list)
    refused: bool = False
    pending_approvals: list[str] = Field(default_factory=list)
