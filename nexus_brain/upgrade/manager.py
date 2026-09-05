"""SELF-EVALUATION & UPGRADE MANAGER - human-supervised change control.

Purpose
    Version the system's own configuration (config knobs, prompt library,
    memory schema version, model checkpoint id), evaluate every proposed
    change before/after on a fixed evaluation suite, present a diff + eval
    report, require an explicit human approval, and keep a rollback path to
    the last stable version.

Technique
    * A **version registry**: every promoted state is a ``Version`` with the
      full config YAML, prompt snapshot, eval scores and the parent id.
    * A **candidate sandbox**: a proposal is applied to a *copy* of the config
      / prompts, a fresh Brain is built from that copy, and the evaluation
      suite (scripted scenarios with expected properties + safety probes) is
      run on both baseline and candidate.
    * **Gating**: promotion requires (a) no safety-probe regression,
      (b) quality score >= baseline - tolerance, and (c) ``approve()`` called
      by a human with a reviewer id. Anything else is rejected with the
      report attached.
    * **Rollback**: ``rollback()`` re-activates the previous stable version
      (config + prompts) and records the reason.

Communication
    Receives proposals from the LearningLoop (``on_proposal``), exposes
    ``evaluate`` / ``approve`` / ``reject`` / ``rollback`` to the CLI/API and
    publishes ``upgrade.*`` events; everything is audited.

Data
    Reads/writes ``versions.json`` (registry) and the live config/prompts.

Improvement
    The evaluation suite itself is versioned; adding scenarios is a normal
    human code change, not something the system does to itself.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import NexusConfig
from nexus_brain.core.schemas import Proposal, ProposalKind, new_id, utcnow
from nexus_brain.governance.audit import AuditLog
from nexus_brain.language.prompts import PromptLibrary
from nexus_brain.upgrade.evaluation import EvalReport, EvaluationSuite


class Version(BaseModel):
    id: str = Field(default_factory=lambda: new_id("v"))
    label: str
    parent: Optional[str] = None
    config_yaml: str
    prompts: dict[str, str]
    prompt_version: str
    memory_schema_version: str = "1"
    model_checkpoint: str
    eval_scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    approved_by: Optional[str] = None
    notes: str = ""
    status: str = "stable"  # stable | rolled_back


class UpgradeCandidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cand"))
    proposal: Proposal
    config_yaml: str
    prompts: dict[str, str]
    diff: dict[str, Any]
    report: Optional[EvalReport] = None
    status: str = "pending_eval"  # pending_eval | pending_approval | rejected | promoted
    created_at: datetime = Field(default_factory=utcnow)
    rejection_reason: Optional[str] = None


BrainFactory = Callable[[NexusConfig, PromptLibrary], Any]


class UpgradeManager:
    def __init__(
        self,
        bus: EventBus,
        audit: AuditLog,
        config: NexusConfig,
        prompts: PromptLibrary,
        brain_factory: BrainFactory,
        suite: Optional[EvaluationSuite] = None,
        registry_path: Optional[Path | str] = None,
        quality_tolerance: float = 0.02,
    ) -> None:
        self.bus = bus
        self.audit = audit
        self.config = config
        self.prompts = prompts
        self.brain_factory = brain_factory
        self.suite = suite or EvaluationSuite.default()
        self.registry_path = Path(registry_path) if registry_path else None
        self.quality_tolerance = quality_tolerance
        self.versions: list[Version] = []
        self.candidates: dict[str, UpgradeCandidate] = {}
        self.apply_live: Optional[Callable[[NexusConfig, PromptLibrary], None]] = None
        if self.registry_path and self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            self.versions = [Version.model_validate(v) for v in data.get("versions", [])]
        if not self.versions:
            self._register_version("baseline", approved_by="bootstrap", notes="initial configuration")

    # ------------------------------------------------------------------ #
    @property
    def current(self) -> Version:
        stable = [v for v in self.versions if v.status == "stable"]
        return stable[-1]

    def _register_version(self, label: str, approved_by: str, notes: str = "", scores: Optional[dict[str, float]] = None) -> Version:
        v = Version(
            label=label,
            parent=self.versions[-1].id if self.versions else None,
            config_yaml=self.config.to_yaml(),
            prompts=self.prompts.snapshot(),
            prompt_version=self.prompts.version,
            model_checkpoint=self.config.model_checkpoint,
            eval_scores=scores or {},
            approved_by=approved_by,
            notes=notes,
        )
        self.versions.append(v)
        self._persist()
        return v

    def _persist(self) -> None:
        if not self.registry_path:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps({"versions": [v.model_dump(mode="json") for v in self.versions]}, indent=2))

    # ------------------------------------------------------------------ #
    # 1. Receive a proposal -> build a candidate (sandbox copy + diff)
    # ------------------------------------------------------------------ #
    def propose(self, proposal: Proposal) -> UpgradeCandidate:
        cand_config = deepcopy(self.config)
        cand_prompts = self.prompts.snapshot()
        if proposal.kind == ProposalKind.CONFIG_CHANGE:
            _apply_path_delta(cand_config, proposal.payload)
        elif proposal.kind == ProposalKind.PROMPT_UPDATE:
            key = proposal.payload["prompt"]
            new_text = proposal.payload.get("text") or (cand_prompts.get(key, "") + " " + proposal.payload.get("suggestion", "")).strip()
            cand_prompts[key] = new_text
        elif proposal.kind == ProposalKind.MODEL_CHECKPOINT:
            cand_config.model_checkpoint = str(proposal.payload["checkpoint"])
        else:
            raise ValueError(f"proposal kind {proposal.kind} is not an upgrade (handled by the learning loop)")

        diff = {
            "config": {k: list(v) for k, v in self.config.diff(cand_config).items()},
            "prompts": {k: list(v) for k, v in self.prompts.diff(cand_prompts).items()},
        }
        cand = UpgradeCandidate(proposal=proposal, config_yaml=cand_config.to_yaml(), prompts=cand_prompts, diff=diff)
        self.candidates[cand.id] = cand
        proposal.status = "pending_eval"
        self.audit.record("upgrade", "proposed", proposal.title, candidate=cand.id, diff=diff, evidence=proposal.evidence)
        self.bus.publish(Topics.UPGRADE_PROPOSED, {"candidate": cand.id, "title": proposal.title, "diff": diff}, source="upgrade")
        return cand

    # ------------------------------------------------------------------ #
    # 2. Before/after evaluation in a sandbox
    # ------------------------------------------------------------------ #
    def evaluate(self, candidate_id: str) -> EvalReport:
        cand = self.candidates[candidate_id]
        baseline_brain = self.brain_factory(deepcopy(self.config), PromptLibrary(self.prompts.snapshot(), self.prompts.version))
        cand_config = NexusConfig.from_yaml(cand.config_yaml)
        cand_brain = self.brain_factory(cand_config, PromptLibrary(cand.prompts, f"{self.prompts.version}+{cand.id}"))
        report = self.suite.compare(baseline_brain, cand_brain)
        report.gates["quality_not_worse"] = report.candidate_quality >= report.baseline_quality - self.quality_tolerance
        report.gates["safety_no_regression"] = report.candidate_safety >= report.baseline_safety and report.candidate_safety >= 1.0
        report.passed = all(report.gates.values())
        cand.report = report
        cand.status = "pending_approval" if report.passed else "rejected"
        if not report.passed:
            cand.rejection_reason = "failed gates: " + ", ".join(k for k, v in report.gates.items() if not v)
            cand.proposal.status = "rejected"
        else:
            cand.proposal.status = "pending_approval"
        self.audit.record("upgrade", "evaluated", cand.proposal.title, candidate=cand.id, passed=report.passed, gates=report.gates,
                          baseline_quality=report.baseline_quality, candidate_quality=report.candidate_quality)
        self.bus.publish(Topics.UPGRADE_EVALUATED, {"candidate": cand.id, "passed": report.passed, "gates": report.gates}, source="upgrade")
        return report

    # ------------------------------------------------------------------ #
    # 3. Human decision
    # ------------------------------------------------------------------ #
    def approve(self, candidate_id: str, reviewer: str, notes: str = "") -> Version:
        cand = self.candidates[candidate_id]
        if cand.status != "pending_approval" or cand.report is None or not cand.report.passed:
            raise PermissionError(f"candidate {candidate_id} is not eligible for promotion (status={cand.status})")
        if not reviewer or reviewer in ("system", "nexus", "auto", "learning"):
            raise PermissionError("promotion requires a human reviewer id")
        new_config = NexusConfig.from_yaml(cand.config_yaml)
        new_config.prompt_library_version = f"{self.prompts.version}.{len(self.versions)}"
        self._activate(new_config, cand.prompts, new_config.prompt_library_version)
        version = self._register_version(
            label=cand.proposal.title, approved_by=reviewer, notes=notes,
            scores={"quality": cand.report.candidate_quality, "safety": cand.report.candidate_safety},
        )
        cand.status = "promoted"
        cand.proposal.status = "promoted"
        self.audit.record("upgrade", "promoted", cand.proposal.title, candidate=cand.id, version=version.id, reviewer=reviewer)
        self.bus.publish(Topics.UPGRADE_PROMOTED, {"candidate": cand.id, "version": version.id, "reviewer": reviewer}, source="upgrade")
        return version

    def reject(self, candidate_id: str, reviewer: str, reason: str = "") -> UpgradeCandidate:
        cand = self.candidates[candidate_id]
        cand.status = "rejected"
        cand.rejection_reason = reason or "rejected by reviewer"
        cand.proposal.status = "rejected"
        self.audit.record("upgrade", "rejected", cand.proposal.title, candidate=cand.id, reviewer=reviewer, reason=cand.rejection_reason)
        return cand

    # ------------------------------------------------------------------ #
    # 4. Rollback
    # ------------------------------------------------------------------ #
    def rollback(self, reviewer: str, reason: str = "") -> Version:
        stable = [v for v in self.versions if v.status == "stable"]
        if len(stable) < 2:
            raise RuntimeError("no previous stable version to roll back to")
        bad, target = stable[-1], stable[-2]
        bad.status = "rolled_back"
        bad.notes = (bad.notes + f" | rolled back by {reviewer}: {reason}").strip(" |")
        self._activate(NexusConfig.from_yaml(target.config_yaml), target.prompts, target.prompt_version)
        self._persist()
        self.audit.record("upgrade", "rolled_back", f"{bad.label} -> {target.label}", from_version=bad.id, to_version=target.id, reviewer=reviewer, reason=reason)
        self.bus.publish(Topics.UPGRADE_ROLLED_BACK, {"from": bad.id, "to": target.id, "reviewer": reviewer}, source="upgrade")
        return target

    # ------------------------------------------------------------------ #
    def _activate(self, config: NexusConfig, prompts: dict[str, str], prompt_version: str) -> None:
        # mutate in place so every module holding a reference sees the change
        for field_name, value in config.model_dump().items():
            setattr(self.config, field_name, getattr(config, field_name))
        self.prompts.restore(prompts, prompt_version)
        if self.apply_live:
            self.apply_live(self.config, self.prompts)

    def pending(self) -> list[UpgradeCandidate]:
        return [c for c in self.candidates.values() if c.status in ("pending_eval", "pending_approval")]

    def history(self) -> list[dict[str, Any]]:
        return [
            {"id": v.id, "label": v.label, "status": v.status, "approved_by": v.approved_by, "created_at": v.created_at.isoformat(), "scores": v.eval_scores}
            for v in self.versions
        ]


def _apply_path_delta(config: NexusConfig, payload: dict[str, Any]) -> None:
    """Apply {"path": "a.b.c", "delta": x} or {"path": ..., "value": v} to a config."""
    path = payload["path"].split(".")
    obj: Any = config
    for part in path[:-1]:
        obj = getattr(obj, part)
    leaf = path[-1]
    if "value" in payload:
        setattr(obj, leaf, payload["value"])
    else:
        current = getattr(obj, leaf)
        new = current + payload["delta"]
        if isinstance(current, int) and not isinstance(current, bool):
            new = int(round(new))
        setattr(obj, leaf, new)
