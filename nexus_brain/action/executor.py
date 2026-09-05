"""PLANNING & ACTION EXECUTION - the motor system + cerebellum.

Purpose
    Turn a ``Plan`` into concrete tool calls, run them in dependency order,
    watch for failures, retry transient errors, ask the reasoning engine to
    re-plan on persistent failure, and pause at human-approval gates.

Technique
    Deterministic DAG walker with:
      * per-step governance check (deny / require approval / allow)
      * bounded retries with the cerebellum-like "error correction" of a
        replan callback that can substitute a fallback step
      * argument templating: ``{{steps.<id>.output.<key>}}`` lets a later step
        consume an earlier step's output (e.g. a free slot from the calendar)
      * a full ``ExecutionReport`` for the learning loop and audit log.

Communication
    Publishes ``action.started/finished/failed``, ``action.replan``. Calls
    Governance.check_action for every step; calls the optional ``replanner``
    (provided by the reasoning engine) when a step exhausts its retries.

Data
    Reads the ToolRegistry; writes nothing durable itself (memory + learning
    loop persist what matters, governance persists approvals).

Improvement
    Retry/replan limits are versioned config; skill success statistics fed
    back by the learning loop make the planner prefer procedures that work.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import ExecutionConfig
from nexus_brain.core.schemas import ActionResult, ActionSpec, ExecutionReport, Plan, PlanStep, StepStatus, Verdict
from nexus_brain.action.tools import ToolError, ToolRegistry
from nexus_brain.governance.governance import Governance

Replanner = Callable[[Plan, PlanStep, str], Optional[PlanStep]]

_TEMPLATE_RE = re.compile(r"\{\{\s*steps\.([\w-]+)\.output(?:\.([\w.]+))?\s*\}\}")


class Executor:
    def __init__(
        self,
        bus: EventBus,
        tools: ToolRegistry,
        governance: Governance,
        config: Optional[ExecutionConfig] = None,
        replanner: Optional[Replanner] = None,
    ) -> None:
        self.bus = bus
        self.tools = tools
        self.governance = governance
        self.config = config or ExecutionConfig()
        self.replanner = replanner

    # ------------------------------------------------------------------ #
    def execute(self, plan: Plan, cycle_id: Optional[str] = None, on_result: Optional[Callable[[ActionResult], None]] = None) -> ExecutionReport:
        report = ExecutionReport(plan_id=plan.id)
        outputs: dict[str, Any] = {}
        replans = 0
        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            i += 1
            if step.status in (StepStatus.DONE, StepStatus.SKIPPED):
                continue
            # dependency gate
            unmet = [d for d in step.depends_on if _status(plan, d) != StepStatus.DONE]
            if unmet:
                step.status = StepStatus.SKIPPED
                step.error = f"dependencies not satisfied: {unmet}"
                report.results.append(ActionResult(step_id=step.id, tool=step.action.tool, ok=False, error=step.error))
                continue

            step.action = self._render(step.action, outputs)
            result = self._run_step(step, cycle_id)
            report.results.append(result)
            if on_result:
                on_result(result)

            if result.ok:
                outputs[step.id] = result.output
                continue
            if result.governance == Verdict.REQUIRE_APPROVAL.value:
                report.awaiting_approval.append(result.error or "")
                # Downstream steps that depend on this one will be skipped.
                continue
            if result.governance == Verdict.DENY.value:
                continue

            # Persistent failure -> ask for a fallback step (re-plan), bounded.
            if self.replanner and replans < self.config.max_replans:
                fallback = self.replanner(plan, step, result.error or "unknown error")
                if fallback is not None:
                    replans += 1
                    fallback.depends_on = list(step.depends_on)
                    plan.steps.insert(i, fallback)
                    # re-point dependants at the fallback step
                    for later in plan.steps[i + 1 :]:
                        later.depends_on = [fallback.id if d == step.id else d for d in later.depends_on]
                    self.bus.publish(
                        Topics.REPLAN, {"failed_step": step.id, "fallback": fallback.action.tool, "reason": result.error}, source="executor"
                    )
                    self.governance.audit.record("executor", "replan", f"{step.action.tool} failed -> fallback {fallback.action.tool}", reason=result.error, cycle_id=cycle_id)

        report.replans = replans
        pending = [s for s in plan.steps if s.status == StepStatus.AWAITING_APPROVAL]
        failed = [s for s in plan.steps if s.status in (StepStatus.FAILED, StepStatus.DENIED)]
        report.completed = not pending and not failed
        report.summary = self._summarise(plan, report)
        return report

    # ------------------------------------------------------------------ #
    def resume_after_approval(self, plan: Plan, ticket_action: ActionSpec, approved: bool, cycle_id: Optional[str] = None) -> ExecutionReport:
        """Continue a plan whose step was waiting for a human decision."""
        for step in plan.steps:
            if step.status == StepStatus.AWAITING_APPROVAL and step.action.tool == ticket_action.tool and step.action.args == ticket_action.args:
                if approved:
                    step.status = StepStatus.PENDING
                    step.attempts = 0
                    step.error = None
                    return self._execute_from(plan, step, cycle_id, approved_override=True)
                step.status = StepStatus.DENIED
                step.error = "rejected by human reviewer"
                break
        report = ExecutionReport(plan_id=plan.id)
        report.completed = False
        report.summary = "Action was rejected by the reviewer; plan halted."
        return report

    def _execute_from(self, plan: Plan, start: PlanStep, cycle_id: Optional[str], approved_override: bool) -> ExecutionReport:
        report = ExecutionReport(plan_id=plan.id)
        outputs = {s.id: s.result for s in plan.steps if s.status == StepStatus.DONE}
        started = False
        for step in plan.steps:
            if step is start:
                started = True
            if not started or step.status in (StepStatus.DONE,):
                continue
            if step.status == StepStatus.SKIPPED:
                step.status = StepStatus.PENDING
            step.action = self._render(step.action, outputs)
            result = self._run_step(step, cycle_id, skip_governance=(step is start and approved_override))
            report.results.append(result)
            if result.ok:
                outputs[step.id] = result.output
        report.completed = all(s.status == StepStatus.DONE for s in plan.steps)
        report.summary = self._summarise(plan, report)
        return report

    # ------------------------------------------------------------------ #
    def _run_step(self, step: PlanStep, cycle_id: Optional[str], skip_governance: bool = False) -> ActionResult:
        action = step.action
        if not skip_governance:
            verdict = self.governance.check_action(action, {"known_tools": self.tools.names()})
            if verdict.verdict == Verdict.DENY:
                step.status = StepStatus.DENIED
                step.error = verdict.reason
                self.bus.publish(Topics.ACTION_FAILED, {"step": step.id, "tool": action.tool, "reason": verdict.reason}, source="executor")
                return ActionResult(step_id=step.id, tool=action.tool, ok=False, error=verdict.reason, governance=verdict.verdict.value)
            if verdict.verdict == Verdict.REQUIRE_APPROVAL:
                ticket = self.governance.request_approval(action, verdict, cycle_id=cycle_id)
                step.status = StepStatus.AWAITING_APPROVAL
                step.error = ticket.id
                return ActionResult(step_id=step.id, tool=action.tool, ok=False, error=ticket.id, governance=verdict.verdict.value)

        tool = self.tools.get(action.tool)
        step.status = StepStatus.RUNNING
        self.bus.publish(Topics.ACTION_STARTED, {"step": step.id, "tool": action.tool}, source="executor")
        last_err = ""
        t0 = time.perf_counter()
        max_attempts = 1 + (self.config.max_retries if tool.idempotent or step.attempts == 0 else 0)
        for attempt in range(1, max_attempts + 1):
            step.attempts += 1
            try:
                output = tool(**action.args)
                step.status = StepStatus.DONE
                step.result = output
                ms = (time.perf_counter() - t0) * 1000
                self.bus.publish(Topics.ACTION_FINISHED, {"step": step.id, "tool": action.tool, "attempts": attempt}, source="executor")
                self.governance.audit.record("executor", "action_done", f"{action.tool} ok", args=action.args, attempts=attempt, cycle_id=cycle_id)
                return ActionResult(step_id=step.id, tool=action.tool, ok=True, output=output, attempts=attempt, duration_ms=ms, governance="allow")
            except ToolError as e:  # recoverable -> retry
                last_err = str(e)
                if not tool.idempotent:
                    break
            except TypeError as e:  # bad arguments -> do not retry
                last_err = f"bad arguments: {e}"
                break
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                break
        step.status = StepStatus.FAILED
        step.error = last_err
        ms = (time.perf_counter() - t0) * 1000
        self.bus.publish(Topics.ACTION_FAILED, {"step": step.id, "tool": action.tool, "reason": last_err}, source="executor")
        self.governance.audit.record("executor", "action_failed", f"{action.tool} failed: {last_err}", args=action.args, attempts=step.attempts, cycle_id=cycle_id)
        return ActionResult(step_id=step.id, tool=action.tool, ok=False, error=last_err, attempts=step.attempts, duration_ms=ms, governance="allow")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _render(action: ActionSpec, outputs: dict[str, Any]) -> ActionSpec:
        def _resolve(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            m = _TEMPLATE_RE.fullmatch(value.strip())
            if m:
                return _dig(outputs.get(m.group(1)), m.group(2))
            return _TEMPLATE_RE.sub(lambda mm: str(_dig(outputs.get(mm.group(1)), mm.group(2))), value)

        return ActionSpec(tool=action.tool, args={k: _resolve(v) for k, v in action.args.items()}, description=action.description)

    @staticmethod
    def _summarise(plan: Plan, report: ExecutionReport) -> str:
        parts = []
        for s in plan.steps:
            parts.append(f"{s.action.tool}:{s.status.value}")
        return ", ".join(parts)


def _status(plan: Plan, step_id: str) -> StepStatus:
    for s in plan.steps:
        if s.id == step_id:
            return s.status
    return StepStatus.FAILED


def _dig(obj: Any, path: Optional[str]) -> Any:
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else None
        else:
            return None
    return cur
