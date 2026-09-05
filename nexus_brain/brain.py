"""The Brain - orchestrator for one cognitive cycle.

    perceive -> screen(input) -> recall -> affect -> reason/plan -> govern
            -> act -> speak -> screen(output) -> remember -> learn

Every stage appends a ``TraceStage`` so a single ``CycleResult`` *is* the
"day in the life" record of that request.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from nexus_brain.action.executor import Executor
from nexus_brain.action.tools import MockWorld, ToolRegistry, build_default_registry
from nexus_brain.affect.affect import AffectState
from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import NexusConfig
from nexus_brain.core.schemas import (
    ActionSpec,
    CycleResult,
    Decision,
    ExecutionReport,
    MemoryKind,
    Modality,
    Outcome,
    Percept,
    Plan,
    Skill,
    TraceStage,
    new_id,
)
from nexus_brain.governance.audit import AuditLog
from nexus_brain.governance.governance import Governance
from nexus_brain.language.language import LanguageModule
from nexus_brain.language.llm import LLM, TemplateLLM
from nexus_brain.language.prompts import PromptLibrary
from nexus_brain.learning.reflection import LearningLoop
from nexus_brain.memory.long_term import LongTermMemory
from nexus_brain.memory.store import InMemoryVectorStore
from nexus_brain.memory.working import Turn, WorkingMemory
from nexus_brain.perception.embedder import Embedder, HashingEmbedder
from nexus_brain.perception.perception import Perception
from nexus_brain.upgrade.manager import UpgradeManager


def default_skills() -> list[Skill]:
    return [
        Skill(
            name="schedule_meeting",
            description="Check the calendar, create the event, confirm",
            triggers=["schedule", "set up a meeting", "arrange a meeting", "book a meeting", "book a call"],
            steps=[
                ActionSpec("check_calendar", {"date": "{date}"}, "check availability"),
                ActionSpec("create_event", {"title": "{title}", "date": "{date}", "time": "{{steps.prev.output.free_slots.0}}"}, "create the event"),
            ],
            successes=2,
            failures=0,
            risk_level="medium",
        ),
        Skill(
            name="quick_math",
            description="Evaluate an arithmetic expression",
            triggers=["calculate", "compute"],
            steps=[ActionSpec("calculator", {"expression": "{expression}"}, "evaluate")],
            successes=0,
            failures=0,
            risk_level="low",
        ),
        Skill(
            name="send_update_email",
            description="Send a short status e-mail",
            triggers=["email", "send an update", "send a status"],
            steps=[ActionSpec("send_email", {"to": "{email}", "subject": "Update", "body": "{text}"}, "send e-mail")],
            risk_level="high",
        ),
    ]


class Brain:
    def __init__(
        self,
        config: Optional[NexusConfig] = None,
        data_dir: Optional[Path | str] = None,
        embedder: Optional[Embedder] = None,
        tools: Optional[ToolRegistry] = None,
        world: Optional[MockWorld] = None,
        prompts: Optional[PromptLibrary] = None,
        llm: Optional[LLM] = None,
        seed_skills: bool = True,
        enable_upgrade_manager: bool = True,
    ) -> None:
        self.config = config or NexusConfig()
        self.data_dir = Path(data_dir) if data_dir else None
        p = (lambda name: self.data_dir / name) if self.data_dir else (lambda name: None)  # type: ignore[misc]

        self.bus = EventBus()
        self.audit = AuditLog(p("audit.jsonl"))
        self.embedder = embedder or HashingEmbedder()
        self.perception = Perception(self.bus, self.embedder)
        self.ltm = LongTermMemory(self.bus, self.embedder, InMemoryVectorStore(p("memory.json")), self.config.retrieval, skills_path=p("skills.json"))
        self.wm = WorkingMemory(self.bus, self.config.working_memory, consolidate=self._consolidate_turn)
        self.affect = AffectState(self.bus, self.config.affect)
        self.governance = Governance(self.bus, self.config.governance, self.audit)
        if tools is None:
            tools, world = build_default_registry(world)
        self.tools = tools
        self.world = world
        self.prompts = prompts if prompts is not None else PromptLibrary(path=p("prompts.json"))
        self.language = LanguageModule(self.bus, self.prompts, self.affect, llm or TemplateLLM())
        from nexus_brain.reasoning.engine import ReasoningEngine  # local import keeps module graph acyclic

        self.reasoning = ReasoningEngine(self.bus, self.ltm, self.affect, self.governance, self.tools, self.config.reasoning)
        self.executor = Executor(self.bus, self.tools, self.governance, self.config.execution, replanner=self.reasoning.replan_step)
        self.upgrade: Optional[UpgradeManager] = None
        self.learning = LearningLoop(self.bus, self.ltm, self.audit, self.config.learning, log_path=p("outcomes.jsonl"), on_proposal=self._on_proposal)
        if enable_upgrade_manager:
            self.upgrade = UpgradeManager(self.bus, self.audit, self.config, self.prompts, brain_factory=self._sandbox_factory, registry_path=p("versions.json"))
            self.upgrade.apply_live = self._apply_live_config

        if seed_skills and not self.ltm.skills:
            for s in default_skills():
                self.ltm.add_skill(s)

        self.session_id = new_id("sess")
        self.user_profile: dict[str, str] = {}
        self.pending_plans: dict[str, tuple[Plan, str]] = {}  # ticket id -> (plan, cycle id)
        self.cycles = 0
        self.last_result: Optional[CycleResult] = None
        self._restore_profile()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def new_session(self) -> str:
        self.session_id = new_id("sess")
        self.wm.clear()
        self.affect.reset()
        return self.session_id

    def process(self, payload: Any, modality: Modality | str = Modality.TEXT, user_feedback: Optional[float] = None) -> CycleResult:
        """Run one full cognitive cycle and return the response + trace."""
        t_cycle = time.perf_counter()
        cycle_id = new_id("cyc")
        trace: list[TraceStage] = []

        def stage(name: str, summary: str, t0: float, **details: Any) -> None:
            trace.append(TraceStage(stage=name, summary=summary, details=details, elapsed_ms=round((time.perf_counter() - t0) * 1000, 3)))

        # 1. PERCEPTION ---------------------------------------------------
        t0 = time.perf_counter()
        percept = self.perception.perceive(payload, modality, session_id=self.session_id)
        stage("perception", f"{percept.modality.value}: intents={percept.intents} entities={percept.entities} sentiment={percept.sentiment:+.2f} urgency={percept.urgency:.2f}", t0,
              percept_id=percept.id, text=percept.text[:200])

        # approval replies are handled before anything else
        handled = self._maybe_handle_approval_reply(percept)
        if handled is not None:
            return self._finish_cycle(cycle_id, percept, handled, None, None, [], trace, t_cycle, refused=False)

        # 2. GOVERNANCE: input screen -------------------------------------
        t0 = time.perf_counter()
        verdict = self.governance.screen_input(percept)
        stage("governance.input", f"{verdict.verdict.value} ({verdict.rule_id}): {verdict.reason}", t0)
        self.wm.add_turn("user", percept.text)
        if not verdict.allowed:
            t0 = time.perf_counter()
            self.affect.on_percept(percept)
            response = self.language.respond(percept, None, None, [], self.user_profile, refusal_reason=verdict.reason)
            stage("language", "refusal rendered", t0, response=response)
            return self._finish_cycle(cycle_id, percept, response, None, None, [], trace, t_cycle, refused=True)

        # 3. MEMORY: recall -----------------------------------------------
        t0 = time.perf_counter()
        retrieved = self.ltm.retrieve(percept.text, k=5, query_embedding=percept.embedding)
        self.wm.set_retrieved(retrieved)
        self.wm.set_goal(percept.text)
        self._update_profile_from_percept(percept)
        stage("memory.recall", f"{len(retrieved)} memories (top={retrieved[0].score if retrieved else 0:.2f}); wm={self.wm.used_tokens} tokens",
              t0, memories=[{"kind": r.item.kind.value, "score": round(r.score, 3), "content": r.item.content[:100]} for r in retrieved])

        # 4. AFFECT -------------------------------------------------------
        t0 = time.perf_counter()
        self.affect.on_percept(percept)
        snap = self.affect.on_retrieval(retrieved[0].score if retrieved else 0.0, len(retrieved))
        stage("affect", f"conf={snap.confidence:.2f} urg={snap.urgency:.2f} frust={snap.frustration_proxy:.2f} rapport={snap.rapport:.2f} -> style={self.affect.style()}", t0,
              **snap.model_dump(mode="json", exclude={"timestamp"}), risk_tolerance=round(self.affect.risk_tolerance(), 3))

        # 5. REASONING & PLANNING -----------------------------------------
        t0 = time.perf_counter()
        decision = self.reasoning.decide(percept, retrieved, self.wm.context_window())
        plan = decision.chosen.plan
        stage("reasoning", f"{decision.mode.value}: '{decision.chosen.description}' (confidence={decision.confidence:.2f}, {len(plan.steps)} step(s))", t0,
              rationale=decision.rationale, constraints=decision.constraints_applied, plan=[s.action.tool for s in plan.steps])

        # 6. ACTION -------------------------------------------------------
        execution: Optional[ExecutionReport] = None
        approval_tickets: list[tuple[str, str, str]] = []
        if plan.steps:
            t0 = time.perf_counter()
            execution = self.executor.execute(plan, cycle_id=cycle_id, on_result=lambda r: self.affect.on_action_result(r.ok and r.governance == "allow") if r.governance == "allow" else None)
            for r in execution.results:
                if r.governance == "require_approval" and r.error:
                    ticket = self.governance.tickets[r.error]
                    approval_tickets.append((ticket.id, f"run {ticket.action.tool} {ticket.action.args}", ticket.reason))
                    self.pending_plans[ticket.id] = (plan, cycle_id)
            stage("action", f"{execution.summary}; ok={execution.succeeded} failed={execution.failed} replans={execution.replans} approvals={len(approval_tickets)}", t0,
                  results=[{"tool": r.tool, "ok": r.ok, "governance": r.governance, "error": r.error, "attempts": r.attempts} for r in execution.results])
        elif plan.skill_name == "refuse":
            pass

        # 7. LANGUAGE -----------------------------------------------------
        t0 = time.perf_counter()
        refusal = None
        if plan.skill_name == "refuse":
            refusal = "; ".join(decision.constraints_applied) or "the action is not permitted"
        response = self.language.respond(percept, decision, execution, retrieved, self.user_profile, self.wm.context_window(), refusal_reason=refusal, approval_tickets=approval_tickets)
        out_verdict = self.governance.screen_output(response)
        if not out_verdict.allowed:
            response = self.prompts.get("self_description") if out_verdict.rule_id == "C-002" else self.prompts.get("refusal", reason=out_verdict.reason)
        stage("language", f"style={self.affect.style()['tone']}/{self.affect.style()['verbosity']} output_screen={out_verdict.verdict.value}", t0, response=response)

        return self._finish_cycle(cycle_id, percept, response, decision, execution, approval_tickets, trace, t_cycle, refused=bool(refusal), user_feedback=user_feedback)

    # ------------------------------------------------------------------ #
    def approve(self, ticket_id: str, approve: bool = True, by: str = "human") -> CycleResult:
        """Human decision on a pending high-stakes action; resumes the plan."""
        ticket = self.governance.decide(ticket_id, approve, by=by)
        plan, cycle_id = self.pending_plans.pop(ticket_id, (None, None))
        percept = Percept(modality=Modality.TEXT, text=f"[approval:{ticket.status}] {ticket_id}", intents=["approval"], session_id=self.session_id)
        trace: list[TraceStage] = []
        t_cycle = time.perf_counter()
        execution = None
        if plan is not None:
            t0 = time.perf_counter()
            execution = self.executor.resume_after_approval(plan, ticket.action, approve, cycle_id=cycle_id)
            trace.append(TraceStage(stage="action.resume", summary=execution.summary, details={"approved": approve, "by": by}, elapsed_ms=(time.perf_counter() - t0) * 1000))
            for r in execution.results:
                self.affect.on_action_result(r.ok)
        if approve and execution is not None and execution.results:
            response = "Approved - " + self.language._describe_execution(execution, None)  # type: ignore[arg-type]
        elif approve:
            response = "Approved, but I could not find the pending plan to resume."
        else:
            response = "Understood - I've cancelled that action."
        self.ltm.remember_episode(f"Human {'approved' if approve else 'rejected'} {ticket.action.tool} {ticket.action.args}", importance=0.7,
                                  tags=["approval", ticket.action.tool], metadata={"outcome": "success" if approve else "rejected", "tool": ticket.action.tool})
        return self._finish_cycle(cycle_id or new_id("cyc"), percept, response, None, execution, [], trace, t_cycle, refused=False)

    def reflect(self) -> list:
        return self.learning.reflect()

    def sleep(self) -> dict[str, int]:
        """Offline maintenance: reflect, decay, forget."""
        lessons = self.learning.reflect()
        decayed = self.ltm.decay()
        forgotten = self.ltm.forget()
        self.audit.record("brain", "sleep", f"lessons={len(lessons)} decayed={decayed} forgotten={forgotten}")
        return {"lessons": len(lessons), "decayed": decayed, "forgotten": forgotten}

    def status(self) -> dict[str, Any]:
        return {
            "config_version": self.config.version,
            "config_fingerprint": self.config.fingerprint(),
            "prompt_version": self.prompts.version,
            "model_checkpoint": self.config.model_checkpoint,
            "llm_backend": getattr(self.language.llm, "name", "?"),
            "cycles": self.cycles,
            "memory": self.ltm.stats(),
            "working_memory_tokens": self.wm.used_tokens,
            "affect": self.affect.snapshot().model_dump(mode="json", exclude={"timestamp"}),
            "retrieval_weights": self.ltm.weights.model_dump(),
            "pending_approvals": [t.id for t in self.governance.pending()],
            "pending_upgrades": [c.id for c in self.upgrade.pending()] if self.upgrade else [],
            "current_version": self.upgrade.current.id if self.upgrade else None,
            "audit_entries": len(self.audit.entries),
            "audit_chain_valid": self.audit.verify(),
            "metrics": self.learning.metrics(),
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _finish_cycle(self, cycle_id: str, percept: Percept, response: str, decision: Optional[Decision], execution: Optional[ExecutionReport],
                      approval_tickets: list[tuple[str, str, str]], trace: list[TraceStage], t_cycle: float, refused: bool,
                      user_feedback: Optional[float] = None) -> CycleResult:
        # 8. MEMORY: remember the episode ------------------------------------
        t0 = time.perf_counter()
        self.wm.add_turn("assistant", response)
        success: Optional[bool] = None
        if refused:
            success = None  # a correct refusal is neither a task success nor failure
        elif execution is not None:
            success = execution.completed if not approval_tickets else None
        elif decision is not None and decision.chosen.plan.skill_name in ("direct_answer", "memory_write"):
            success = True
        facts = self._extract_facts(percept)
        for fact in facts:
            self.ltm.remember_fact(fact, importance=0.8, source="user", tags=["user_stated"])
        episode_text = f"User said: '{percept.text[:160]}'. I {'refused' if refused else 'responded'}: '{response[:160]}'"
        if execution is not None:
            episode_text += f" (actions: {execution.summary})"
        importance = 0.4 + 0.2 * abs(percept.sentiment) + (0.2 if execution and execution.results else 0.0) + (0.1 if refused else 0.0)
        self.ltm.remember_episode(episode_text, importance=min(1.0, importance), tags=list(percept.intents),
                                  metadata={"cycle_id": cycle_id, "outcome": "success" if success else ("failure" if success is False else "n/a"),
                                            "skill": decision.chosen.plan.skill_name if decision else None, "facts": facts})
        trace.append(TraceStage(stage="memory.store", summary=f"episode stored (importance={min(1.0, importance):.2f}); {len(facts)} fact(s) extracted", details={"facts": facts}, elapsed_ms=(time.perf_counter() - t0) * 1000))

        # 9. LEARNING: log outcome ------------------------------------------
        t0 = time.perf_counter()
        affect_snapshot = self.affect.on_cycle_end(success)
        if user_feedback is None:
            if "feedback_negative" in percept.intents:
                user_feedback = -1.0
            elif "feedback_positive" in percept.intents:
                user_feedback = 1.0
        outcome = Outcome(
            cycle_id=cycle_id, request=percept.text, intents=percept.intents, mode=decision.mode if decision else self._mode_none(),
            skill_name=decision.chosen.plan.skill_name if decision else None,
            tools_used=[r.tool for r in execution.results] if execution else [],
            actions_ok=execution.succeeded if execution else 0, actions_failed=execution.failed if execution else 0,
            replans=execution.replans if execution else 0, response=response, affect=affect_snapshot,
            memories_retrieved=len(self.wm.retrieved), top_retrieval_score=self.wm.retrieved[0].score if self.wm.retrieved else 0.0,
            user_feedback=user_feedback, success=success, latency_ms=round((time.perf_counter() - t_cycle) * 1000, 3),
            metadata={"failed_tools": [r.tool for r in execution.results if not r.ok and r.governance == "allow"] if execution else [], "tone": self.affect.style()["tone"], "refused": refused},
        )
        lessons = self.learning.log_outcome(outcome)
        trace.append(TraceStage(stage="learning", summary=f"outcome logged (success={success}); " + (f"reflection ran -> {len(lessons)} lesson(s)" if lessons is not None else f"reflection in {self.config.learning.reflect_every_n_cycles - self.learning._since_reflection} cycle(s)"),
                                details={"lessons": [l.text for l in lessons] if lessons else []}, elapsed_ms=(time.perf_counter() - t0) * 1000))

        self.cycles += 1
        result = CycleResult(cycle_id=cycle_id, response=response, decision=decision, execution=execution, affect=affect_snapshot, percept=percept, trace=trace,
                             refused=refused, pending_approvals=[t[0] for t in approval_tickets])
        self.last_result = result
        self.bus.publish(Topics.CYCLE_COMPLETE, {"cycle_id": cycle_id, "success": success, "latency_ms": outcome.latency_ms}, source="brain")
        return result

    @staticmethod
    def _mode_none():
        from nexus_brain.core.schemas import ReasoningMode

        return ReasoningMode.SYSTEM1

    def _maybe_handle_approval_reply(self, percept: Percept) -> Optional[str]:
        low = percept.text.strip().lower()
        for verb, approve in (("approve", True), ("reject", False)):
            if low.startswith(verb + " "):
                ticket_id = percept.text.strip().split(None, 1)[1].strip()
                if ticket_id in self.governance.tickets and self.governance.tickets[ticket_id].status == "pending":
                    res = self.approve(ticket_id, approve, by="user")
                    return res.response
                return f"I couldn't find a pending approval with id '{ticket_id}'."
        return None

    def _extract_facts(self, percept: Percept) -> list[str]:
        facts: list[str] = []
        text = percept.text.strip()
        low = text.lower()
        for ent in percept.entities:
            if ent.startswith("person:"):
                facts.append(f"User's name is {ent.split(':', 1)[1]}")
        for marker in ("i prefer ", "i like ", "i don't like ", "i always ", "i never ", "my timezone is ", "i work at ", "i live in ", "call me "):
            if marker in low:
                idx = low.index(marker)
                clause = text[idx:].split(".")[0].split(" and ")[0].strip()
                if len(clause.split()) >= 3:
                    facts.append(f"User stated: {clause}")
        if "remember" in percept.intents and not facts and len(text.split()) > 3:
            body = text.split("remember", 1)[-1].lstrip(" :that") or text
            facts.append(f"User asked me to remember: {body.strip()}")
        return facts

    def _update_profile_from_percept(self, percept: Percept) -> None:
        for ent in percept.entities:
            if ent.startswith("person:"):
                self.user_profile["name"] = ent.split(":", 1)[1]

    def _restore_profile(self) -> None:
        for item in self.ltm.store.all([MemoryKind.SEMANTIC]):
            if item.content.startswith("User's name is "):
                self.user_profile["name"] = item.content[len("User's name is "):].strip(". ")

    def _consolidate_turn(self, turn: Turn) -> None:
        self.ltm.remember_episode(f"(from earlier in session) {turn.role}: {turn.content[:200]}", importance=turn.salience * 0.8, tags=["consolidated"])

    def _on_proposal(self, proposal) -> None:
        if self.upgrade is not None:
            self.upgrade.propose(proposal)

    def _sandbox_factory(self, config: NexusConfig, prompts: PromptLibrary) -> "Brain":
        """Build an isolated Brain (fresh memory, mock world) for evaluation."""
        return Brain(config=config, data_dir=None, embedder=self.embedder, prompts=prompts, llm=self.language.llm, enable_upgrade_manager=False)

    def _apply_live_config(self, config: NexusConfig, prompts: PromptLibrary) -> None:
        self.ltm.weights = config.retrieval
        self.wm.config = config.working_memory
        self.affect.config = config.affect
        self.reasoning.config = config.reasoning
        self.executor.config = config.execution
        self.learning.config = config.learning
        self.governance.config = config.governance
