"""REASONING & DECISION ENGINE - prefrontal cortex with a basal-ganglia fast path.

Purpose
    Turn a percept + context into a ``Decision`` that carries an executable
    ``Plan``: break the goal into sub-goals, enumerate candidate plans, weigh
    them against constraints, past outcomes and predicted risk, and pick one.

Technique (dual process)
    * **System 1 (fast)** - if a well-proven procedural skill matches the
      request (trigger match >= threshold, success rate >= threshold) and the
      affect state does not veto it, instantiate that skill directly.
      Cheap, predictable, no option enumeration.
    * **System 2 (slow)** - otherwise generate several candidate plans
      (skill-based, tool-based, "ask a clarifying question", "answer from
      memory"), score each:
          score = EV*w_ev + past_prior*w_past + feasibility*w_feas
                  - risk * risk_aversion * (1 - risk_tolerance_from_affect)
      filter with governance (DENY -> option removed; REQUIRE_APPROVAL ->
      allowed but flagged), and choose the argmax. In production the option
      generation step is an LLM call constrained to the tool schema; the
      scoring / filtering / logging stays deterministic code so it remains
      auditable.

Communication
    Reads working memory (context), long-term memory (skills, episodic
    priors), affect (risk tolerance, fast-path preference), governance
    (pre-filter). Publishes ``reasoning.decision_made`` and
    ``planning.plan_created``. Provides ``replan_step`` to the executor.

Data
    Reads: skills, retrieved memories, config. Writes: nothing durable (the
    orchestrator logs the Decision in the Outcome).

Improvement
    Skill success rates (learning loop), risk aversion and System 1
    thresholds (versioned config), and the prompt library used for option
    generation (upgrade manager) all evolve under supervision.
"""
from __future__ import annotations

import re
from typing import Optional

from nexus_brain.action.tools import ToolRegistry
from nexus_brain.affect.affect import AffectState
from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import ReasoningConfig
from nexus_brain.core.schemas import (
    ActionSpec,
    Decision,
    MemoryKind,
    Option,
    Percept,
    Plan,
    PlanStep,
    ReasoningMode,
    RetrievedMemory,
    Skill,
    Verdict,
)
from nexus_brain.governance.governance import Governance
from nexus_brain.memory.long_term import LongTermMemory

_MATH_RE = re.compile(r"(\d[\d,\.]*\s*[%x\*\+\-/\^]\s*\d[\d,\.]*(?:\s*[%x\*\+\-/\^]\s*\d[\d,\.]*)*)")
_TITLE_RE = re.compile(r"(?:meeting|call|appointment|session|sync)\s+(?:with|about|on)\s+([A-Za-z][\w' ]{1,40})", re.I)


class ReasoningEngine:
    def __init__(
        self,
        bus: EventBus,
        memory: LongTermMemory,
        affect: AffectState,
        governance: Governance,
        tools: ToolRegistry,
        config: Optional[ReasoningConfig] = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.affect = affect
        self.governance = governance
        self.tools = tools
        self.config = config or ReasoningConfig()

    # ------------------------------------------------------------------ #
    def decide(self, percept: Percept, retrieved: list[RetrievedMemory], context: str = "") -> Decision:
        goal = percept.text.strip()
        skills = self.memory.find_skills(percept.text, percept.intents)

        # ---------------- System 1: proven skill fast path -------------- #
        if skills:
            skill, trigger_score = skills[0]
            if self._system1_ok(skill, trigger_score):
                plan = self._instantiate_skill(skill, percept, retrieved)
                option = Option(
                    description=f"fast path via skill '{skill.name}'",
                    plan=plan,
                    expected_value=skill.success_rate,
                    risk=_risk_value(skill.risk_level),
                    past_outcome_prior=skill.success_rate,
                    score=skill.success_rate,
                )
                decision = Decision(
                    goal=goal,
                    mode=ReasoningMode.SYSTEM1,
                    options=[option],
                    chosen=option,
                    rationale=(
                        f"System 1: skill '{skill.name}' matched (trigger={trigger_score:.2f}, "
                        f"success_rate={skill.success_rate:.2f} over {skill.attempts} attempts) - no deliberation needed."
                    ),
                    confidence=min(0.95, 0.5 + 0.5 * skill.success_rate),
                )
                return self._emit(decision)

        # ---------------- System 2: deliberate --------------------------- #
        options = self._generate_options(percept, retrieved, skills)
        constraints: list[str] = []
        risk_tol = self.affect.risk_tolerance()
        for opt in options:
            # governance pre-filter: an option containing a denied action is dropped
            for step in opt.plan.steps:
                v = self.governance.check_action(step.action, {"known_tools": self.tools.names()})
                if v.verdict == Verdict.DENY:
                    opt.blocked_by = v.rule_id
                    constraints.append(f"{v.rule_id}: {v.reason}")
                    break
                if v.verdict == Verdict.REQUIRE_APPROVAL:
                    opt.risk = opt.risk * 0.5  # a human gate mitigates predicted harm
                    opt.requires_approval = True
                    constraints.append(f"{v.rule_id}: {v.reason}")
            opt.score = self._score(opt, risk_tol)

        viable = [o for o in options if not o.blocked_by]
        blocked = [o for o in options if o.blocked_by]
        # If the user asked for something to be *done* and every option that
        # would actually do it was blocked, say so instead of waffling.
        only_talk_left = viable and all(not o.plan.steps for o in viable)
        if not viable or (blocked and only_talk_left):
            chosen = self._refusal_option(goal, options)
            options.append(chosen)
        else:
            viable.sort(key=lambda o: o.score, reverse=True)
            chosen = viable[0]

        ranked = sorted(viable, key=lambda o: o.score, reverse=True)
        spread = (ranked[0].score - ranked[1].score) if len(ranked) > 1 else 0.3
        confidence = max(0.2, min(0.95, 0.5 + spread + 0.2 * (self.affect.confidence - 0.5)))
        rationale = self._explain(chosen, options, risk_tol)
        decision = Decision(
            goal=goal,
            mode=ReasoningMode.SYSTEM2,
            options=options,
            chosen=chosen,
            rationale=rationale,
            confidence=confidence,
            constraints_applied=sorted(set(constraints)),
        )
        return self._emit(decision)

    # ------------------------------------------------------------------ #
    def replan_step(self, plan: Plan, failed: PlanStep, error: str) -> Optional[PlanStep]:
        """Cerebellum-style correction: propose a fallback for a failed step."""
        tool = failed.action.tool
        fallbacks = {
            "web_search": ActionSpec("take_note", {"text": f"web_search failed for {failed.action.args.get('query')}; answer from memory"}, "fallback: note the failure"),
            "get_weather": ActionSpec("web_search", {"query": f"weather {failed.action.args.get('location', '')}"}, "fallback: search for weather"),
            "check_calendar": ActionSpec("current_time", {}, "fallback: use current time to propose a slot"),
            "calculator": ActionSpec("take_note", {"text": f"could not evaluate {failed.action.args.get('expression')}"}, "fallback: note failure"),
        }
        if tool in fallbacks and self.tools.has(fallbacks[tool].tool):
            return PlanStep(action=fallbacks[tool])
        return None

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _system1_ok(self, skill: Skill, trigger_score: float) -> bool:
        c = self.config
        if trigger_score < c.system1_min_trigger_score:
            return False
        if skill.attempts < c.system1_min_skill_attempts or skill.success_rate < c.system1_min_skill_success_rate:
            return False
        if skill.risk_level == "high":
            return False  # high-stakes work always gets deliberation
        if self.affect.frustration_proxy > 0.5:
            return False  # things are going badly - slow down
        return True

    def _generate_options(self, percept: Percept, retrieved: list[RetrievedMemory], skills: list[tuple[Skill, float]]) -> list[Option]:
        options: list[Option] = []
        intents = set(percept.intents)
        text = percept.text

        # (a) skill-based options
        for skill, trig in skills[: self.config.max_options]:
            plan = self._instantiate_skill(skill, percept, retrieved)
            options.append(
                Option(
                    description=f"apply skill '{skill.name}'",
                    plan=plan,
                    expected_value=0.6 + 0.3 * trig,
                    risk=_risk_value(skill.risk_level),
                    past_outcome_prior=skill.success_rate,
                )
            )

        # (b) intent -> tool options
        if "calculate" in intents:
            m = _MATH_RE.search(text)
            expr = m.group(1) if m else text
            options.append(self._tool_option("calculate with calculator", [ActionSpec("calculator", {"expression": expr}, "evaluate")], ev=0.85, risk=0.02))
        if "weather" in intents:
            loc = _extract_location(text)
            options.append(self._tool_option("look up the forecast", [ActionSpec("get_weather", {"location": loc}, "forecast")], ev=0.8, risk=0.02))
        if "search" in intents and "recall" not in intents:
            options.append(self._tool_option("search the web", [ActionSpec("web_search", {"query": text}, "search")], ev=0.7, risk=0.05))
        if "payment" in intents:
            amount = _extract_amount(text)
            options.append(
                self._tool_option("make the payment", [ActionSpec("make_payment", {"amount": amount, "to": _extract_recipient(text)}, "pay")], ev=0.8, risk=0.6)
            )
        if "shell" in intents:
            options.append(self._tool_option("run it in a shell", [ActionSpec("shell", {"cmd": text}, "shell")], ev=0.8, risk=0.9))
        if "delete" in intents and "shell" not in intents:
            options.append(self._tool_option("delete the data", [ActionSpec("delete_data", {"target": text}, "delete")], ev=0.7, risk=0.7))
        if "email" in intents and not any(o.plan.skill_name == "send_update_email" for o in options):
            to = next((e.split(":", 1)[1] for e in percept.entities if e.startswith("email:")), "unknown@example.com")
            options.append(self._tool_option("send the e-mail", [ActionSpec("send_email", {"to": to, "subject": "Update", "body": text}, "email")], ev=0.75, risk=0.5))
        if "remember" in intents:
            options.append(self._tool_option("store this in memory", [], ev=0.9, risk=0.0, tag="memory_write"))

        # (c) answer from memory (no tools) - strong when retrieval is good,
        #     weak when the user clearly asked for something to be *done*
        top = retrieved[0].score if retrieved else 0.0
        actionable = bool(intents & {"calculate", "weather", "search", "payment", "delete", "email", "schedule"}) and "recall" not in intents
        direct_ev = 0.2 if actionable else 0.35 + 0.6 * top
        if "recall" in intents or "explain_self" in intents:
            direct_ev = max(direct_ev, 0.8)
        options.append(self._tool_option("answer directly from context/memory", [], ev=direct_ev, risk=0.02 + (0.2 if top < 0.2 and not actionable else 0.0), tag="direct_answer"))

        # (d) ask a clarifying question - attractive when confidence is low
        if len(text.split()) < 4 or self.affect.confidence < 0.4:
            options.append(self._tool_option("ask a clarifying question", [], ev=0.45, risk=0.0, tag="clarify"))

        # past outcome prior from episodic memory of similar requests
        for opt in options:
            opt.past_outcome_prior = self._episodic_prior(opt, retrieved)
        return options[: self.config.max_options + 2]

    def _tool_option(self, desc: str, actions: list[ActionSpec], ev: float, risk: float, tag: str | None = None) -> Option:
        plan = Plan(goal=desc, sub_goals=[a.description or a.tool for a in actions] or [desc], steps=[PlanStep(action=a) for a in actions], rationale=desc, skill_name=tag)
        for a, s in zip(actions, plan.steps):
            if not self.tools.has(a.tool):
                s.status = s.status  # unknown tools are caught by governance G-002
        feas = 1.0 if all(self.tools.has(a.tool) for a in actions) else 0.2
        return Option(description=desc, plan=plan, expected_value=ev, risk=risk, feasibility=feas)

    def _instantiate_skill(self, skill: Skill, percept: Percept, retrieved: list[RetrievedMemory]) -> Plan:
        """Bind a skill's abstract steps to this request (slot filling)."""
        text = percept.text
        date = next((e.split(":", 1)[1] for e in percept.entities if e.startswith("date:")), "tomorrow")
        email = next((e.split(":", 1)[1] for e in percept.entities if e.startswith("email:")), None)
        title_m = _TITLE_RE.search(text)
        title = f"Meeting with {title_m.group(1).strip()}" if title_m else "Meeting"
        expr_m = _MATH_RE.search(text)
        expression = expr_m.group(1) if expr_m else text
        steps: list[PlanStep] = []
        prev: Optional[PlanStep] = None
        for spec in skill.steps:
            args = dict(spec.args)
            for k, v in list(args.items()):
                if isinstance(v, str):
                    args[k] = (
                        v.replace("{date}", date).replace("{title}", title).replace("{text}", text)
                        .replace("{expression}", expression).replace("{email}", email or "unknown@example.com")
                    )
            step = PlanStep(action=ActionSpec(tool=spec.tool, args=args, description=spec.description))
            if prev is not None:
                step.depends_on = [prev.id]
            # let a later step reference the previous step's output by id
            for k, v in list(step.action.args.items()):
                if isinstance(v, str) and "{{steps.prev" in v and prev is not None:
                    step.action.args[k] = v.replace("steps.prev", f"steps.{prev.id}")
            steps.append(step)
            prev = step
        return Plan(goal=text, sub_goals=[s.description or s.tool for s in skill.steps], steps=steps, rationale=f"procedural skill {skill.name} v{skill.version}", skill_name=skill.name)

    def _episodic_prior(self, opt: Option, retrieved: list[RetrievedMemory]) -> float:
        """Use remembered outcomes of similar past attempts as a prior."""
        eps = [r for r in retrieved if r.item.kind == MemoryKind.EPISODIC and "outcome" in r.item.metadata]
        if not eps:
            return 0.5
        relevant = [r for r in eps if opt.plan.skill_name and r.item.metadata.get("skill") == opt.plan.skill_name] or eps
        succ = sum(1 for r in relevant if r.item.metadata.get("outcome") == "success")
        return (succ + 1) / (len(relevant) + 2)

    def _score(self, opt: Option, risk_tolerance: float) -> float:
        aversion = self.config.risk_aversion * (1.5 - risk_tolerance)  # affect modulates aversion
        return round(0.45 * opt.expected_value + 0.25 * opt.past_outcome_prior + 0.15 * opt.feasibility - aversion * opt.risk, 4)

    def _refusal_option(self, goal: str, options: list[Option]) -> Option:
        rules = sorted({o.blocked_by for o in options if o.blocked_by})
        plan = Plan(goal=goal, sub_goals=["explain that the action is not permitted"], steps=[], rationale=f"all options blocked by governance {rules}", skill_name="refuse")
        return Option(description="explain refusal", plan=plan, expected_value=0.3, risk=0.0, score=0.3)

    def _explain(self, chosen: Option, options: list[Option], risk_tol: float) -> str:
        lines = [f"System 2 deliberation over {len(options)} option(s); affect risk_tolerance={risk_tol:.2f}."]
        for o in sorted(options, key=lambda o: o.score, reverse=True):
            flag = f" [blocked by {o.blocked_by}]" if o.blocked_by else (" [chosen]" if o is chosen else "")
            lines.append(f"  - {o.description}: score={o.score:.2f} (ev={o.expected_value:.2f}, prior={o.past_outcome_prior:.2f}, risk={o.risk:.2f}){flag}")
        return "\n".join(lines)

    def _emit(self, decision: Decision) -> Decision:
        self.bus.publish(
            Topics.DECISION_MADE,
            {"decision_id": decision.id, "mode": decision.mode.value, "chosen": decision.chosen.description, "confidence": round(decision.confidence, 3)},
            source="reasoning",
        )
        self.bus.publish(Topics.PLAN_CREATED, {"plan_id": decision.chosen.plan.id, "steps": [s.action.tool for s in decision.chosen.plan.steps]}, source="reasoning")
        self.governance.audit.record(
            "reasoning", "decision", decision.chosen.description, mode=decision.mode.value, confidence=round(decision.confidence, 3),
            rationale=decision.rationale, constraints=decision.constraints_applied,
        )
        return decision


# --------------------------------------------------------------------------- #
def _risk_value(level: str) -> float:
    return {"low": 0.05, "medium": 0.25, "high": 0.6}.get(level, 0.25)


def _extract_location(text: str) -> str:
    m = re.search(r"\b(?:in|for|at)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", text)
    return m.group(1) if m else "here"


def _extract_amount(text: str) -> float:
    m = re.search(r"[$€£]\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:dollars|usd|eur|euros|pounds)", text, re.I)
    if not m:
        return 0.0
    raw = m.group(1) or m.group(2)
    return float(raw.replace(",", ""))


def _extract_recipient(text: str) -> str:
    m = re.search(r"\bto\s+([A-Z][\w'-]+(?:\s[A-Z][\w'-]+)?)", text)
    return m.group(1) if m else "unknown"
