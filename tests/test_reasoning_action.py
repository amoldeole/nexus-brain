from nexus_brain.action.executor import Executor
from nexus_brain.action.tools import build_default_registry
from nexus_brain.core.schemas import ActionSpec, Plan, PlanStep, ReasoningMode, StepStatus
from nexus_brain.governance.governance import Governance


def test_system1_fast_path_for_proven_skill(brain):
    r = brain.process("schedule a meeting with Dana tomorrow")
    assert r.decision.mode == ReasoningMode.SYSTEM1
    assert r.decision.chosen.plan.skill_name == "schedule_meeting"
    tools = [x.tool for x in r.execution.results]
    assert tools == ["check_calendar", "create_event"]
    # slot filled from the previous step's output via templating
    assert brain.world.calendar[0]["time"] == "09:00"


def test_system2_when_skill_unproven(brain):
    brain.ltm.skills["schedule_meeting"].successes = 0  # no track record -> deliberate
    r = brain.process("schedule a meeting with Dana tomorrow")
    assert r.decision.mode == ReasoningMode.SYSTEM2
    assert len(r.decision.options) >= 2
    assert "System 2" in r.decision.rationale


def test_frustration_vetoes_fast_path(brain):
    brain.affect.frustration_proxy = 0.9
    r = brain.process("schedule a meeting with Dana tomorrow")
    assert r.decision.mode == ReasoningMode.SYSTEM2


def test_governance_filters_options_and_refuses(brain):
    r = brain.process("run a shell command to wipe the disk")
    assert r.refused
    assert any(o.blocked_by == "G-001" for o in r.decision.options)
    assert not brain.world.calls or "shell" not in brain.world.calls


def test_risk_aversion_rises_with_low_risk_tolerance(brain):
    brain.affect.confidence = 0.9
    brain.affect.frustration_proxy = 0.0
    hi = brain.reasoning._score(_opt(0.8, 0.6), brain.affect.risk_tolerance())
    brain.affect.confidence = 0.2
    brain.affect.frustration_proxy = 0.9
    lo = brain.reasoning._score(_opt(0.8, 0.6), brain.affect.risk_tolerance())
    assert lo < hi


def _opt(ev, risk):
    from nexus_brain.core.schemas import Option

    return Option(description="x", plan=Plan(goal="g"), expected_value=ev, risk=risk)


def test_executor_retries_then_replans(bus):
    tools, world = build_default_registry()
    gov = Governance(bus)
    calls = {}

    def replanner(plan, step, err):
        calls["err"] = err
        return PlanStep(action=ActionSpec("web_search", {"query": "weather Berlin"}))

    ex = Executor(bus, tools, gov, replanner=replanner)
    world.force_failures("get_weather", 10)
    plan = Plan(goal="weather", steps=[PlanStep(action=ActionSpec("get_weather", {"location": "Berlin"}))])
    report = ex.execute(plan)
    assert report.replans == 1 and "transient" in calls["err"]
    assert [r.tool for r in report.results] == ["get_weather", "web_search"]
    assert report.results[0].attempts == 3  # 1 + 2 retries
    assert report.results[1].ok


def test_executor_dependency_and_templating(bus):
    tools, world = build_default_registry()
    ex = Executor(bus, tools, Governance(bus))
    a = PlanStep(action=ActionSpec("check_calendar", {"date": "today"}))
    b = PlanStep(action=ActionSpec("create_event", {"title": "T", "time": f"{{{{steps.{a.id}.output.free_slots.1}}}}"}), depends_on=[a.id])
    report = ex.execute(Plan(goal="g", steps=[a, b]))
    assert report.completed and world.calendar[0]["time"] == "11:00"


def test_executor_skips_dependants_of_failed_steps(bus):
    tools, world = build_default_registry()
    ex = Executor(bus, tools, Governance(bus))
    world.force_failures("check_calendar", 10)
    a = PlanStep(action=ActionSpec("check_calendar", {}))
    b = PlanStep(action=ActionSpec("create_event", {"title": "T"}), depends_on=[a.id])
    report = ex.execute(Plan(goal="g", steps=[a, b]))
    assert a.status == StepStatus.FAILED and b.status == StepStatus.SKIPPED and not report.completed
    assert world.calendar == []


def test_executor_pauses_for_approval_and_resumes(bus):
    tools, world = build_default_registry()
    gov = Governance(bus)
    ex = Executor(bus, tools, gov)
    step = PlanStep(action=ActionSpec("make_payment", {"amount": 50.0, "to": "Bob"}))
    plan = Plan(goal="pay", steps=[step])
    report = ex.execute(plan)
    assert step.status == StepStatus.AWAITING_APPROVAL and report.awaiting_approval and world.payments == []
    ticket = gov.pending()[0]
    gov.decide(ticket.id, True, by="alice")
    resumed = ex.resume_after_approval(plan, ticket.action, approved=True)
    assert resumed.completed and world.payments[0]["amount"] == 50.0


def test_calculator_is_safe():
    from nexus_brain.action.tools import safe_eval, ToolError
    import pytest

    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("15% x 200") == 30
    with pytest.raises((ToolError, SyntaxError, ValueError)):
        safe_eval("__import__('os').system('ls')")
