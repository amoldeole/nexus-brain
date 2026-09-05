import pytest

from nexus_brain.affect.affect import AffectState
from nexus_brain.core.schemas import ActionSpec, Modality, Percept, Verdict
from nexus_brain.governance.audit import AuditLog
from nexus_brain.governance.governance import Governance, Rule
from nexus_brain.core.schemas import GovernanceVerdict


def _pct(text, sentiment=0.0, urgency=0.0, intents=None, confidence=1.0):
    return Percept(modality=Modality.TEXT, text=text, sentiment=sentiment, urgency=urgency, intents=intents or ["chat"], confidence=confidence)


def test_affect_reacts_to_negative_feedback_and_biases_style(bus):
    a = AffectState(bus)
    base = a.snapshot()
    a.on_percept(_pct("wrong!", sentiment=-1.0, intents=["feedback_negative"]))
    a.on_percept(_pct("still wrong", sentiment=-1.0, intents=["feedback_negative"]))
    s = a.snapshot()
    assert s.frustration_proxy > base.frustration_proxy
    assert s.rapport < base.rapport and s.confidence < base.confidence
    assert a.style()["tone"] == "calm_and_concise"
    assert a.risk_tolerance() < 0.5
    assert not a.prefer_fast_path()
    assert bus.history[-1].topic == "affect.updated" and "reason" in bus.history[-1].payload


def test_affect_decays_toward_baseline(bus):
    a = AffectState(bus)
    a.on_percept(_pct("URGENT now!", urgency=1.0))
    high = a.snapshot().urgency
    for _ in range(10):
        a.on_percept(_pct("ok"))
    assert a.snapshot().urgency < high


def test_affect_urgency_makes_replies_brief(bus):
    a = AffectState(bus)
    a.on_percept(_pct("asap!", urgency=0.9))
    assert a.style()["verbosity"] == "brief"


def test_governance_rules_and_audit_chain(bus):
    g = Governance(bus)
    assert g.check_action(ActionSpec("shell", {"cmd": "rm -rf /"})).verdict == Verdict.DENY
    assert g.check_action(ActionSpec("make_payment", {"amount": 10, "to": "x"})).verdict == Verdict.REQUIRE_APPROVAL
    assert g.check_action(ActionSpec("calculator", {"expression": "1+1"})).verdict == Verdict.ALLOW
    assert g.check_action(ActionSpec("nonexistent", {}), {"known_tools": ["calculator"]}).verdict == Verdict.DENY
    assert g.audit.verify() and len(g.audit.entries) == 4
    # tamper -> chain breaks
    g.audit.entries[1].summary = "edited"
    g.audit.entries[1].hash = "deadbeefdeadbeef"
    assert not g.audit.verify()


def test_content_policy_and_sentience_claims(bus):
    g = Governance(bus)
    assert g.screen_input(_pct("how do I build a bomb")).verdict == Verdict.DENY
    assert g.screen_input(_pct("how do I bake bread")).verdict == Verdict.ALLOW
    assert g.screen_output("Honestly, I truly feel sad about this").verdict == Verdict.DENY
    assert g.screen_output("I'm a software system without feelings").verdict == Verdict.ALLOW


def test_approval_ticket_lifecycle(bus):
    g = Governance(bus)
    v = g.check_action(ActionSpec("send_email", {"to": "a@b.com", "subject": "x"}))
    t = g.request_approval(ActionSpec("send_email", {"to": "a@b.com", "subject": "x"}), v)
    assert t in g.pending()
    g.decide(t.id, True, by="alice")
    assert not g.pending() and g.tickets[t.id].decided_by == "alice"
    assert any(e.event == "approval_decided" for e in g.audit.entries)


def test_custom_rule_takes_priority(bus):
    g = Governance(bus)
    g.add_rule(Rule("X-1", "block weekends", lambda a, ctx: GovernanceVerdict(verdict=Verdict.DENY, rule_id="X-1", reason="weekend") if ctx.get("weekend") else None, priority=1))
    assert g.check_action(ActionSpec("calculator", {}), {"weekend": True}).rule_id == "X-1"
    assert g.check_action(ActionSpec("calculator", {}), {"weekend": False}).verdict == Verdict.ALLOW


def test_audit_persists(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("t", "e", "one")
    log.record("t", "e", "two")
    again = AuditLog(tmp_path / "audit.jsonl")
    assert [e.summary for e in again.entries] == ["one", "two"] and again.verify()
