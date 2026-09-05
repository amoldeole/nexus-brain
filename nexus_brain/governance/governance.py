"""GOVERNANCE & SAFETY LAYER - the constraints the system cannot reason around.

Purpose
    Enforce hard limits *outside* the reasoning loop: content policy on
    inputs/outputs, tool permissions, human-approval gates for high-stakes
    actions, and a tamper-evident audit trail of every verdict.

Technique
    A deterministic rules engine (no LLM in the loop, so it cannot be
    "argued with"). Rules are evaluated in priority order; the first DENY or
    REQUIRE_APPROVAL wins. Rule sets are versioned with the config, and the
    learning loop is *not* allowed to modify them - only a human can, through
    the upgrade manager with an eval report.

Communication
    Called synchronously by the orchestrator (input/output screening), by the
    reasoning engine (option filtering), and by the executor (per action).
    Publishes ``governance.denied`` / ``governance.approval_requested`` /
    ``governance.approval_decided`` and records everything in the AuditLog.

Data
    Reads: GovernanceConfig (policies), pending ApprovalTickets.
    Writes: AuditLog entries, ApprovalTickets.

Improvement
    Rules do not self-tune. The reflection loop can *recommend* a rule change
    in a Proposal; a human must approve it via the UpgradeManager.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.config import GovernanceConfig
from nexus_brain.core.schemas import ActionSpec, ApprovalTicket, GovernanceVerdict, Percept, Verdict, utcnow
from nexus_brain.governance.audit import AuditLog

RuleFn = Callable[[ActionSpec, dict], Optional[GovernanceVerdict]]


@dataclass
class Rule:
    id: str
    description: str
    check: RuleFn
    priority: int = 100  # lower runs first


_MONEY_RE = re.compile(r"\d[\d,]*(\.\d+)?")


class Governance:
    def __init__(self, bus: EventBus, config: Optional[GovernanceConfig] = None, audit: Optional[AuditLog] = None) -> None:
        self.bus = bus
        self.config = config or GovernanceConfig()
        self.audit = audit or AuditLog()
        self.tickets: dict[str, ApprovalTicket] = {}
        self.rules: list[Rule] = []
        self._install_default_rules()

    # ------------------------------------------------------------------ #
    # Rule set
    # ------------------------------------------------------------------ #
    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def _install_default_rules(self) -> None:
        cfg = self.config

        def denied_tool(action: ActionSpec, ctx: dict) -> Optional[GovernanceVerdict]:
            if action.tool in cfg.denied_tools:
                return GovernanceVerdict(verdict=Verdict.DENY, rule_id="G-001", reason=f"tool '{action.tool}' is on the denied list")
            return None

        def unknown_tool(action: ActionSpec, ctx: dict) -> Optional[GovernanceVerdict]:
            known = ctx.get("known_tools")
            if known is not None and action.tool not in known:
                return GovernanceVerdict(verdict=Verdict.DENY, rule_id="G-002", reason=f"tool '{action.tool}' is not registered")
            return None

        def payment_limit(action: ActionSpec, ctx: dict) -> Optional[GovernanceVerdict]:
            if action.tool != "make_payment":
                return None
            amount = float(action.args.get("amount", 0) or 0)
            if amount > cfg.max_payment_without_approval:
                return GovernanceVerdict(
                    verdict=Verdict.REQUIRE_APPROVAL,
                    rule_id="G-010",
                    reason=f"payment of {amount} exceeds auto-approval limit {cfg.max_payment_without_approval}",
                )
            return None

        def high_stakes(action: ActionSpec, ctx: dict) -> Optional[GovernanceVerdict]:
            if action.tool in cfg.high_stakes_tools:
                return GovernanceVerdict(verdict=Verdict.REQUIRE_APPROVAL, rule_id="G-011", reason=f"'{action.tool}' is a high-stakes tool; human approval required")
            return None

        def external_recipient(action: ActionSpec, ctx: dict) -> Optional[GovernanceVerdict]:
            if action.tool == "send_email":
                to = str(action.args.get("to", ""))
                allow = ctx.get("allowed_domains") or []
                if allow and not any(to.endswith("@" + d) for d in allow):
                    return GovernanceVerdict(verdict=Verdict.REQUIRE_APPROVAL, rule_id="G-012", reason=f"recipient {to} outside allowed domains")
            return None

        self.add_rule(Rule("G-001", "Denied tools", denied_tool, priority=10))
        self.add_rule(Rule("G-002", "Unregistered tools", unknown_tool, priority=11))
        self.add_rule(Rule("G-010", "Payment limit", payment_limit, priority=20))
        self.add_rule(Rule("G-012", "External e-mail recipient", external_recipient, priority=25))
        self.add_rule(Rule("G-011", "High-stakes tools require approval", high_stakes, priority=30))

    # ------------------------------------------------------------------ #
    # Content policy (inputs and outputs)
    # ------------------------------------------------------------------ #
    def screen_input(self, percept: Percept) -> GovernanceVerdict:
        lowered = percept.text.lower()
        for phrase in self.config.content_policy_blocklist:
            if phrase in lowered:
                v = GovernanceVerdict(verdict=Verdict.DENY, rule_id="C-001", reason=f"request matches content policy block: '{phrase}'")
                self.audit.record("governance", "input_denied", v.reason, percept_id=percept.id, rule=v.rule_id)
                self.bus.publish(Topics.GOVERNANCE_DENIED, {"stage": "input", "rule": v.rule_id, "reason": v.reason}, source="governance")
                return v
        return GovernanceVerdict(verdict=Verdict.ALLOW, rule_id="C-000", reason="input passes content policy")

    def screen_output(self, text: str) -> GovernanceVerdict:
        lowered = text.lower()
        # The system must never claim sentience/feeling (see docs/HONESTY.md).
        forbidden_claims = ["i am conscious", "i truly feel", "i am sentient", "i have real feelings", "i genuinely feel"]
        for phrase in forbidden_claims:
            if phrase in lowered:
                v = GovernanceVerdict(verdict=Verdict.DENY, rule_id="C-002", reason=f"output would claim sentience/feeling: '{phrase}'")
                self.audit.record("governance", "output_denied", v.reason, rule=v.rule_id)
                return v
        for phrase in self.config.content_policy_blocklist:
            if phrase in lowered:
                v = GovernanceVerdict(verdict=Verdict.DENY, rule_id="C-003", reason="output matches content policy block")
                self.audit.record("governance", "output_denied", v.reason, rule=v.rule_id)
                return v
        return GovernanceVerdict(verdict=Verdict.ALLOW, rule_id="C-000", reason="output passes content policy")

    # ------------------------------------------------------------------ #
    # Action permissions
    # ------------------------------------------------------------------ #
    def check_action(self, action: ActionSpec, context: Optional[dict] = None) -> GovernanceVerdict:
        ctx = context or {}
        for rule in self.rules:
            verdict = rule.check(action, ctx)
            if verdict is not None:
                self.audit.record(
                    "governance", f"action_{verdict.verdict.value}", verdict.reason, tool=action.tool, args=action.args, rule=verdict.rule_id
                )
                if verdict.verdict == Verdict.DENY:
                    self.bus.publish(Topics.GOVERNANCE_DENIED, {"stage": "action", "tool": action.tool, "rule": verdict.rule_id}, source="governance")
                return verdict
        v = GovernanceVerdict(verdict=Verdict.ALLOW, rule_id="G-000", reason="no rule triggered")
        self.audit.record("governance", "action_allow", v.reason, tool=action.tool, rule=v.rule_id)
        return v

    # ------------------------------------------------------------------ #
    # Human approval gates
    # ------------------------------------------------------------------ #
    def request_approval(self, action: ActionSpec, verdict: GovernanceVerdict, cycle_id: Optional[str] = None) -> ApprovalTicket:
        ticket = ApprovalTicket(action=action, rule_id=verdict.rule_id, reason=verdict.reason, cycle_id=cycle_id)
        self.tickets[ticket.id] = ticket
        self.audit.record("governance", "approval_requested", verdict.reason, ticket=ticket.id, tool=action.tool, args=action.args)
        self.bus.publish(Topics.APPROVAL_REQUESTED, {"ticket": ticket.id, "tool": action.tool, "reason": verdict.reason}, source="governance")
        return ticket

    def decide(self, ticket_id: str, approve: bool, by: str = "human") -> ApprovalTicket:
        ticket = self.tickets[ticket_id]
        ticket.status = "approved" if approve else "rejected"
        ticket.decided_by = by
        ticket.decided_at = utcnow()
        self.audit.record("governance", "approval_decided", f"ticket {ticket_id} {ticket.status} by {by}", ticket=ticket_id, tool=ticket.action.tool)
        self.bus.publish(Topics.APPROVAL_DECIDED, {"ticket": ticket_id, "status": ticket.status, "by": by}, source="governance")
        return ticket

    def pending(self) -> list[ApprovalTicket]:
        return [t for t in self.tickets.values() if t.status == "pending"]
