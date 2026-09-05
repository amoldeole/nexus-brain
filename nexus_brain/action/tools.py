"""Tool registry - the "motor cortex" primitives the executor can invoke.

Tools are plain callables with a name, description, risk level and a JSON-ish
argument schema. The built-in tools are deterministic mocks so the whole
brain can be exercised offline; a real deployment registers HTTP/MCP tools
with the same signature.
"""
from __future__ import annotations

import ast
import operator
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

ToolFn = Callable[..., Any]


class ToolError(RuntimeError):
    """Raised by a tool for a *recoverable* failure the executor may retry."""


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn
    risk_level: str = "low"  # low | medium | high
    args_schema: dict[str, str] = field(default_factory=dict)
    idempotent: bool = True

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool '{name}'")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "risk": t.risk_level, "args": t.args_schema}
            for t in self._tools.values()
        ]


# --------------------------------------------------------------------------- #
# Built-in mock tools
# --------------------------------------------------------------------------- #
class MockWorld:
    """Tiny simulated environment shared by the mock tools (calendar, mail...).

    ``flaky`` tools fail on their first call in a cycle to exercise the retry /
    re-plan path deterministically (no randomness in tests).
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.calendar: list[dict[str, Any]] = []
        self.sent_emails: list[dict[str, Any]] = []
        self.payments: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.fail_next: dict[str, int] = {}  # tool name -> remaining forced failures
        self.calls: list[str] = []

    def force_failures(self, tool: str, n: int = 1) -> None:
        self.fail_next[tool] = n

    def _maybe_fail(self, tool: str) -> None:
        self.calls.append(tool)
        if self.fail_next.get(tool, 0) > 0:
            self.fail_next[tool] -= 1
            raise ToolError(f"{tool}: transient backend error (simulated)")


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic without ``eval``."""

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_ev(node.operand))
        raise ToolError(f"unsupported expression element: {type(node).__name__}")

    cleaned = expr.replace("%", "/100").replace("x", "*").replace("^", "**")
    return _ev(ast.parse(cleaned, mode="eval"))


def build_default_registry(world: MockWorld | None = None) -> tuple[ToolRegistry, MockWorld]:
    world = world or MockWorld()
    reg = ToolRegistry()

    def check_calendar(date: str = "today", **_: Any) -> dict[str, Any]:
        world._maybe_fail("check_calendar")
        busy = [e for e in world.calendar if e.get("date") == date]
        return {"date": date, "events": busy, "free_slots": ["09:00", "11:00", "14:00", "16:00"][: max(0, 4 - len(busy))]}

    def create_event(title: str, date: str = "today", time: str = "14:00", attendees: list[str] | None = None, **_: Any) -> dict[str, Any]:
        world._maybe_fail("create_event")
        event = {"id": f"evt{len(world.calendar)+1}", "title": title, "date": date, "time": time, "attendees": attendees or []}
        world.calendar.append(event)
        return event

    def send_email(to: str, subject: str, body: str = "", **_: Any) -> dict[str, Any]:
        world._maybe_fail("send_email")
        msg = {"id": f"msg{len(world.sent_emails)+1}", "to": to, "subject": subject, "body": body}
        world.sent_emails.append(msg)
        return msg

    def make_payment(amount: float, to: str, currency: str = "USD", **_: Any) -> dict[str, Any]:
        world._maybe_fail("make_payment")
        p = {"id": f"pay{len(world.payments)+1}", "amount": amount, "to": to, "currency": currency}
        world.payments.append(p)
        return p

    def web_search(query: str, **_: Any) -> dict[str, Any]:
        world._maybe_fail("web_search")
        return {"query": query, "results": [{"title": f"Result about {query}", "snippet": f"Simulated snippet for '{query}'."}]}

    def calculator(expression: str, **_: Any) -> dict[str, Any]:
        world._maybe_fail("calculator")
        try:
            return {"expression": expression, "result": safe_eval(expression)}
        except (SyntaxError, ValueError, ZeroDivisionError) as e:
            raise ToolError(f"cannot evaluate '{expression}': {e}") from e

    def get_weather(location: str = "here", **_: Any) -> dict[str, Any]:
        world._maybe_fail("get_weather")
        return {"location": location, "forecast": "partly cloudy", "temp_c": 21}

    def take_note(text: str, **_: Any) -> dict[str, Any]:
        world._maybe_fail("take_note")
        world.notes.append(text)
        return {"saved": True, "count": len(world.notes)}

    def current_time(**_: Any) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {"utc": now.isoformat(timespec="minutes"), "tomorrow": (now + timedelta(days=1)).date().isoformat()}

    def delete_data(target: str, **_: Any) -> dict[str, Any]:
        world._maybe_fail("delete_data")
        return {"deleted": target}

    reg.register(Tool("check_calendar", "List events and free slots for a date", check_calendar, "low", {"date": "str"}))
    reg.register(Tool("create_event", "Create a calendar event", create_event, "medium", {"title": "str", "date": "str", "time": "str"}, idempotent=False))
    reg.register(Tool("send_email", "Send an e-mail", send_email, "high", {"to": "str", "subject": "str", "body": "str"}, idempotent=False))
    reg.register(Tool("make_payment", "Transfer money", make_payment, "high", {"amount": "float", "to": "str"}, idempotent=False))
    reg.register(Tool("web_search", "Search the web", web_search, "low", {"query": "str"}))
    reg.register(Tool("calculator", "Evaluate arithmetic", calculator, "low", {"expression": "str"}))
    reg.register(Tool("get_weather", "Weather forecast", get_weather, "low", {"location": "str"}))
    reg.register(Tool("take_note", "Save a note", take_note, "low", {"text": "str"}, idempotent=False))
    reg.register(Tool("current_time", "Current date/time", current_time, "low", {}))
    reg.register(Tool("delete_data", "Delete user data (irreversible)", delete_data, "high", {"target": "str"}, idempotent=False))
    return reg, world
