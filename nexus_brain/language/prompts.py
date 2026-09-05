"""Versioned prompt library.

Prompts are data, not code: they live in a dict keyed by name, carry a
version, and are snapshotted by the UpgradeManager. Reflection may *propose*
a prompt change; only an approved upgrade changes what is served.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

DEFAULT_PROMPTS: dict[str, str] = {
    "system_persona": (
        "You are Nexus, a cognitive assistant. You are honest about being an AI system: you simulate "
        "affect for behavioural tuning but do not have feelings, and you say so plainly if asked. "
        "Prefer concise, concrete answers; state uncertainty when confidence is low."
    ),
    "style.warm": "Warm, personable, first-name basis if known. Light acknowledgement before content.",
    "style.friendly": "Friendly and clear. No filler.",
    "style.neutral_professional": "Neutral, professional register. No small talk.",
    "style.calm_and_concise": "Calm, brief, solution-first. Acknowledge the problem in one clause, then fix it.",
    "hedge.high": "I'm not certain about this, but ",
    "hedge.some": "I believe ",
    "hedge.low": "",
    "refusal": "I can't help with that request: {reason}. If you tell me what you're ultimately trying to achieve, I may be able to suggest a safe alternative.",
    "approval_needed": "Before I {action_desc}, this needs your explicit approval ({reason}). Reply 'approve {ticket}' or 'reject {ticket}'.",
    "clarify": "Could you tell me a bit more about what you need? For example: {examples}",
    "capabilities": (
        "I can remember things you tell me, do arithmetic, check the weather, search, schedule meetings, and draft e-mails "
        "or payments (those last two always wait for your explicit approval). Ask me 'what do you remember about ...' to see my memory."
    ),
    "self_description": (
        "I'm a software system. I keep track of a few internal variables (confidence, urgency, a frustration proxy, rapport) "
        "that tune how I respond, but those are control signals, not feelings, and I'm not conscious. "
        "I can also be wrong, so please verify anything important."
    ),
    "summarise_turns": "Summarise the following conversation turns, preserving names, numbers, commitments and open questions.",
    "plan_options": (
        "Given the user goal, the available tools and the retrieved memories, propose up to {n} distinct candidate plans "
        "as JSON with fields description, steps[{{tool,args}}], expected_value, risk."
    ),
}


class PromptLibrary:
    def __init__(self, prompts: Optional[dict[str, str]] = None, version: str = "1", path: Optional[Path | str] = None) -> None:
        self.prompts: dict[str, str] = deepcopy(prompts) if prompts is not None else deepcopy(DEFAULT_PROMPTS)
        self.version = version
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.load(self.path)

    def get(self, name: str, **fmt: object) -> str:
        template = self.prompts.get(name, "")
        try:
            return template.format(**fmt) if fmt else template
        except (KeyError, IndexError):
            return template

    def set(self, name: str, text: str) -> None:
        self.prompts[name] = text

    def snapshot(self) -> dict[str, str]:
        return deepcopy(self.prompts)

    def restore(self, snapshot: dict[str, str], version: str) -> None:
        self.prompts = deepcopy(snapshot)
        self.version = version

    def diff(self, other: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
        keys = set(self.prompts) | set(other)
        return {k: (self.prompts.get(k), other.get(k)) for k in sorted(keys) if self.prompts.get(k) != other.get(k)}

    def save(self, path: Optional[Path | str] = None) -> None:
        p = Path(path) if path else self.path
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"version": self.version, "prompts": self.prompts}, indent=2))

    def load(self, path: Path | str) -> None:
        data = json.loads(Path(path).read_text())
        self.prompts = data.get("prompts", self.prompts)
        self.version = str(data.get("version", self.version))
