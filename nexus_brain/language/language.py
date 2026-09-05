"""LANGUAGE & COMMUNICATION - Broca's/Wernicke's areas.

Purpose
    Produce the natural-language reply from the decision + execution results,
    in a register that fits the user and the situation.

Technique
    Two layers:
      1. **Content assembly** (deterministic) - what to say: results, failures,
         approval requests, refusals, memory answers, self-description.
      2. **Rendering** - how to say it: tone/verbosity/hedging come from the
         affect biases; user-specific register comes from semantic memory
         ("user prefers terse replies", user's name). With an LLM backend the
         assembled content + style instructions become the prompt; the offline
         backend renders templates directly.

Communication
    Called last in the cycle by the orchestrator; publishes
    ``language.response_generated``. Output is screened by governance before
    it is returned.

Data
    Reads the prompt library, affect style, retrieved memories, working-memory
    summary. Writes nothing.

Improvement
    Prompt/style templates are versioned in the PromptLibrary; the learning
    loop can propose new phrasing when feedback shows a template under-performs.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from nexus_brain.affect.affect import AffectState
from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.schemas import Decision, ExecutionReport, Percept, RetrievedMemory, MemoryKind
from nexus_brain.language.llm import LLM, TemplateLLM
from nexus_brain.language.prompts import PromptLibrary


class LanguageModule:
    def __init__(self, bus: EventBus, prompts: Optional[PromptLibrary] = None, affect: Optional[AffectState] = None, llm: Optional[LLM] = None) -> None:
        self.bus = bus
        self.prompts = prompts if prompts is not None else PromptLibrary()
        self.affect = affect
        self.llm = llm or TemplateLLM()

    # ------------------------------------------------------------------ #
    def respond(
        self,
        percept: Percept,
        decision: Optional[Decision],
        execution: Optional[ExecutionReport],
        retrieved: list[RetrievedMemory],
        user_profile: dict[str, str],
        context: str = "",
        refusal_reason: Optional[str] = None,
        approval_tickets: Optional[list[tuple[str, str, str]]] = None,
    ) -> str:
        style = self.affect.style() if self.affect else {"tone": "friendly", "verbosity": "normal", "hedging": "low"}
        name = user_profile.get("name")

        if refusal_reason:
            content = self.prompts.get("refusal", reason=refusal_reason)
            return self._finish(content, style, name, greet=False)

        if "explain_self" in percept.intents:
            return self._finish(self.prompts.get("self_description"), style, name, greet=False)

        parts: list[str] = []
        if approval_tickets:
            for ticket_id, action_desc, reason in approval_tickets:
                parts.append(self.prompts.get("approval_needed", action_desc=action_desc, reason=reason, ticket=ticket_id))

        if decision is not None:
            tag = decision.chosen.plan.skill_name
            if tag == "clarify":
                parts.append(self.prompts.get("clarify", examples="what outcome you want, any deadline, and who is involved"))
            elif tag == "memory_write":
                parts.append("Noted - I'll remember that.")
            elif tag == "direct_answer" or not decision.chosen.plan.steps:
                parts.append(self._answer_from_memory(percept, retrieved, user_profile))
            if execution is not None and execution.results:
                parts.append(self._describe_execution(execution, decision))

        content = " ".join(p for p in parts if p).strip() or "Okay."
        return self._finish(content, style, name, greet="greeting" in percept.intents)

    # ------------------------------------------------------------------ #
    def _answer_from_memory(self, percept: Percept, retrieved: list[RetrievedMemory], profile: dict[str, str]) -> str:
        intents = set(percept.intents)
        low = percept.text.lower()
        if "recall" in intents:
            facts = [r for r in retrieved if r.item.kind == MemoryKind.SEMANTIC]
            episodes = [r for r in retrieved if r.item.kind == MemoryKind.EPISODIC]
            if facts:
                lines = [f.item.content for f in facts[:3]]
                return "Here's what I have on that: " + "; ".join(lines) + "."
            if episodes:
                e = episodes[0].item
                return f"Last relevant thing I recall ({e.created_at:%Y-%m-%d}): {e.content}"
            return "I don't have anything stored about that yet."
        if "feedback_positive" in intents and not (intents & {"calculate", "weather", "search", "schedule", "email"}):
            return "You're welcome - glad that helped."
        if "feedback_negative" in intents:
            return "Sorry about that. Tell me what was off and I'll correct it."
        if "help" in intents or "what can you do" in low:
            return self.prompts.get("capabilities")
        if "greeting" in intents and len(percept.text.split()) <= 4:
            return "How can I help today?"
        if retrieved and retrieved[0].score > 0.45 and retrieved[0].item.kind == MemoryKind.SEMANTIC:
            return f"Based on what I know: {retrieved[0].item.content}"
        return "Understood." if len(percept.text.split()) < 4 else "Got it. Tell me if you'd like me to act on this."

    def _describe_execution(self, ex: ExecutionReport, decision: Decision) -> str:
        bits: list[str] = []
        for r in ex.results:
            if r.ok:
                bits.append(self._render_output(r.tool, r.output))
            elif r.governance == "require_approval":
                continue  # already covered by approval text
            elif r.governance == "deny":
                bits.append(f"I wasn't permitted to run {r.tool} ({r.error}).")
            else:
                bits.append(f"{r.tool} failed ({r.error}).")
        if ex.replans:
            bits.append(f"I had to adjust the plan {ex.replans} time(s) after a failure.")
        return " ".join(bits)

    @staticmethod
    def _render_output(tool: str, output: Any) -> str:
        if not isinstance(output, dict):
            return f"{tool}: {output}"
        if tool == "calculator":
            res = output.get("result")
            res_s = f"{res:g}" if isinstance(res, float) else str(res)
            return f"{output.get('expression')} = {res_s}."
        if tool == "get_weather":
            return f"Forecast for {output.get('location')}: {output.get('forecast')}, {output.get('temp_c')}°C."
        if tool == "check_calendar":
            slots = ", ".join(output.get("free_slots", [])) or "no free slots"
            return f"Calendar for {output.get('date')}: {len(output.get('events', []))} event(s); free: {slots}."
        if tool == "create_event":
            return f"Created '{output.get('title')}' on {output.get('date')} at {output.get('time')} (id {output.get('id')})."
        if tool == "send_email":
            return f"E-mail sent to {output.get('to')} ('{output.get('subject')}')."
        if tool == "make_payment":
            return f"Payment of {output.get('amount')} {output.get('currency')} to {output.get('to')} completed."
        if tool == "web_search":
            top = (output.get("results") or [{}])[0]
            return f"Top result: {top.get('title', 'n/a')} - {top.get('snippet', '')}"
        if tool == "take_note":
            return "Saved a note."
        if tool == "current_time":
            return f"It is {output.get('utc')} UTC."
        return f"{tool}: {json.dumps(output, default=str)[:160]}"

    # ------------------------------------------------------------------ #
    def _finish(self, content: str, style: dict[str, Any], name: Optional[str], greet: bool) -> str:
        hedge = self.prompts.get(f"hedge.{style.get('hedging', 'low')}")
        tone = str(style.get("tone", "friendly"))
        prefix = ""
        if greet and tone in ("warm", "friendly"):
            prefix = f"Hi{f' {name}' if name else ''}! "
        elif tone == "warm" and name:
            prefix = f"{name}, "
        elif tone == "calm_and_concise":
            prefix = ""
        if hedge and not content.lower().startswith(("i can't", "before i", "noted", "here's", "i'm a software")):
            content = hedge + content[0].lower() + content[1:]
        text = (prefix + content).strip()
        if style.get("verbosity") == "brief":
            text = _first_sentences(text, 2)
        if not isinstance(self.llm, TemplateLLM):
            system = self.prompts.get("system_persona") + f"\nStyle: {self.prompts.get('style.' + tone)}\nVerbosity: {style.get('verbosity')}. Hedging: {style.get('hedging')}."
            user = f"Rewrite the following assistant message in the requested style without adding facts:\n\n{text}"
            try:
                text = self.llm.complete(system, user)
            except Exception:  # noqa: BLE001 - never let a network hiccup kill the reply
                pass
        self.bus.publish(Topics.RESPONSE_GENERATED, {"chars": len(text), "tone": tone, "verbosity": style.get("verbosity")}, source="language")
        return text


def _first_sentences(text: str, n: int) -> str:
    out, count, buf = [], 0, ""
    for ch in text:
        buf += ch
        if ch in ".!?":
            out.append(buf)
            buf = ""
            count += 1
            if count >= n:
                break
    return ("".join(out) + ("" if count >= n else buf)).strip()
