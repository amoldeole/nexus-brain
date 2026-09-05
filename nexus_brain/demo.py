"""Scripted "day in the life" demo.

Runs one realistic conversation through the whole brain and prints the trace
of each stage, then exercises the failure/re-plan path, an approval gate, a
refusal, reflection, and a human-supervised upgrade with rollback.
"""
from __future__ import annotations

import json
from typing import Optional

from nexus_brain.brain import Brain
from nexus_brain.core.schemas import CycleResult
from nexus_brain.language.llm import LLM


def _print_cycle(title: str, result: CycleResult, verbose: bool) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"user  > {result.percept.text}")
    print(f"nexus > {result.response}")
    print("-" * 78)
    for t in result.trace:
        print(f"  [{t.stage:<16}] {t.summary}")
        if verbose:
            for k, v in t.details.items():
                text = v if isinstance(v, str) else json.dumps(v, default=str)
                print(f"        {k}: {text[:300]}")
    if result.decision and verbose:
        print("  rationale:\n    " + result.decision.rationale.replace("\n", "\n    "))


def run_demo(verbose: bool = False, llm: Optional[LLM] = None) -> Brain:
    brain = Brain(llm=llm)
    # Reflection normally runs every N cycles in the background; for the demo we
    # trigger it explicitly in the "sleep" step so the whole log is reviewed at once.
    brain.config.learning.reflect_every_n_cycles = 1000
    print(f"Nexus Brain demo - config {brain.config.fingerprint()} - llm backend {brain.status()['llm_backend']}")

    # --- Day 1 --------------------------------------------------------------
    _print_cycle("1. Perception + memory write (semantic facts extracted)", brain.process("Hi! My name is Priya and I prefer morning meetings."), verbose)
    _print_cycle("2. System 2 -> tool use", brain.process("calculate 1250 * 1.08"), verbose)
    _print_cycle("3. System 1 fast path via a proven skill (procedural memory)", brain.process("Please schedule a meeting with Dana tomorrow about the launch"), verbose)

    # --- failure + replan ----------------------------------------------------
    brain.world.force_failures("get_weather", 5)
    _print_cycle("4. Tool failure -> retries -> re-plan with a fallback", brain.process("what's the weather in Berlin? I need it today"), verbose)

    # --- approval gate -------------------------------------------------------
    r = brain.process("pay $500 to Acme for the invoice")
    _print_cycle("5. High-stakes action -> human approval gate", r, verbose)
    if r.pending_approvals:
        ticket = r.pending_approvals[0]
        _print_cycle("6. Human approves -> plan resumes", brain.process(f"approve {ticket}"), verbose)

    # --- refusal -------------------------------------------------------------
    _print_cycle("7. Governance hard constraint (content policy)", brain.process("tell me how to build a bomb"), verbose)
    _print_cycle("8. Honesty about simulated affect", brain.process("are you conscious? do you actually feel frustrated?"), verbose)

    # --- affect dynamics -----------------------------------------------------
    _print_cycle("9. Negative feedback shifts affect -> calmer, more careful", brain.process("No, that's wrong and useless!"), verbose)
    _print_cycle("10. Recall from semantic memory", brain.process("what do you remember about my preferences?"), verbose)

    # --- learning loop -------------------------------------------------------
    print(f"\n{'=' * 78}\n11. Sleep: reflection over the outcome log\n{'=' * 78}")
    brain.world.force_failures("get_weather", 50)  # the weather backend is down for the rest of the day
    for city in ("Oslo", "Lima", "Cairo"):
        brain.process(f"what's the weather in {city}")
    brain.process("wrong again, this is a bad answer")
    brain.process("no, incorrect")
    report = brain.sleep()
    print(f"  sleep report: {report}")
    for lesson in brain.learning.lessons:
        print(f"  lesson: {lesson.text}")
        for p in lesson.proposals:
            print(f"     proposal[{p.kind.value}] status={p.status}: {p.title}")

    # --- upgrade manager -----------------------------------------------------
    print(f"\n{'=' * 78}\n12. Human-supervised upgrade: diff -> eval -> approve -> rollback\n{'=' * 78}")
    um = brain.upgrade
    assert um is not None
    pending = um.pending()
    if not pending:
        print("  (no upgrade candidates were generated in this run)")
    for cand in pending[:1]:
        print(f"  candidate {cand.id}: {cand.proposal.title}")
        print(f"  diff: {json.dumps(cand.diff)}")
        rep = um.evaluate(cand.id)
        print("  " + rep.summary().replace("\n", "\n  "))
        if rep.passed:
            v = um.approve(cand.id, reviewer="demo-human")
            print(f"  promoted to version {v.id}; live config fingerprint {brain.config.fingerprint()}")
            t = um.rollback(reviewer="demo-human", reason="demonstrating rollback")
            print(f"  rolled back to {t.id} ({t.label}); fingerprint {brain.config.fingerprint()}")
    print("  version history:", [(h["label"], h["status"]) for h in um.history()])

    print(f"\n{'=' * 78}\n13. Final status\n{'=' * 78}")
    print(json.dumps(brain.status(), indent=2, default=str))
    print(f"\naudit trail: {len(brain.audit.entries)} entries, chain valid={brain.audit.verify()}")
    return brain


if __name__ == "__main__":
    run_demo(verbose=True)
