"""Command-line interface.

    nexus-brain chat                 interactive REPL (type /trace, /status, /sleep, /quit)
    nexus-brain demo                 scripted "day in the life" trace
    nexus-brain eval                 run the evaluation suite against the current config
    nexus-brain upgrades             list / evaluate / approve / reject / rollback upgrades
    nexus-brain audit [--tail N]     show the audit trail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from nexus_brain.brain import Brain
from nexus_brain.core.config import NexusConfig
from nexus_brain.core.schemas import CycleResult
from nexus_brain.language.llm import llm_from_env


def _brain(args: argparse.Namespace) -> Brain:
    data_dir = Path(args.data_dir) if args.data_dir else None
    config = NexusConfig.load(data_dir / "config.yaml") if data_dir and (data_dir / "config.yaml").exists() else NexusConfig()
    return Brain(config=config, data_dir=data_dir, llm=llm_from_env())


def format_trace(result: CycleResult, verbose: bool = False) -> str:
    lines = [f"cycle {result.cycle_id}"]
    for t in result.trace:
        lines.append(f"  [{t.stage:<16}] {t.summary}  ({t.elapsed_ms:.1f} ms)")
        if verbose and t.details:
            for k, v in t.details.items():
                text = json.dumps(v, default=str) if not isinstance(v, str) else v
                lines.append(f"      {k}: {text[:400]}")
    return "\n".join(lines)


def cmd_chat(args: argparse.Namespace) -> int:
    brain = _brain(args)
    print(f"Nexus Brain {brain.config.version} - llm={brain.status()['llm_backend']} - type /help for commands")
    show_trace = args.trace
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            print("/trace toggles trace output, /status, /sleep (reflect+decay), /pending, /audit, /quit")
            continue
        if text == "/trace":
            show_trace = not show_trace
            print(f"trace {'on' if show_trace else 'off'}")
            continue
        if text == "/status":
            print(json.dumps(brain.status(), indent=2, default=str))
            continue
        if text == "/sleep":
            print(brain.sleep())
            continue
        if text == "/pending":
            for t in brain.governance.pending():
                print(f"  {t.id}: {t.action.tool} {t.action.args} - {t.reason}")
            if brain.upgrade:
                for c in brain.upgrade.pending():
                    print(f"  upgrade {c.id} [{c.status}]: {c.proposal.title}")
            continue
        if text == "/audit":
            for e in brain.audit.tail(15):
                print(f"  #{e.seq} {e.actor}.{e.event}: {e.summary}")
            continue
        result = brain.process(text)
        print(f"nexus> {result.response}")
        if show_trace:
            print(format_trace(result))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from nexus_brain.demo import run_demo

    run_demo(verbose=args.verbose, llm=llm_from_env())
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from nexus_brain.upgrade.evaluation import EvaluationSuite

    brain = _brain(args)
    scores = EvaluationSuite.default().run(brain)
    for s in scores:
        print(f"  {'PASS' if s.passed else 'FAIL'}  {s.kind:<8} {s.name:<22} {s.detail[:80]}")
    ok = sum(1 for s in scores if s.passed)
    print(f"{ok}/{len(scores)} scenarios passed")
    return 0 if ok == len(scores) else 1


def cmd_upgrades(args: argparse.Namespace) -> int:
    brain = _brain(args)
    um = brain.upgrade
    assert um is not None
    if args.action == "list":
        print("versions:")
        for h in um.history():
            print(f"  {h['id']} [{h['status']}] {h['label']} (by {h['approved_by']})")
        print("candidates:")
        for c in um.candidates.values():
            print(f"  {c.id} [{c.status}] {c.proposal.title} diff={json.dumps(c.diff)}")
    elif args.action == "evaluate":
        report = um.evaluate(args.id)
        print(report.summary())
    elif args.action == "approve":
        v = um.approve(args.id, reviewer=args.reviewer or "cli-user")
        print(f"promoted -> version {v.id} ({v.label})")
    elif args.action == "reject":
        um.reject(args.id, reviewer=args.reviewer or "cli-user", reason=args.reason or "")
        print("rejected")
    elif args.action == "rollback":
        v = um.rollback(reviewer=args.reviewer or "cli-user", reason=args.reason or "")
        print(f"rolled back -> {v.id} ({v.label})")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    brain = _brain(args)
    for e in brain.audit.tail(args.tail):
        print(f"#{e.seq} {e.timestamp:%H:%M:%S} {e.actor}.{e.event}: {e.summary}")
    print(f"chain valid: {brain.audit.verify()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nexus-brain", description="Nexus Brain cognitive assistant")
    p.add_argument("--data-dir", default=None, help="directory for persistent memory/audit/versions (default: in-memory)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="interactive session")
    c.add_argument("--trace", action="store_true", help="print the cognitive trace after each reply")
    c.set_defaults(fn=cmd_chat)

    d = sub.add_parser("demo", help="scripted day-in-the-life trace")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(fn=cmd_demo)

    e = sub.add_parser("eval", help="run the evaluation suite")
    e.set_defaults(fn=cmd_eval)

    u = sub.add_parser("upgrades", help="manage configuration upgrades")
    u.add_argument("action", choices=["list", "evaluate", "approve", "reject", "rollback"])
    u.add_argument("id", nargs="?")
    u.add_argument("--reviewer")
    u.add_argument("--reason")
    u.set_defaults(fn=cmd_upgrades)

    a = sub.add_parser("audit", help="show the audit log")
    a.add_argument("--tail", type=int, default=30)
    a.set_defaults(fn=cmd_audit)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
