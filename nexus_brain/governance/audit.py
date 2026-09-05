"""Tamper-evident audit log.

Every consequential decision (governance verdicts, decisions, actions,
upgrades) is appended with the *reason*. Entries form a hash chain so that
after-the-fact edits are detectable. Backed by a JSONL file when a path is
given; otherwise memory only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from nexus_brain.core.schemas import AuditEntry, utcnow


class AuditLog:
    GENESIS = "0" * 16

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[AuditEntry] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.entries.append(AuditEntry.model_validate_json(line))

    @property
    def last_hash(self) -> str:
        return self.entries[-1].hash if self.entries else self.GENESIS

    def record(self, actor: str, event: str, summary: str, **details: Any) -> AuditEntry:
        seq = len(self.entries) + 1
        ts = utcnow()
        prev = self.last_hash
        body = json.dumps(
            {"seq": seq, "ts": ts.isoformat(), "actor": actor, "event": event, "summary": summary, "details": details, "prev": prev},
            sort_keys=True,
            default=str,
        )
        h = hashlib.sha256(body.encode()).hexdigest()[:16]
        entry = AuditEntry(seq=seq, timestamp=ts, actor=actor, event=event, summary=summary, details=_jsonable(details), prev_hash=prev, hash=h)
        self.entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(entry.model_dump_json() + "\n")
        return entry

    def verify(self) -> bool:
        prev = self.GENESIS
        for e in self.entries:
            if e.prev_hash != prev:
                return False
            prev = e.hash
        return True

    def tail(self, n: int = 20) -> list[AuditEntry]:
        return self.entries[-n:]

    def query(self, event: Optional[str] = None, actor: Optional[str] = None) -> list[AuditEntry]:
        return [e for e in self.entries if (event is None or e.event == event) and (actor is None or e.actor == actor)]


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, default=str))
