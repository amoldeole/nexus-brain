"""HTTP API (FastAPI) - a thin, stateless-looking shell around one Brain.

    POST /chat                  {"text": "...", "modality": "text"} -> response + trace
    POST /approvals/{id}        {"approve": true, "by": "alice"}
    GET  /status
    GET  /memory?kind=semantic
    GET  /audit?tail=50
    POST /sleep                 run reflection + decay + forget
    GET  /upgrades              versions + candidates
    POST /upgrades/{id}/evaluate
    POST /upgrades/{id}/approve {"reviewer": "alice", "notes": "..."}
    POST /upgrades/{id}/reject  {"reviewer": "alice", "reason": "..."}
    POST /upgrades/rollback     {"reviewer": "alice", "reason": "..."}

Run:  uvicorn nexus_brain.api:app --host 0.0.0.0 --port 8000
Optional: NEXUS_DATA_DIR=./data for persistence; NEXUS_LLM_* for a real LLM.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from nexus_brain.brain import Brain
from nexus_brain.core.config import NexusConfig
from nexus_brain.core.schemas import MemoryKind
from nexus_brain.language.llm import llm_from_env


class ChatIn(BaseModel):
    text: Union[str, dict[str, Any], list[Any]]
    modality: str = "text"
    feedback: Optional[float] = None


class ApprovalIn(BaseModel):
    approve: bool = True
    by: str = "human"


class ReviewIn(BaseModel):
    reviewer: str
    notes: str = ""
    reason: str = ""


def create_app(brain: Optional[Brain] = None) -> FastAPI:
    if brain is None:
        data_dir = os.environ.get("NEXUS_DATA_DIR")
        cfg_path = Path(data_dir) / "config.yaml" if data_dir else None
        config = NexusConfig.load(cfg_path) if cfg_path and cfg_path.exists() else NexusConfig()
        brain = Brain(config=config, data_dir=data_dir, llm=llm_from_env())

    app = FastAPI(title="Nexus Brain", version=brain.config.version, description="Human-brain-inspired, human-supervised cognitive assistant")
    app.state.brain = brain

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _CONSOLE_HTML

    @app.post("/chat")
    def chat(body: ChatIn = Body(...)) -> dict[str, Any]:
        result = brain.process(body.text, modality=body.modality, user_feedback=body.feedback)
        return {
            "cycle_id": result.cycle_id,
            "response": result.response,
            "refused": result.refused,
            "pending_approvals": result.pending_approvals,
            "mode": result.decision.mode.value if result.decision else None,
            "chosen": result.decision.chosen.description if result.decision else None,
            "rationale": result.decision.rationale if result.decision else None,
            "affect": result.affect.model_dump(mode="json", exclude={"timestamp"}),
            "trace": [t.model_dump(mode="json") for t in result.trace],
        }

    @app.post("/approvals/{ticket_id}")
    def approvals(ticket_id: str, body: ApprovalIn = Body(...)) -> dict[str, Any]:
        if ticket_id not in brain.governance.tickets:
            raise HTTPException(404, "unknown ticket")
        result = brain.approve(ticket_id, body.approve, by=body.by)
        return {"response": result.response, "trace": [t.model_dump(mode="json") for t in result.trace]}

    @app.get("/approvals")
    def pending_approvals() -> list[dict[str, Any]]:
        return [t.model_dump(mode="json") for t in brain.governance.pending()]

    @app.get("/status")
    def status() -> dict[str, Any]:
        return brain.status()

    @app.get("/memory")
    def memory(kind: Optional[str] = None, limit: int = 50) -> dict[str, Any]:
        kinds = [MemoryKind(kind)] if kind else None
        items = sorted(brain.ltm.store.all(kinds), key=lambda i: i.created_at, reverse=True)[:limit]
        return {
            "items": [i.model_dump(mode="json", exclude={"embedding"}) for i in items],
            "skills": [s.model_dump(mode="json") | {"success_rate": s.success_rate} for s in brain.ltm.skills.values()],
            "working_memory": brain.wm.context_window(),
        }

    @app.get("/audit")
    def audit(tail: int = 50) -> dict[str, Any]:
        return {"valid": brain.audit.verify(), "entries": [e.model_dump(mode="json") for e in brain.audit.tail(tail)]}

    @app.post("/sleep")
    def sleep() -> dict[str, Any]:
        report = brain.sleep()
        return {"report": report, "lessons": [l.model_dump(mode="json") for l in brain.learning.lessons[-10:]]}

    @app.get("/upgrades")
    def upgrades() -> dict[str, Any]:
        um = brain.upgrade
        if um is None:
            raise HTTPException(400, "upgrade manager disabled")
        return {
            "current": um.current.id,
            "versions": um.history(),
            "candidates": [
                {"id": c.id, "status": c.status, "title": c.proposal.title, "kind": c.proposal.kind.value, "diff": c.diff,
                 "report": c.report.model_dump(mode="json") if c.report else None, "rejection_reason": c.rejection_reason}
                for c in um.candidates.values()
            ],
        }

    @app.post("/upgrades/rollback")
    def rollback(body: ReviewIn = Body(...)) -> dict[str, Any]:
        um = brain.upgrade
        if um is None:
            raise HTTPException(400, "upgrade manager disabled")
        try:
            v = um.rollback(reviewer=body.reviewer, reason=body.reason)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"active_version": v.id, "label": v.label}

    @app.post("/upgrades/{candidate_id}/evaluate")
    def evaluate(candidate_id: str) -> dict[str, Any]:
        um = brain.upgrade
        if um is None or candidate_id not in um.candidates:
            raise HTTPException(404, "unknown candidate")
        report = um.evaluate(candidate_id)
        return report.model_dump(mode="json") | {"summary": report.summary()}

    @app.post("/upgrades/{candidate_id}/approve")
    def approve(candidate_id: str, body: ReviewIn = Body(...)) -> dict[str, Any]:
        um = brain.upgrade
        if um is None or candidate_id not in um.candidates:
            raise HTTPException(404, "unknown candidate")
        try:
            v = um.approve(candidate_id, reviewer=body.reviewer, notes=body.notes)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        return {"version": v.model_dump(mode="json", exclude={"prompts", "config_yaml"})}

    @app.post("/upgrades/{candidate_id}/reject")
    def reject(candidate_id: str, body: ReviewIn = Body(...)) -> dict[str, Any]:
        um = brain.upgrade
        if um is None or candidate_id not in um.candidates:
            raise HTTPException(404, "unknown candidate")
        c = um.reject(candidate_id, reviewer=body.reviewer, reason=body.reason)
        return {"id": c.id, "status": c.status}

    return app


_CONSOLE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Nexus Brain console</title>
<style>
 body{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0f1117;color:#e6e6e6;margin:0;display:grid;grid-template-columns:1fr 1fr;height:100vh}
 #left,#right{padding:16px;overflow:auto}#right{border-left:1px solid #2a2f3a;background:#0b0d12}
 h1{font-size:16px;margin:0 0 8px}small{color:#8b93a7}
 #log{min-height:60vh}.u{color:#7cc4ff}.n{color:#b5f5a0}.sys{color:#8b93a7}
 form{display:flex;gap:8px;margin-top:8px}input{flex:1;padding:10px;background:#161a24;border:1px solid #2a2f3a;color:#fff;border-radius:6px}
 button{padding:10px 14px;background:#3b82f6;border:0;color:#fff;border-radius:6px;cursor:pointer}
 .stage{margin:4px 0;padding:6px 8px;background:#161a24;border-left:3px solid #3b82f6;border-radius:4px;font-size:12px}
 .stage b{color:#ffd479}pre{white-space:pre-wrap;font-size:11px;color:#c9d1d9;margin:4px 0 0}
 .bar{display:inline-block;height:8px;background:#3b82f6;border-radius:4px;vertical-align:middle}
 .aff{font-size:12px;margin:2px 0}.aff span{display:inline-block;width:130px}
</style></head><body>
<div id="left"><h1>Nexus Brain <small>console - simulated affect, human-gated actions, audited decisions</small></h1>
<div id="log"></div>
<form onsubmit="send(event)"><input id="t" placeholder="Try: 'My name is Priya and I prefer morning meetings' / 'calculate 12*7' / 'schedule a meeting with Dana tomorrow' / 'pay $500 to Acme'" autofocus><button>Send</button></form>
<div class="sys" style="margin-top:8px">Pending approvals: reply <code>approve &lt;ticket&gt;</code> or <code>reject &lt;ticket&gt;</code>. <a href="/docs" style="color:#7cc4ff">API docs</a></div></div>
<div id="right"><h1>Cognitive trace</h1><div id="affect"></div><div id="trace"><small>send a message to see perception -> memory -> affect -> reasoning -> action -> language -> learning</small></div></div>
<script>
const log=document.getElementById('log'),trace=document.getElementById('trace'),aff=document.getElementById('affect');
function add(cls,who,text){const d=document.createElement('div');d.className=cls;d.textContent=who+'> '+text;log.appendChild(d);log.scrollTop=log.scrollHeight}
function bars(a){aff.innerHTML=Object.entries(a).map(([k,v])=>`<div class="aff"><span>${k}</span><i class="bar" style="width:${Math.round(v*200)}px"></i> ${v.toFixed(2)}</div>`).join('')+'<small>simulated control variables - not feelings</small>'}
async function send(e){e.preventDefault();const t=document.getElementById('t');const text=t.value.trim();if(!text)return;t.value='';add('u','you',text);
 const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});const j=await r.json();
 add('n','nexus',j.response);bars(j.affect);
 trace.innerHTML=j.trace.map(s=>`<div class="stage"><b>${s.stage}</b> ${s.summary}<pre>${s.details&&Object.keys(s.details).length?JSON.stringify(s.details,null,1).slice(0,900):''}</pre></div>`).join('')+(j.rationale?`<div class="stage"><b>rationale</b><pre>${j.rationale}</pre></div>`:'')}
</script></body></html>"""


app = create_app()
