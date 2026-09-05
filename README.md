# Nexus Brain

A **human-brain-inspired, human-supervised cognitive architecture** for AI
assistants — ten cooperating faculties (perception, working memory,
long-term memory, reasoning, simulated affect, language, planning/action,
learning loop, upgrade manager, governance) behind one auditable cognitive
cycle.

This repository contains both the **design** (docs/) and a **working
reference implementation** (nexus_brain/) that runs fully offline with no
model keys, so every mechanism — System 1 vs. System 2, memory scoring,
affect-driven style, re-planning, approval gates, reflection, evaluated and
roll-back-able upgrades — can be exercised and tested end to end. Plug in an
LLM, a vector DB and real tools through the provided interfaces to go to
production.

> **Honesty first:** Nexus Brain is not conscious, has no feelings (its
> "affect" is four documented control variables), and is not guaranteed to
> be correct. See [docs/HONESTY.md](docs/HONESTY.md).

## Documents

| Deliverable | Where |
|---|---|
| System diagram (Mermaid) + data flow + module-by-module design (purpose · technique · communication · data · improvement) | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Suggested tech stack per module | [docs/ARCHITECTURE.md §3](docs/ARCHITECTURE.md#3-suggested-tech-stack-per-module) |
| "Day in the life" trace of one request through every faculty | [docs/DAY_IN_THE_LIFE.md](docs/DAY_IN_THE_LIFE.md) |
| Phased roadmap (MVP → v2 → v3) | [docs/ROADMAP.md](docs/ROADMAP.md) |
| What the system will NOT claim to be | [docs/HONESTY.md](docs/HONESTY.md) |

## The cognitive cycle

```
input ─► PERCEPTION ─► GOVERNANCE(input) ─► WORKING MEMORY + LONG-TERM RECALL ─► AFFECT
      ─► REASONING (System 1 fast path | System 2 options × EV/prior/feasibility − risk)
      ─► GOVERNANCE(per action: allow / require approval / deny) ─► EXECUTOR (retry · re-plan · pause)
      ─► LANGUAGE (style from affect) ─► GOVERNANCE(output) ─► MEMORY write-back ─► OUTCOME log
      ┄┄ every N cycles / sleep ┄┄► REFLECTION ─► lessons ─► bounded auto-updates | proposals
                                              ─► UPGRADE MANAGER: sandbox eval ─► diff + report ─► HUMAN approve ─► version (rollback kept)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

nexus-brain demo -v          # scripted day-in-the-life with full traces
nexus-brain chat --trace     # interactive REPL (/status, /sleep, /pending, /audit)
nexus-brain eval             # quality + safety scenario suite
pytest                       # 48 tests

# persistent brain + HTTP API + browser console at http://localhost:8000/
NEXUS_DATA_DIR=./data uvicorn nexus_brain.api:app --host 0.0.0.0 --port 8000
```

Optional real LLM for the language layer (any OpenAI-compatible endpoint):

```bash
export NEXUS_LLM_BASE_URL=https://api.openai.com/v1   # or http://localhost:11434/v1 for Ollama
export NEXUS_LLM_API_KEY=...
export NEXUS_LLM_MODEL=gpt-4o-mini
```

## Python API

```python
from nexus_brain import Brain

brain = Brain(data_dir="./data")             # omit data_dir for an in-memory brain
r = brain.process("Hi, my name is Priya and I prefer morning meetings")
r = brain.process("schedule a meeting with Dana tomorrow")
print(r.response)                            # Calendar for tomorrow ... Created 'Meeting with Dana' at 09:00
print(r.decision.mode, r.decision.rationale) # system1 / system2 + why
for stage in r.trace:                        # perception → memory → affect → reasoning → action → language → learning
    print(stage.stage, stage.summary)

r = brain.process("pay $500 to Acme")        # high-stakes → approval ticket, nothing executed
brain.process(f"approve {r.pending_approvals[0]}")

brain.sleep()                                # reflection + memory decay/forgetting
for cand in brain.upgrade.pending():         # proposals the learning loop could not auto-apply
    report = brain.upgrade.evaluate(cand.id) # sandboxed before/after eval
    if report.passed:
        brain.upgrade.approve(cand.id, reviewer="alice")
brain.upgrade.rollback(reviewer="alice", reason="regression")
```

## What is implemented

| # | Module | Package | Highlights |
|---|---|---|---|
| 1 | Perception | `perception/` | text / voice / image / structured → `Percept`; intents, entities, sentiment/urgency cues, embeddings; swappable STT/vision/embedding adapters |
| 2 | Working memory | `memory/working.py` | token budget; keep-last-k / discard low salience / summarise; consolidation into LTM |
| 3 | Long-term memory | `memory/long_term.py` | episodic + semantic + procedural; `relevance·w + recency·w + importance·w` retrieval; access strengthening, decay, forgetting, dedupe; JSON persistence; `VectorStore` protocol |
| 4 | Reasoning | `reasoning/engine.py` | System 1 skill fast path with affect veto; System 2 option generation, governance pre-filter, risk-adjusted scoring, episodic priors, explicit refusal, re-plan hook |
| 5 | Affect (simulated) | `affect/affect.py` | leaky-integrator state; style / risk-tolerance / fast-path biases; every update published with its cause |
| 6 | Language | `language/` | content assembly + affect-driven register; versioned prompt library; offline template or OpenAI-compatible LLM |
| 7 | Planning & action | `action/` | DAG executor, retries, fallback re-planning, dependency skipping, output templating, approval pause/resume; tool registry with mock world |
| 8 | Learning loop | `learning/reflection.py` | outcome log; skill / tool / retrieval / feedback / consolidation miners; bounded auto-apply vs. gated proposals |
| 9 | Upgrade manager | `upgrade/` | version registry; sandboxed before/after eval; quality + safety gates; human-only approval; rollback |
| 10 | Governance | `governance/` | deterministic rules (deny / approve / allow), content policy in & out, approval tickets, hash-chained audit log |
| — | Orchestrator, CLI, API | `brain.py`, `cli.py`, `api.py` | per-cycle trace; REPL; FastAPI + browser console |

## Design constraints honoured

* **Human-supervised self-modification** — the system changes memory,
  statistics and bounded weights on its own; prompts, config and model
  checkpoints only via *propose → sandbox eval → diff/report → human approve*,
  with rollback. It never modifies code or governance rules, and cannot
  approve itself.
* **Well-tested components over novel ML** — the architecture composes
  LLMs, vector databases, a rules engine and a workflow executor; the
  offline reference uses simple deterministic stand-ins with identical
  interfaces.

## License

MIT
