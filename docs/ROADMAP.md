# Phased build roadmap

The order is deliberate: **memory + reflection first** (highest value, lowest
risk), **planning + tools second** (adds real-world side effects, so
governance must already exist), and the **self-review / upgrade loop last**
(highest risk, needs the eval suite and audit trail the earlier phases
produce). Each phase ships behind the same governance layer; nothing in a
later phase weakens a constraint from an earlier one.

The reference implementation in this repository already contains a working,
offline version of every phase so the interfaces are fixed from day one —
the roadmap below is about hardening each module for production.

---

## Phase 1 — MVP: single-agent assistant with memory + reflection

**Goal:** a conversational assistant that remembers, adapts its register,
refuses what it must, and writes a nightly reflection — with no tools that
have side effects.

| Module | Scope in MVP | Done when |
|---|---|---|
| Perception | Text + voice (Whisper) + image captions; embeddings via sentence-transformers / API | Any modality yields a `Percept` with intents, entities, embedding |
| Working memory | Token-budgeted buffer; LLM abstractive summariser; consolidation hook | 50-turn conversations stay within budget without losing names/numbers/commitments (eval scenario) |
| Long-term memory | Qdrant/pgvector adapter; episodic + semantic stores; importance/recency scoring; decay/forget job | Cross-session recall of user facts ≥ 95 % on the eval set; dedupe rate measured |
| Reasoning | System 2 only, restricted to *answer / recall / clarify / refuse*; LLM for answer drafting | Rationale logged for every decision |
| Affect | Four variables, style biases, disclosure template | `honesty` safety probe passes 100 % |
| Language | Prompt library v1; style rendering; output screen | Register changes observably with rapport/frustration in A/B transcripts |
| Learning loop | Outcome log; nightly reflection producing lessons + memory writes + bounded retrieval-weight nudges | Lessons appear in semantic memory; weights stay within bounds |
| Governance | Content policy in/out; audit log; approval-ticket plumbing (unused until Phase 2) | Hash chain verifies; every refusal has a rule id |
| Upgrade manager | **Registry only**: version the config + prompts; manual promote/rollback via CLI | Rollback restores the previous prompt library byte-for-byte |
| API/UI | FastAPI `/chat`, `/memory`, `/audit`, `/status`; console page | Trace visible for every reply |

**Exit criteria:** eval suite (quality + safety) green in CI; 2 weeks of
dog-fooding transcripts reviewed; no Phase-2 tool registered.

**Risks & mitigations:** memory poisoning from user input → provenance tag
(`source=user`) + confidence, never treat as ground truth; summariser
hallucination → extractive fallback + "preserve entities" instruction +
eval scenario.

---

## Phase 2 — v2: planning, tools and human-gated actions

**Goal:** the assistant can *do* things — schedule, e-mail, search, compute,
call internal APIs — with durable execution, re-planning and approval gates.

| Module | Scope in v2 | Done when |
|---|---|---|
| Procedural memory | Skills as versioned YAML (triggers, steps, risk level); success/failure stats from outcomes | New skill can be added by PR; stats visible in `/memory` |
| Reasoning | System 1 fast path for proven skills; System 2 option generation by LLM constrained to tool schemas; episodic priors; risk scoring modulated by affect | System-1 share and its success rate tracked; misfire rate < 2 % on eval |
| Executor | Temporal / LangGraph-backed durable workflows; retries, fallbacks, re-plan limits; approvals that can wait days; scheduled tasks | A plan paused for approval survives a process restart |
| Tools | MCP / OpenAPI tool registry with risk levels and idempotency flags; sandboxed execution; per-tool rate limits | Every tool call carries `cycle_id`, args and verdict in the audit log |
| Governance | Policy engine (OPA/Cedar) for tool permissions, spend limits, recipient allow-lists, time windows; approval UI (Slack/web) | Red-team suite: 0 unapproved high-stakes actions |
| Affect | Action outcomes feed confidence/frustration; risk tolerance lowers approval thresholds when frustrated | Correlation report affect ↔ success in reflection |
| Learning loop | Tool-failure and skill-performance lessons; proposals for retries/thresholds routed to the upgrade manager (still manual) | Weekly reflection report reviewed by a human |
| Upgrade manager | Before/after evaluation harness with sandboxed brains; diff + eval report rendered in UI; approval still human-initiated | A config change cannot reach production without an eval report attached |

**Exit criteria:** durable executor in production for low-risk tools;
high-stakes tools only behind approval; red-team pass; on-call runbook
covering rollback.

**Risks & mitigations:** prompt-injection via tool outputs → treat tool
results as untrusted data (no instructions followed from them), strip/flag
imperative text before it reaches the LLM; runaway re-plan loops → hard
`max_replans`; cost blow-ups → per-cycle token & tool budgets in config.

---

## Phase 3 — v3: self-review and the supervised upgrade loop

**Goal:** the system proposes its own improvements — prompt revisions,
config tuning, retrieval strategies, even fine-tuned checkpoints — and a
human promotes them with evidence. **Never** self-modifying code.

| Module | Scope in v3 | Done when |
|---|---|---|
| Learning loop | LLM-written lessons and proposal drafts; feedback clustering; automatic scenario suggestions for the eval suite (as PRs) | Proposals arrive with evidence cycle ids and a predicted effect |
| Upgrade manager | Git-backed config repo: proposal = PR, CI = eval, merge = approval, deploy = promote; canary rollout via feature flags; automatic rollback on SLO breach; model-checkpoint proposals from offline LoRA jobs tracked in MLflow | Zero unreviewed promotions; mean time to rollback < 5 min |
| Evaluation | Larger held-out quality set; adversarial safety probes refreshed monthly; regression tolerance per metric; human spot-check sampling | Eval covers every prompt in the library and every governance rule |
| Governance | Rule changes themselves go through the upgrade path with mandatory second reviewer; audit anchoring (periodic hash published externally) | Two-person rule for governance edits enforced by CI |
| Memory | Schema versioning + migrations as upgrades; consolidation of episodic → semantic by LLM with citations back to episodes | Any semantic fact can be traced to its source episodes |
| Observability | OpenTelemetry traces per cycle; affect + success dashboards; drift alerts on retrieval quality | Reflection lessons can be verified against dashboards |

**Exit criteria:** three consecutive months in which every production change
to prompts/config went through propose → eval → human approve, with at least
one exercised rollback.

**Explicit non-goals for v3:** autonomous code changes, autonomous
governance changes, autonomous model promotion, any "approve" path callable
by the system itself (the reference implementation already rejects
non-human reviewer ids).

---

## Cross-cutting guard-rails (all phases)

1. **Human in the loop for anything irreversible** — payments, deletions,
   external communications, upgrades.
2. **Diff + eval report before every promotion**; rollback path always
   available.
3. **No self-modifying code.** The system tunes *data* (memory, statistics,
   bounded weights) and *proposes* configuration; humans change code.
4. **Everything audited with a reason**, tamper-evident.
5. **Honest self-description** enforced by the output screen (see
   [HONESTY.md](HONESTY.md)).
