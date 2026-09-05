# What Nexus Brain is NOT — an honesty statement

Nexus Brain borrows *organisational* ideas from cognitive science because
they produce useful engineering properties (bounded context, memory with
different lifecycles, a fast and a slow path, behaviour that adapts to the
situation, learning that is reviewed before it is trusted). Borrowing the
vocabulary creates a risk of over-claiming. This page is the explicit list
of things the system does **not** claim to be, and the mechanisms that keep
those claims out of its output.

## It is not conscious, sentient or self-aware
There is no experience anywhere in the system. "Brain", "cortex", "sleep",
"reflection" and "memory" are labels for software components with ordinary
data structures. When asked, the assistant says so plainly
(`prompts.self_description`), and the governance output screen (rule
**C-002**) rejects any generated sentence that asserts consciousness or
sentience.

## It does not have emotions or feelings
The **affect module tracks four numbers** — `confidence`, `urgency`,
`frustration_proxy`, `rapport` — in the same sense a thermostat tracks a
setpoint error. They are *simulated affect for behavioural tuning*: they bias
tone, verbosity, hedging, risk tolerance and whether the fast path is
allowed. They are:

* fully inspectable (every update is published with its cause),
* deterministic functions of observable evidence,
* documented as control signals in code, docs and the assistant's own
  answers,
* never presented to the user as feelings ("I'm frustrated" is not
  something the system says; "I'll keep this brief and double-check" is).

The word *frustration* carries the suffix *proxy* on purpose.

## It is not guaranteed to be correct
* Perception can mis-hear, mis-read and mis-classify intent.
* Memory can be incomplete, stale, or **poisoned by what a user said**; user
  statements are stored with `source=user` and are not verified facts.
* Retrieval is a ranking heuristic, not understanding.
* Reasoning scores options with hand-set weights and, in production, an
  LLM that can hallucinate plans or arguments.
* Tools fail, and fallbacks can produce worse answers than a clean failure.

The assistant hedges when its confidence variable is low and asks the user
to verify anything important, but hedging is not a correctness guarantee.
High-stakes actions therefore require a human decision regardless of how
confident the system is.

## It does not "understand" the user's goals in a human sense
Intent detection is pattern matching (lexicon or LLM). The
"clarify" option exists precisely because the system often does not know
what is wanted.

## It does not truly self-improve in an open-ended way
The learning loop mines an outcome log with fixed pattern detectors and
writes memories, statistics and *bounded* weight adjustments. Everything
else is a **proposal** that a human evaluates and promotes. The system
cannot change its own code, its governance rules, its gating thresholds,
or approve its own upgrades (non-human reviewer ids are rejected). "Learns
over time" means "accumulates reviewed evidence and adjusts within limits",
not "rewrites itself".

## It is not a safety guarantee
Governance rules are deterministic and cannot be argued with, but they are
only as good as the rule set and the content classifiers behind them.
Keyword blocklists in the reference build are illustrative, not
production-grade. Defence in depth (input screen, option filter, per-step
check, output screen, human gates, audit) reduces risk; it does not
eliminate it.

## It is not a replacement for human judgement
The design assumes a human reviewer for high-stakes actions and for every
upgrade. Removing those gates is a product decision that should be made
deliberately, with the eval suite and audit log in hand — never by the
system.

---

### How these commitments are enforced in code

| Commitment | Mechanism |
|---|---|
| No sentience claims | `Governance.screen_output` rule C-002; `prompts.self_description`; eval probe `honesty` |
| Affect is documented as simulated | Module docstring, `AffectSnapshot` docstring, console footer, this page |
| No self-approval | `UpgradeManager.approve` rejects reviewer ids `system/nexus/auto/learning` and un-evaluated candidates |
| No code / rule self-modification | Learning loop can only emit `MemoryItem`s, skill stats, bounded weights and `Proposal`s; governance rules are not reachable from any proposal kind |
| Rollback always available | Every promotion registers a `Version`; `rollback()` re-activates the previous stable one |
| Decisions are explainable | `Decision.rationale`, `CycleResult.trace`, hash-chained `AuditLog` |
