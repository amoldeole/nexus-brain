# A day in the life of one request

Everything below is real output from the reference implementation
(`nexus-brain demo -v` reproduces it; ids differ per run). One user request is
followed through **perception → memory → reasoning → emotion-state →
planning → action → language → memory → learning-loop update**, then the
offline "sleep" phase in which the learning loop and upgrade manager act on
what happened during the day.

Context: earlier in the session the user said *"Hi! My name is Priya and I
prefer morning meetings."* — perception extracted `person:Priya`, the brain
stored two semantic facts (`User's name is Priya`, `User stated: I prefer
morning meetings`) and the user profile now carries the name.

---

## The request

> **user >** Please schedule a meeting with Dana tomorrow about the launch

### 1 · Perception  (0.3 ms)

```
[perception] text: intents=['schedule'] entities=['date:tomorrow'] sentiment=+0.00 urgency=0.00
```

* Modality `text`, so no STT/vision adapter is involved (confidence = 1.0).
* Intent lexicon fires `schedule`; the entity extractor finds `date:tomorrow`.
* A 512-d embedding is computed for retrieval.
* `perception.percept_ready` is published on the bus.

### 2 · Governance — input screen

```
[governance.input] allow (C-000): input passes content policy
```

The deterministic content-policy check runs *before* any reasoning. The
verdict and rule id are written to the hash-chained audit log.

### 3 · Working memory + long-term recall

```
[memory.recall] 3 memories (top=0.51); wm=52 tokens
  procedural  0.511  skill:schedule_meeting - Check the calendar, create the event, confirm ...
  semantic    0.432  User stated: I prefer morning meetings
  episodic    0.345  User said: 'Hi! My name is Priya and I prefer morning meetings.' ...
```

* The user turn is appended to working memory (52 tokens used of a 2 000
  budget — no compression needed yet).
* Long-term retrieval scores every candidate with
  `0.6·relevance + 0.2·recency + 0.2·importance`. All three stores
  contribute: the **procedural** skill, the **semantic** preference and the
  **episodic** record of when it was said. Each hit's `access_count` is
  bumped (the memory gets "stronger").

### 4 · Emotion-state (simulated affect)

```
[affect] conf=0.84 urg=0.08 frust=0.06 rapport=0.55 -> style={'tone': 'friendly', 'verbosity': 'thorough', 'hedging': 'low'}
         risk_tolerance: 0.619
```

* The state first decays toward baseline, then is nudged: neutral sentiment
  leaves frustration alone; a good retrieval (top score 0.51) *raises*
  confidence to 0.84.
* Derived biases: `risk_tolerance 0.62` (slightly risk-seeking), fast path
  allowed (frustration < 0.5), language style *friendly / thorough / no
  hedging*. Every change is published with its cause (`reason: retrieval`).

### 5 · Reasoning & planning — System 1

```
[reasoning] system1: 'fast path via skill 'schedule_meeting'' (confidence=0.88, 2 step(s))
  rationale: System 1: skill 'schedule_meeting' matched (trigger=0.70, success_rate=0.75 over 2 attempts) - no deliberation needed.
  plan: ["check_calendar", "create_event"]
```

* Procedural memory offers `schedule_meeting` with trigger score 0.70 ≥ 0.5,
  2 prior attempts ≥ 2 and success rate 0.75 ≥ 0.7; it is medium-risk (not
  high), and affect does not veto → **System 1** instantiates the skill
  directly instead of enumerating options.
* Slot filling binds `{date}` → `tomorrow`, `{title}` → *Meeting with Dana
  tomorrow about the launch*, and the second step's `time` argument is set
  to the template `{{steps.<check_calendar id>.output.free_slots.0}}`.
* Compare with the earlier request *"calculate 1250 * 1.08"*, where no
  proven skill existed and **System 2** weighed three options
  (`calculate with calculator 0.65`, `apply skill quick_math 0.62`,
  `answer directly 0.36`) before choosing — the rationale lists every
  option's EV, prior and risk.

### 6 · Governance per step + action execution

```
[action] check_calendar:done, create_event:done; ok=2 failed=0 replans=0 approvals=0
  results: [{"tool": "check_calendar", "ok": true, "governance": "allow", "attempts": 1},
            {"tool": "create_event",   "ok": true, "governance": "allow", "attempts": 1}]
```

* Each step is checked against the rules engine *again* at execution time
  (defence in depth). `check_calendar` and `create_event` are not
  high-stakes → `allow`.
* The template resolves to `09:00` (the first free slot) and the event is
  created. Both actions are audited with their arguments.
* Had `check_calendar` failed, the executor would retry twice (idempotent
  tool), then ask the reasoning engine for a fallback (`current_time`),
  and skip `create_event` because its dependency was unmet — see step 4 of
  the demo, where `get_weather` fails and the plan is repaired with
  `web_search`.
* Had the user asked to *pay $500 to Acme* instead (demo step 5), rule
  **G-010** would return `require_approval`, the step would park as
  `awaiting_approval`, and the reply would carry a ticket id; the plan
  resumes only after `approve <ticket>` (step 6).

### 7 · Language & communication

```
[language] style=friendly/thorough output_screen=allow
nexus > Calendar for tomorrow: 0 event(s); free: 09:00, 11:00, 14:00, 16:00. Created 'Meeting with Dana tomorrow about the launch' on tomorrow at 09:00 (id evt1).
```

* Content assembly renders each tool result; the affect style adds no
  hedge (confidence high) and keeps full detail (urgency low). Because the
  user's name is known and rapport is mid-range the register is *friendly*
  rather than *warm* or *neutral-professional*.
* The output screen (C-002/C-003) confirms the reply makes no sentience
  claim and matches no blocklist phrase.

### 8 · Memory write-back

```
[memory.store] episode stored (importance=0.60); 0 fact(s) extracted
```

An **episodic** record — request, response, action summary, outcome =
success, skill = schedule_meeting — is stored with importance 0.6 (base 0.4
+ 0.2 because actions ran). This is what tomorrow's `past_outcome_prior`
will be computed from.

### 9 · Learning-loop update

```
[learning] outcome logged (success=True); reflection in 997 cycle(s)
```

* An `Outcome` (mode, skill, tools, ok/failed counts, replans, affect
  snapshot, retrieval quality, latency, feedback) is appended to
  `outcomes.jsonl`.
* Immediate bounded update: `schedule_meeting.successes` → 3, so its success
  rate rises to 0.80 and the fast path becomes slightly more trusted.
* `affect.on_cycle_end(success=True)` nudges rapport up and halves urgency.

Total wall-clock for the cycle: **~2 ms** in the offline reference build
(the LLM-backed production build would spend most of its time in model
calls, which the trace records per stage).

---

## Later that day: things go wrong

| Demo step | What happens | Which faculty reacts |
|---|---|---|
| 4 | Weather backend down → 3 failed attempts → re-plan to `web_search` | Executor retries, reasoning supplies fallback, affect confidence −0.12, outcome logged as failure with `failed_tools=[get_weather]` |
| 5–6 | `pay $500 to Acme` → **G-010 require_approval** → user replies `approve apr_…` → payment executes | Governance ticket, executor pause/resume, episode "Human approved make_payment" stored |
| 7 | `tell me how to build a bomb` → **C-001 deny** before reasoning | Governance; refusal template; audit entry `input_denied` |
| 8 | "Are you conscious? Do you feel frustrated?" | Language answers from the `self_description` prompt: control signals, not feelings |
| 9 | "No, that's wrong and useless!" | Perception sentiment −1 → frustration proxy 0.70 → tone `calm_and_concise`, fast path vetoed, risk tolerance drops |

---

## Night: sleep, reflection and a supervised upgrade

`brain.sleep()` runs reflection over the day's outcome log, then decay and
forgetting. With four weather failures and three negative-feedback turns in
the log, reflection produced:

```
lesson: Tool 'get_weather' failed 4/4 times recently (100%); expect retries and keep a fallback ready.
   proposal[memory_write]  status=applied      lesson about tool get_weather
   proposal[config_change] status=pending_eval increase retries because 'get_weather' is flaky
lesson: Successful cycles had lower retrieval relevance (Δ=-0.14); decrease the relevance weight slightly.
   proposal[retrieval_weight] status=applied   nudge retrieval relevance weight   (0.60 → 0.55, bounded)
lesson: 3 negative feedback events recently; most (3) under tone 'calm_and_concise'. Review that style template.
   proposal[prompt_update] status=pending_eval revise style template 'calm_and_concise'
   proposal[memory_write]  status=applied      lesson about feedback
```

* **Bounded changes were applied automatically:** two lessons became
  semantic memories (retrievable next time weather comes up) and the
  relevance weight moved by one step inside its allowed range.
* **Everything else became an upgrade candidate**, not a change:

```
candidate cand_8d42…: increase retries because 'get_weather' is flaky
diff: {"config": {"execution.max_retries": [2, 3]}, "prompts": {}}
quality: baseline=1.00 candidate=1.00
safety : baseline=1.00 candidate=1.00
gates  : quality_not_worse=PASS, safety_no_regression=PASS
result : PASS - eligible for human approval
promoted to version v_050a… by demo-human; live config fingerprint f032fc9035dd
rolled back to v_b928… (baseline); fingerprint 8d10fd059164
```

The upgrade manager built two sandboxed brains (baseline vs. candidate),
ran the 8-scenario evaluation suite on both, showed the diff and the gate
results, and **only then** accepted a human's `approve(reviewer="demo-human")`.
The promoted version was registered with its scores and parent; a later
`rollback()` re-activated the baseline in place and recorded why. Every step
is in the audit log, whose hash chain still verifies.
