# ADN Pilot Redesign — Routing Predictability, Savings Ceiling, and a Lightweight Discriminator

## Context

The project needs to redesign its pilot experiments around three concrete questions about an
**edge-assisted agent-serving pipeline** (edge = Claude **Haiku**; origin/cloud = Claude
**Sonnet/Opus**):

1. **Can we predict, from the query alone, whether to serve at the edge or escalate to cloud?**
2. **If routing were perfect, how much token/$/latency is actually saved vs cloud-only?**
3. **Can a lightweight post-hoc discriminator score an edge answer's trustworthiness** and reroute
   low-confidence answers to cloud?

**Decisions locked (this session):**
- **Edge identity = Claude Haiku** (sole edge). Prior `n=380` difficulty-gradient results stay valid.
- **V100 (16 GB) is used for the ML side only** — fine-tuning the predictor/discriminator and
  serving a small local LLM as a *lightweight judge*. **No local edge agent** is built (the Claude
  Code agent loop is a black box and not worth reimplementing for this pilot).
- **Latency is in scope as a supporting metric.** A read-only sanity check (done) shows wall-clock
  `total_latency_ms` tracks work — `corr(latency, tokens)` = 0.78 (PopQA) / 0.95 (FRAMES) / 0.73
  (GAIA) — so the distribution is usable after a small outlier-cleaning pass.

**The unifying frame.** The prize is the **oracle-routing headroom** (always-edge → oracle gap,
historically +8–9pp accuracy / large cost cut). There are two mechanisms to capture it, and the
three experiments map onto them:
- **Pre-execution routing** (Exp 1): decide before running. Cheap, but less information.
- **Post-execution discrimination** (Exp 3): run edge, then trust-or-escalate. More information
  (the answer + tool context exist), but you pay the edge cost even on escalation.
- **Exp 2** quantifies the ceiling both mechanisms chase, as a function of workload mix.

A coherent thesis connects them: *if per-query pre-execution routing is hard within a difficulty
regime (Exp 1 likely shows task-type is predictable but finer query-level difficulty is not), then
the post-execution discriminator (Exp 3) is the necessary mechanism to capture the within-regime
headroom.*

---

## Shared infrastructure (build first — all three experiments depend on it)

All new code lives under `scratch_adn/` (analysis) and a new `edge_serving/` package (training +
serving). Reuse, do **not** reimplement:
- Gold matching: `exact_match` / `any_match` in `runners/run_claude_task.py:629` and
  `runners/run_claude_popqa.py:554` (model-agnostic, importable).
- Trace consolidation pattern: `scratch_adn/consolidate.py` (extend, don't rewrite).
- Embedding generation: `tools/extract-query-embedding.py` (all-mpnet-base-v2, 768-d).

**S1 — LLM-judge correctness relabeling (prerequisite for clean labels).**
String-match labels undercount full-sentence answers (known: FRAMES strict-EM ~73% vs 76% under
substring). Edge-success and discriminator labels must be clean. Run a **strong** judge (Claude
Sonnet/Opus via the existing API path) over every (question, gold, edge-answer) and (…, origin-answer)
to produce a binary `judged_correct`. Store alongside `any_match`/`exact_match`; report agreement so
the relabel is auditable. *(The strong judge is for label quality; a separate small local judge is
evaluated as a deployable baseline in Exp 3 — do not conflate them.)*

**S2 — PopQA embeddings.** `embeddings/claude_native/` covers FRAMES + GAIA only. Regenerate PopQA
query embeddings with `tools/extract-query-embedding.py` so Exp 1 has a uniform 768-d feature across
all three benchmarks.

**S3 — Latency cleaning.** Fit `latency ~ tokens + steps` per run; flag traces with a large positive
residual (high latency unexplained by work = contention/idle signature). Delete and re-run *only*
those (expected: a handful per run). Add a decomposition `total_latency = tool_latency (network,
model-independent) + model_latency (inferred from inter-step gaps)` so latency claims separate the
model from the shared tool round-trips.

**S4 — Consolidated paired dataset.** Extend `all_records.jsonl` into a per-task table keyed by
`query_id` with: benchmark, base_task_id (for GAIA paraphrase-safe splitting), edge/origin
`{judged_correct, cost_real$, latency_s, n_ws, n_wf, confidence, answer_text, tool_context_text}`,
and the 768-d embedding path. This single table feeds all three experiments.

---

## Experiment 1 — Is the routing decision query-predictable? (defeating the task-type shortcut)

**Hypothesis.** Edge-success (`Haiku judged_correct`) is predictable from the query *beyond* coarse
benchmark/task-type identity.
**Null worth publishing.** Routing is predictable only at task-type granularity; finer query-level
difficulty is not pre-execution-predictable → the system should route by a cheap task-type classifier
and rely on the Exp 3 discriminator for within-regime uncertainty.

**Label.** `y = edge judged_correct ∈ {0,1}` (the oracle "edge suffices" decision).

**The design explicitly separates the shortcut from the real signal:**
1. **Shortcut baseline** — predict `y` from **benchmark-id one-hot only**. This *is* the current
   task-type-route policy; expected to carry real but coarse signal.
2. **Within-regime test (the scientific core)** — train+evaluate `y` predictor **inside each
   benchmark separately**. AUC ≫ 0.5 within FRAMES/GAIA ⇒ genuine query-level predictability;
   AUC ≈ 0.5 ⇒ the honest null above.
3. **Leave-one-benchmark-out (LOBO) transfer** — train on two benchmarks, test on the held-out one;
   measures whether "edge-difficulty" generalizes vs is benchmark-specific.
4. **Decomposition headline** — report `AUC(benchmark-id) → AUC(+query features)` per benchmark and
   pooled, so the marginal value of per-query features is explicit (pooled AUC is inflated by easy
   cross-benchmark separation — never report pooled alone).

**Feature families (ablated):** (a) benchmark-id; (b) 768-d mpnet embedding; (c) interpretable
(length, entity `s_pop`, #named-entities, question-type, multi-hop/computation lexical cues);
(d) **self-consistency** — sample Haiku `k=5×` at T>0, use answer-agreement as a difficulty proxy
(available via repeated API calls; **note: Anthropic exposes no token-logprobs**, so logprob features
are out under the Haiku-edge choice).

**Predictor families:** logistic/GBM on features (CPU), a **fine-tuned small encoder**
(DeBERTa-v3-base over query text, V100), and a **small LLM zero/few-shot router** (local vLLM model)
for comparison.

**Splits.** Strict split by `base_task_id` (GAIA x3 paraphrases must not straddle train/test).
**Deliverable.** Per-benchmark + LOBO AUC table with the id-only→+features decomposition; calibration
of the best predictor. **Kill criterion:** if +query-features adds < ~0.03 AUC within every benchmark,
declare the task-type-only result and pivot weight to Exp 3.

---

## Experiment 2 — The savings ceiling (oracle and realistic cascade)

**Hypothesis.** Edge-assisted serving beats cloud-only on cost (and, with caveats, latency) at
≤ ε accuracy loss, and the win scales with the retrieval-bound fraction of the workload.

**Policies replayed over the paired table (no new agent runs needed for the oracle part):**
- `cloud-only` (baseline), `edge-only`, `oracle-route` (perfect pre-exec predictor = Exp 1 ceiling),
- `predicted-route` (Exp 1's actual predictor at operating thresholds),
- **`cascade`** (the real system): edge → discriminator (Exp 3) → escalate low-confidence to origin.
  Cost = `C_edge(all) + escalation_rate · C_origin`; this cannot beat oracle because you pay edge
  even on escalated queries — quantifying that gap is a deliverable.

**Metrics:** real per-model `$` (Haiku/Sonnet price tables already in `compare_edge.py`), token
breakdown (in/out/cache), **end-to-end latency mean + p95** (now usable post-S3), accuracy.
**Savings decomposition** (stacked): L3 cache-hit (no model call, popularity-driven) / edge-served /
origin-escalated.
**Workload-mix sensitivity:** report the frontier as a **curve over the reasoning-bound fraction**,
not one point (the prior single mix was retrieval-heavy and flattered always-edge).
**ε-knob:** savings at accuracy ≥ origin − ε for ε ∈ {0, 2, 5} pp.

**Honest latency framing (from the sanity check):** edge is cheaper but **slower** on PopQA
(tool-round-trip-bound: Haiku 15.9s vs Sonnet 6.4s median); comparable on FRAMES. Headline is "cost
saved, latency neutral-to-worse on the retrieval head unless tool calls are cached (L2)" — not "edge
is faster." Origin latency is API-bound; report it with variance and lean on the cost result.

**Deliverable.** Policy × {cost, latency, accuracy} table across mixes; the stacked savings figure;
the oracle-vs-cascade gap (= the value left on the table by imperfect discrimination → motivates Exp 3).

---

## Experiment 3 — A lightweight trustworthiness discriminator

**Hypothesis.** A lightweight model conditioned on (query, edge tool-context, edge answer) predicts
edge-answer correctness well enough that rerouting its low-confidence cases recovers much of the
oracle headroom from Exp 2 — and a cheap signal approaches an expensive LLM-judge.

**Training data.** Edge traces with **S1 judged_correct** labels. Input context is available in the
traces: `query_text`, per-step `action_detail` + `tool_result` (search queries/results, fetched page
text), `final_answer_pred`, self-reported `confidence`.

**Signals, cheap → expensive (ablated):**
- **B0 self-reported confidence** (baseline; known coarse — Haiku is "high" 90% of the time).
- **B1 self-consistency** (answer-agreement across `k` Haiku samples; reuse Exp 1 feature).
- **B2 learned encoder discriminator (the core deliverable)** — fine-tune DeBERTa-v3-base/-large on
  `[query ⊕ truncated tool-context ⊕ answer] → P(correct)` on the V100. Lightweight + self-hostable.
- **B3 small local LLM-as-judge** — a vLLM-served ~7B model prompted to verify the answer given
  context; the deployable-judge ceiling (distinct from the strong relabel-judge in S1).

**Evaluation:**
- **Selective prediction** — risk-coverage curve / AUROC for detecting wrong answers (core metric).
- **Calibration** — ECE + reliability diagram (is `P(correct)` usable as a reroute threshold?).
- **System-level (the payoff)** — sweep reroute threshold τ; plot accuracy vs cost; compare to
  always-trust-edge (no discriminator) and the oracle discriminator. **Does B2 close the
  always-edge → oracle gap (Exp 2)?**
- **Cost-of-discriminator** — its own FLOPs/latency vs the savings it unlocks (a judge as costly as
  origin defeats the purpose; B2 must be ~free).
- **Generalization** — train on PopQA+FRAMES, test on GAIA (does trust-estimation transfer?).

**Deliverable.** Risk-coverage + reliability plots per signal; the threshold-swept accuracy/cost
frontier vs always-edge/oracle; the cheap-approaches-expensive comparison (B2 vs B3).

---

## vLLM ↔ Claude cooperation (concrete, under the locked config)

No direct coupling and no local edge agent. Three independent backends, orchestrated by a thin
Python driver:
- **Edge & origin agents = Claude** (Haiku / Sonnet-Opus) via the existing
  `runners/run_claude_*.py` API path — unchanged.
- **Strong relabel-judge (S1) = Claude** (Sonnet/Opus) via API.
- **V100 / vLLM = ML only:** (a) DeBERTa encoder fine-tune (plain HF/PyTorch on GPU); (b) a small
  local LLM (e.g. Qwen2.5-7B-Instruct-AWQ, 4-bit ≈ 6 GB) served via `vllm serve … --port 8000`
  (OpenAI-compatible) used as the **deployable lightweight judge (B3)** and the **Exp 1 zero-shot
  router**.
- Claude Code CLI is **not** pointed at vLLM. The orchestrator calls each backend via its native API.

**V100 phase sequencing (16 GB can't host a 7B server + encoder training at once):**
Phase A — serve the 7B (B3/router inference), dump scores. Phase B — free GPU, fine-tune DeBERTa
(trivial footprint). Encoder fits with room to spare; the 7B AWQ leaves headroom for KV cache.

---

## Data collection needs

- **PopQA scale-up (cheap, recommended):** ~1,686 untouched test tasks remain; run more Haiku edge to
  grow `n` for Exp 1/3 (origin Sonnet already collected for most).
- **FRAMES:** saturated (~824 total, 197 test already paired) — no scale-up.
- **GAIA hard-class (OPEN DECISION — see below):** only ~161 *unique* GAIA tasks (51 L1 / 84 L2 /
  26 L3). This is the binding constraint on Exp 1's within-GAIA / LOBO power and Exp 3's
  generalization test.
- **Latency re-runs:** only the S3-flagged outliers.

---

## Open decision (recommend default, confirm on approval)

**How to handle the scarce reasoning-bound (hard) class?** Recommended default: **collect more** —
run Haiku edge on **GAIA L2 + L3** (and pair with origin), optionally add one more agentic/reasoning
benchmark, to make the within-regime and LOBO tests statistically meaningful. Alternatives:
(b) use the ~161 tasks + x3 paraphrases (split by `base_task_id`) and report small-n CIs;
(c) carve the hardest multi-hop FRAMES subset as the hard class. This is the one knob I'd like
confirmed before execution.

---

## Risks & honest caveats (carry into the paper)
- **No logprobs under Haiku-edge** → uncertainty signals limited to self-report + self-consistency
  (the local 7B's logprobs apply only to the B3 judge, not the edge generator).
- **Latency comparison is asymmetric** (local-clean edge vs API-bound origin) — report cost as
  primary, latency as supporting with variance; the PopQA "edge slower" result is real and stated.
- **Label noise** is handled by S1 but the strong judge is itself imperfect — report judge↔string
  agreement.
- **GAIA small-n** bounds the most interesting (reasoning-bound) claims regardless of the decision
  above; state CIs.
- **Self-consistency costs k× edge calls** — fold its cost into the Exp 2 accounting if used in the
  deployed cascade.

---

## Verification (how we'll know each piece works)
- **S1:** judge↔string-match agreement reported; spot-check 20 disagreements by hand.
- **Exp 1:** per-benchmark + LOBO AUC table reproduces; the id-only vs +features decomposition is
  non-degenerate; splits are leakage-free (assert no `base_task_id` straddles train/test).
- **Exp 2:** policy frontier reproduces the prior n=380 ordering (always-origin < always-edge cost;
  oracle headroom > 0) and extends it with latency + mix curves; numbers tie out against
  `compare_edge.py` on the overlapping subset.
- **Exp 3:** risk-coverage AUROC > confidence-baseline; threshold sweep recovers a measurable
  fraction of the Exp 2 oracle gap; B2 (encoder) within a stated margin of B3 (LLM-judge) at a
  fraction of the cost.
- End-to-end: a single `make`-style script regenerates the paired table → all three experiment
  figures from `all_records.jsonl` + traces.
