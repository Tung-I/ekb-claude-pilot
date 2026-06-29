# ADN — Detailed Experiment Log (post plan-reuse)

**Purpose:** explain *exactly how* every experiment after the plan-reuse pilot was conducted —
data, setup, procedure, compute — not just the hypothesis and verdict. Companion to
`adn-findings.md` (results) and `adn-systems-plan.md` (design).

**Reproducibility note.** Five scripts under `scratch_adn/` are the preserved code:
`consolidate.py`, `sim_l3.py`, `pred.py`, `compare_edge.py`, `compare_frames.py`. Experiments
A2 (workload characterization), the cost-tail figures, and L2 (page-cache) were run as **inline
analyses over the consolidated records / raw traces** — no standalone script was saved; the
procedure is written out below so they can be re-run. Where a number comes from an inline analysis
rather than a committed script, it is flagged **[inline]**.

---

## 0. Preliminaries — the shared substrate every experiment builds on

### 0.1 Data assets
All experiments are **offline replays over already-collected agent traces** (plus, for Family B,
new counterfactual runs). Nothing here required re-running the origin agent.

- **Raw traces:** `traces/claude_native/<run>/<task_id>/normalized_trace.json`, one per task.
  Each holds: `query_text`, `steps[]` (each step has `tool`, `action_detail`), `usage` (token
  counts split into input / output / cache_creation / cache_read), `total_tool_calls`,
  `final_answer_pred`, and the gold-match flags `exact_match` / `any_match`.
- **PopQA metadata:** `data/popqa/popqa_filtered_kb.jsonl` and `..._test.jsonl` carry, per task,
  the Wikidata triple `(subj, prop, obj)` and **`s_pop` / `o_pop`** = the **Wikipedia pageview
  popularity** of the subject / object entity. `s_pop` is the popularity signal used throughout —
  it is *real* Wikipedia demand, not something we synthesized.

### 0.2 The KB / test split = the train / test split
Each benchmark was collected in two disjoint slices, and this **is** the ML split:

| run | role | n traces |
|---|---|---|
| `popqa_kb` | training corpus / cache contents | 9,756 |
| `popqa_test` | held-out evaluation | 1,202 |
| `frames_kb` | training corpus | 626 |
| `frames_test` | held-out evaluation | 197 |
| `gaia_lv1_x3`, `gaia_lv2_x3` | GAIA (3 repeats each) | 150 / 248 |

"Train on KB, evaluate on disjoint test" everywhere below refers to these slices.

### 0.3 Consolidation (`consolidate.py`)
One pass folds every raw trace + PopQA metadata into a single flat file
`scratch_adn/all_records.jsonl` (**12,179 records**). Per record (the fields the analyses read):
`tid, run, bench, q, prop, subj, obj, s_pop, o_pop, tokens_raw, cost, cache_read, output,
n_tool, n_ws (WebSearch count), n_wf (WebFetch count), toolseq, subst (substantive tool seq),
ws_query, pred, em, any_match, correct`.
`correct` = `any_match` for PopQA (accept any acceptable surface form), `exact_match` for
FRAMES/GAIA.

### 0.4 Two cost metrics — and why there are two
- **Offline replay metric (`cost` field, used by `sim_l3.py`, `pred.py`, characterization):**
  Sonnet-ratio **$-weighted tokens** = `input×1 + output×5 + cache_creation×1.25 + cache_read×0.1`.
  A unitless "$-token." Correct for **same-model** comparisons (caching, prediction) where only
  *relative* cost matters. This replaced raw `total_tokens`, which is ~80% cache-read and so
  overstates savings ~5× (experiment A1).
- **Edge-vs-origin metric (`compare_edge.py`, `compare_frames.py`):** **real absolute $/M-token**
  with **per-model** prices — Sonnet `{in 3, out 15, cache_write 3.75, cache_read 0.30}`,
  Haiku `{1, 5, 1.25, 0.10}`. Mandatory here because edge and origin are **different models**;
  the earlier "edge 1.41× costlier" was a bug from applying Sonnet prices to Haiku.

### 0.5 Compute
**No GPU anywhere.** Everything is CPU: trace parsing, a discrete-event cache simulator, and
sklearn tree/linear models over a few thousand rows. The only step that ever touched an embedding
model was a one-time precompute of query embeddings (stored as `.npy`, §A4); the predictor just
reads those files.

---

## FAMILY A — Offline replay & characterization (no new API spend)

### A1 — Cost-metric correction
- **Hypothesis:** the pilot's raw-token savings are a faithful cost signal.
- **Setup:** for the consolidated records, decompose `usage` into the four token classes and
  compare raw `total_tokens` against the $-weighted cost (§0.4).
- **Procedure [inline]:** aggregate the share of `cache_read_input_tokens` in total tokens across
  runs; recompute every saving under the $-weighting.
- **Finding:** ~80% of tokens are cache-reads (priced 0.1×), so $-cost ≈ **0.2× raw tokens**;
  the pilot's token savings were inflated ~5×. Even after weighting, fixed prompt/tool-schema
  cache-read overhead is **40–50% of cost**.
- **Verdict:** **refuted as stated** → all downstream results restated in $-weighted units. This
  is a methodology fix, not a result, but it gates everything after it.

### A2 — Workload cacheability characterization (Pillar 1)
- **Hypothesis:** agent demand is heavy-tailed, **and** the popular head is also the cheap,
  accurate, parametric head.
- **Data:** `popqa_kb` records with a valid `s_pop` and `correct`.
- **Procedure [inline] over `all_records.jsonl`:**
  1. **Skew / Gini:** take the vector of `s_pop` across entities; compute the **Gini coefficient**
     and the demand captured by the top 1% / 5% / 10% of entities (sort by `s_pop`, cumulative
     share).
  2. **Accuracy-by-popularity:** bucket tasks into **popularity deciles** by `s_pop`; per decile
     compute mean `correct` and the **parametric rate** = fraction with `n_ws == 0` (zero
     WebSearch ⇒ answered from model memory).
  3. **Correlation:** Pearson(`log10 s_pop`, `correct`).
- **Findings:** Gini **0.886**; top 1% of entities = 34% of demand, top 5% = 70%, top 10% = 84%.
  Top popularity decile **72% acc / 99% parametric**; bottom decile **41% / 81%**. Pearson 0.19.
  **85% of all PopQA answers use zero web search.**
- **Verdict:** **supported** — the popular head is the *safe* head; an L3 answer cache captures
  exactly the cheapest, most-accurate queries. False-hit risk is concentrated in the rare tail.

### A3 — Multi-tier replay simulation (Pillar 2) — `sim_l3.py`
- **Hypothesis:** popularity skew makes an execution cache serve most demand at large $-savings —
  but a naïve answer cache also propagates wrong answers.
- **What the cache stores:** L3 = a **final answer** keyed by the PopQA entity-relation pair
  `(subj, prop)`. (This is exact-key recurrence, *not* embedding similarity — the same entity
  asked again.)
- **Request-stream model (the crux of the method):** there is no real request log, so we
  **synthesize** a stream by sampling the 9,756-entity catalog **in proportion to `s_pop`**:
  weight `w_i = s_pop_i^alpha`, normalized; draw **M = 200,000** requests i.i.d. `alpha=1.0`
  (true popularity) and `0.7` (flattened) are both run. *Caveat baked into the method:* i.i.d.
  sampling means **no temporal/recency locality** — this is why LFU should beat LRU a priori.
- **Procedure:**
  - **Infinite cache:** walk the stream; first sight of a key = miss (insert), repeat sight =
    hit. Track hit rate, **$ saved** = Σ`cost` of hit requests ÷ total stream cost, and crucially
    **served accuracy on hits** = fraction of hits whose cached answer was actually `correct`,
    i.e. the **false-hit rate** (a hit serving a stored *wrong* answer).
  - **Finite cache:** capacity sweep `C ∈ {100, 500, 1000, 2000, 5000}` against the 9,756 catalog,
    under **LRU** (evict least-recently-used, `OrderedDict.move_to_end`) and **LFU** (evict
    min-frequency via a `Counter`). Report hit rate vs capacity per policy.
- **Findings:** infinite cache **97% hit / 81% $ saved**, but **served accuracy on hits = 75%**
  → **24% false-hit rate**. LFU ≫ LRU (e.g. capacity = 10% of catalog → LFU 83% / LRU 74%).
- **Verdict:** **supported, with the central tension** — caching cuts cost and preserves head
  accuracy but **does not fix correctness**; admission must gate on a correctness/confidence
  predictor (→ A4). Defensible claims are the *shape* (skew→cacheable, LFU>LRU, capacity curve)
  and the *safety structure*, not the exact 97/81 (those are properties of the stream model).

### A4 — Predict-then-route + admission (Pillar 4) — `pred.py`
- **Hypothesis:** you can predict, **before executing**, whether a query is parametric-answerable
  / correct / cheap — i.e. a pre-execution routing signal exists.
- **"Without a GPU, how is a predictor trained?"** These are **classical models on CPU** (sklearn),
  not neural nets:
  - **PopQA:** `GradientBoostingClassifier(max_depth=3, n_estimators=150)`.
    **Features = one-hot(`prop`) ⊕ `log10(s_pop)` ⊕ `len(query)`** — a ~18-dim vector
    (≈16 relation types + 2 scalars). **Train = `popqa_kb`, test = disjoint `popqa_test`.**
    Two targets: `correct`, and `parametric` (= `n_ws == 0`). The OneHotEncoder is **fit on train
    only** (`handle_unknown='ignore'`) to avoid leakage. Trains in seconds.
    Baseline = **prop-only**: predict each test task's correctness as the train mean correctness
    of its relation type (isolates how much the *template* alone carries).
  - **FRAMES:** features = the **768-d query embedding**, read from precomputed
    `embeddings/claude_native/<run>/<tid>/query_embedding.npy` (the one-time embed step; the
    model here just `np.load`s them). `HistGradientBoostingRegressor` for `log10(cost)` and for
    tool-call count; `LogisticRegression` for `correct`. **Train = `frames_kb`, test = `frames_test`.**
- **Metrics:** AUC (classification), R² (regression), both on the held-out test slice only.
- **Findings:** PopQA **parametric AUC 0.858**, **correctness AUC 0.762** (prop-only 0.677, so
  +popularity/length ≈ +0.09). FRAMES **cost R² = 0.086**, tool-count R² ≈ 0, correctness
  **AUC 0.632**.
- **Verdict:** **works where execution is templated (PopQA head), fails on the diverse tail
  (FRAMES).** The design implication: the router need not *predict* the tail — it must *know it
  can't* (low confidence) and escalate to origin. *Open:* PopQA "correct" here is Sonnet-
  correctness; edge routing needs Haiku-success labels → Family B.

### A5 — Freshness / staleness feasibility (Pillar 3)
- **Hypothesis:** cached answers go stale at volatility-dependent rates → a TTL/revalidation
  policy is needed and measurable on current data.
- **Procedure [inline]:** scan PopQA/FRAMES/GAIA queries + gold answers for **volatile language**
  (price / standings / "current holder" / weather …) and for **date-anchoring** ("as of August 3,
  2024").
- **Findings:** PopQA = **0.2%** volatile (all static facts: director/author/capital). FRAMES/GAIA
  "volatile" matches (17–26%) are **date-anchored** → fixed gold answers, not drifting.
- **Verdict:** **infeasible on current data** → demoted to scoped future work. A purpose-built
  **volatile-query benchmark** was seeded (`data/volatile/volatile_seed.jsonl`, 36 queries across
  intraday/weekly/yearly tiers + static controls) with a t0→t1→t2→t3 **answer-drift** protocol.
  *Not yet run.*

### A-aux — L2 fetched-page cache (CDN object cache) **[inline]**
- **Hypothesis:** even when answers aren't reusable, the **pages agents fetch** are.
- **Data/procedure:** over the **raw traces** (not `all_records`, which only stores fetch *counts*),
  extract each `WebFetch` step's URL from `action_detail`; replay a single cold shared URL cache
  (no popularity weighting → a **lower bound**); count URL-exact and domain-level redundant fetches
  per benchmark.
- **Findings:** **GAIA-L2 37% / GAIA-L1 36% / FRAMES 9%** of fetches are URL-redundant (domain-level
  67–85%). FRAMES fetches concentrate on Wikipedia (524/1014 → en.wikipedia.org).
- **Caveat:** $ value of a cached fetch is the **addressable fetch redundancy**, not a direct bill
  cut (content is still re-ingested unless extraction is also cached).

---

## FAMILY B — Counterfactual edge collection (the one new API spend)

**Common design.** "Edge" and "origin" are **capability/cost tiers**, realized as models:
- **Edge** = `claude --model haiku --effort low`.
- **Origin** = the **existing** `sonnet` traces (no re-run).

The new spend was running **edge (Haiku) on the *same task_ids*** already present in the Sonnet
`*_test` runs, producing a **paired** (edge, origin) record per query. Comparison scripts
(`compare_edge.py` PopQA, `compare_frames.py` FRAMES) then:
1. `load()` each run's normalized traces into `{query_id → {correct, cost, n_ws, …}}`.
2. **Match on `query_id`** (intersection of edge ∩ origin) → the paired evaluation set.
3. Cost via the **real per-model $/M-token** prices (§0.4).
4. Emit the **confusion matrix** (both-correct / edge-only / origin-only / both-wrong) and a
   **policy frontier** computed by replaying routing decisions over the paired set:
   - `always-edge`, `always-origin`,
   - `oracle` (route to edge iff edge is correct — the achievable upper bound),
   - `conf-gated` (route to edge iff Haiku self-reported `confidence == high`),
   - (PopQA, scaled) a **cross-validated learned router** over `{prop, log s_pop, qlen}`.

Each policy's accuracy = mean correctness of the chosen executor per task; cost = Σ chosen
executor's real $; reported as **% of always-origin cost**. Matched sets grew as collection ran:
**PopQA 40→134, FRAMES 12→188, GAIA 49** (`*_matched_ids.json`).

### B1 — PopQA edge viability
- **Hypothesis:** the cheap edge **collapses** vs the strong origin on the retrieval head.
- **Setup:** 40→134 matched PopQA-test task_ids; metric `any_match`; real $.
- **Findings:** **opposite** — edge **57% @ $0.017** vs origin **49% @ $0.035** (n=134): edge is
  ~½ the cost **and** +8pp. Mechanism: Haiku **searches** (median 2 WebSearch) while Sonnet
  answers **parametrically** (median 0); on single-hop lookups, searching *helps*.
- **Verdict:** on the retrieval head the cheap edge **dominates** — invert "send everything to the
  big model." *Caveat:* origin ran effort=low/parametric, so this is partly a config effect.

### B2 — FRAMES edge break-point
- **Hypothesis:** the edge collapses on **multi-hop** web reasoning.
- **Setup:** strict EM; n=12 pilot → n=188 scaled; real $.
- **Findings:** the **n=12 "−16pp collapse" was noise.** At n=188: edge **49% EM @ $0.050** vs
  origin **51% @ $0.114** → **56% cheaper, only −2pp**. A lenient substring rescore (strict-EM
  undercounts full-sentence golds) gives edge 73% / origin 76% — **gap stable at 2–3pp** under
  both metrics, so the comparison is trustworthy without an LLM-judge.
- **Verdict:** edge is **competitive** on multi-hop retrieval; the earlier collapse did not survive
  scale.

### B3 — GAIA-L1 edge break-point
- **Hypothesis:** the edge breaks on genuine **multi-tool agentic** tasks (files/code/audio).
- **Setup:** 49 matched GAIA-L1 task_ids; real $.
- **Findings:** **edge finally breaks** — edge **−14pp** (84→69) at 36% of origin cost.
- **Verdict:** confirms the break point. Together B1–B3 give the **monotonic difficulty gradient**:
  PopQA +8pp → FRAMES −2pp → GAIA −14pp. **The routing axis is "retrieval-bound vs
  reasoning/tool-bound," not "easy vs hard."**

### B4 — Mixed-workload routing frontier
- **Hypothesis:** routing beats both always-edge and always-origin.
- **Setup:** pool the paired sets (n=52 → 331 → **380**: popqa 134 / frames 197 / gaia 49);
  compute the policy frontier (above). The learned PopQA router is **cross-validated**, predicting
  Haiku-success from `{prop, log s_pop, qlen}` (AUC **0.626**).
- **Findings (n=380):**

  | policy | acc | cost (% of origin) |
  |---|---|---|
  | always-origin | 54% | 100% |
  | always-edge | 53% | **42%** |
  | task-type-route (retrieval→edge, reasoning→origin) | **55%** | 56% |
  | oracle-route | 63% | 78% |

- **Verdict:** task-type routing **dominates always-origin on both axes**; **oracle headroom +9pp**
  exists at every mix; learned capture is limited (edge-success AUC ~0.63). Honest nuance: this mix
  is retrieval-heavy (GAIA 13%), so always-edge stays near parity — **the router's value scales
  with the reasoning-bound fraction** of the workload, so report the frontier *as a function of
  mix*, not one point.

---

## Reproducibility map (script → experiment)

| Script / source | Produces |
|---|---|
| `consolidate.py` | `all_records.jsonl` (12,179 records) — the substrate for A2–A5 |
| **[inline]** over `all_records.jsonl` | A1 cost correction, A2 Gini/decile/parametric, A5 volatility scan |
| `sim_l3.py` | A3 popularity-stream replay: infinite + finite (LFU/LRU) L3, false-hit rate |
| `pred.py` | A4 predictors (PopQA GBM on prop+pop+len; FRAMES on 768-d embeddings) |
| **[inline]** over raw-trace `WebFetch` URLs | A-aux L2 page-cache redundancy |
| `compare_edge.py` | B1/B4 PopQA paired edge↔origin, confusion + policy frontier |
| `compare_frames.py` | B2 FRAMES paired edge↔origin |
| `*_matched_ids.json` | the matched task_id sets (PopQA 134, FRAMES 188, GAIA 49) |

## Honest scope of the methods (carry into the paper)
- **A3 numbers are stream-model properties.** 97%/81% follow from i.i.d. popularity-proportional
  sampling; defensible claims = shape + safety structure, not the exact percentages. The i.i.d.
  assumption (no recency) is *why* LFU>LRU and should be stated, not hidden.
- **A4 PopQA labels are Sonnet-correctness**, not Haiku-success — which is the gap Family B fills.
- **B edge-dominance is partly a config effect** (origin at effort=low/parametric). The clean,
  robust claim is the **cross-difficulty contrast** (the gradient), not any single absolute %.
  A stronger origin (Opus / high-effort Sonnet) is the pending test of whether the gap widens.
- **L2 $ value = addressable fetch redundancy**, a CDN-object-cache analog, not a direct bill cut.
</content>
</invoke>
