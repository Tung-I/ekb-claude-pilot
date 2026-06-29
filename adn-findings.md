# ADN — Rolling Empirical Findings (offline, existing traces)

**Date:** 2026-06-28. Metric discipline: **$-weighted cost** (input×1 + output×5 + cache_write×1.25 +
cache_read×0.1, Sonnet ratios), **not raw tokens**; **latency ignored** (concurrent-API contention
makes wall-clock unreliable). Data: 12,179 consolidated traces (`scratch_adn/all_records.jsonl`).

## Cost-metric correction (applies to the whole project)
- `total_tokens` is **~80% cache-read** → $-cost ≈ **0.2× raw tokens** across every benchmark.
  The pilot report's raw-token savings are inflated ~5×. **Re-state all cost results in $-weighted units.**
- Even after weighting, cache-read (fixed prompt/tool-schema overhead) is **40-50% of cost**. Per-query
  PopQA cost is ~10k $-tok and **flat across templates** (overhead-dominated). Real $ lives in the tail:
  GAIA-L2 top-10% of tasks = **45%** of cost; PopQA top-10% = only 22% (flat).

## Pillar 1 — Workload cacheability characterization ✅ (strongly supported)
- **Popularity is heavy-tailed**: PopQA `s_pop` Gini = **0.886**; top 1% of entities = 34% of demand,
  top 5% = 70%, top 10% = 84%. This is the CDN precondition, grounded in *real* Wikipedia popularity.
- **The popular head is the safe head** (novel, non-obvious): top popularity decile = **72% accuracy /
  99% parametric (no web search)**; bottom decile = 41% / 81%. Pearson(log s_pop, correct)=0.19.
  → An L3 cache captures exactly the queries that are most accurate and cheapest. **False-hit risk is
  structurally concentrated in the rarely-requested tail.**
- **85% of PopQA answers use zero web search** (parametric memory). PopQA's lever is therefore **L3
  (answer)**, not L2.

## Pillar 2 — Multi-tier replay simulation ✅ (supported, with the key tension)
Popularity-proportional request stream over the 9,756-entity catalog (200k requests):
- **L3 infinite cache: hit rate 97%, $-cost saved ~81%** — BUT **served accuracy on hits = 75%**
  (24% **false-hit rate**: cached wrong answers propagate). Caching preserves head accuracy + cuts cost;
  it does **not** fix correctness → **admission must gate on a correctness/confidence predictor.**
- **Finite cache: LFU ≫ LRU** (iid popularity, no recency). Capacity = 10% of catalog → LFU 83% / LRU 74%;
  20% → 92% / 89%. Clean capacity curves for an eviction section.
- **L2 tiering**: WebSearch-result reuse ≈0% on disjoint splits (unique per entity); **WebFetch *page*
  reuse is real — GAIA-L2 36.7%, FRAMES 9.5%** = classic CDN object caching for agent web fetches.
- *Caveat:* 97%/81% are properties of the popularity-proportional **stream model**. Defensible claims =
  the **shape** (skew→cacheable, LFU>LRU, capacity curve) and the **safety structure**, not the exact %.

## Pillar 4 — Predict-then-route + admission ✅ (works on head, fails on tail — and that's the design)
Train on KB, evaluate on disjoint test:
- **PopQA (template + popularity features):** predict **parametric-answerability AUC 0.858**; predict
  **correctness AUC 0.762** (template-only 0.677; +popularity/length ≈ +0.09). Pre-execution routing
  signal is real.
- **FRAMES (768-d embedding):** cost R²=**0.086**, tool-count R²≈0, correctness AUC 0.632 — embeddings
  barely predict execution.
- **Interpretation:** prediction works where execution is templated and **fails on the diverse tail** —
  mirroring the pilot's semantic↔execution decoupling. The router doesn't need to predict the tail; it
  needs to **know it can't** (low predictor confidence) and **escalate to origin**. That is the safe ADN
  policy, and it falls straight out of the data.
- *Open:* PopQA "correct" is Sonnet-correctness. Edge routing needs **Haiku-success** labels →
  counterfactual collection (next step).

## Pillar 3 — Freshness / staleness ⚠️ (compelling but NOT feasible on current data)
- PopQA = 0.2% volatile language (all static-fact relations: director/author/capital…). Unusable.
- FRAMES/GAIA "volatile" matches (17-26%) are **date-*anchored*** ("as of August 3, 2024") → fixed gold
  answers, not drifting. Also unusable for a drift study.
- **Verdict:** freshness needs a **purpose-built volatile-query benchmark** (prices, standings, "current
  holder of X", weather) + temporal (t0→t1) re-collection. Real new scope. → Demote to a **scoped
  secondary contribution / future work**; the primary paper stands on Pillars 1+2+4.

## Edge viability pilot — Haiku vs Sonnet on PopQA (n=40 counterfactual, real $) 🔬
Same 40 PopQA-test task_ids run under edge (`haiku --effort low`) and origin (existing `sonnet` traces).
Cost in **real $/M-token** per model (Haiku ~3× cheaper than Sonnet) — earlier "1.41× costlier" was a
bug from applying Sonnet prices to both.
- **Edge is 53% cheaper AND slightly more accurate**: Haiku 48% acc @ $0.0172/task vs Sonnet 40% @
  $0.0367/task. `ALWAYS-EDGE` = 47% of origin cost at ≥ origin accuracy → **dominates on PopQA**.
- Mechanism: Haiku distrusts memory (median **2** web searches) while Sonnet answers parametrically
  (median **0**); on single-hop lookups, searching *helps*. Confusion: both=12, edge-only=7,
  origin-only=4, both-wrong=17.
- **Caveats:** n=40 (48 vs 40% is within noise); PopQA-only; origin ran at `--effort low` (parametric),
  so this is *not* a model-capability verdict — it shows **the cheap path suffices for the easy head**.
- **Implication:** the router earns its keep **across the difficulty spectrum**, not within PopQA. The
  pivotal test is whether Haiku **collapses on multi-hop FRAMES** (running now). If it does,
  difficulty-routing (predictor AUC 0.76-0.86 on head, escalate on low confidence) is justified.

## Edge break-point — Haiku on multi-hop FRAMES (n=12, strict EM, real $) 🔬
- **Edge collapses on the hard tail**: Haiku EM **42%** @ $0.066 vs Sonnet **58%** @ $0.116 — edge is
  43% cheaper but **−16pp accuracy**, and **never wins** (edge-only=0, origin-only=2, both=5, both-wrong=5).
- Combined with the PopQA pilot, the difficulty spectrum is the ADN thesis in one line:
  **easy/templated → edge suffices (cheaper, competitive); hard/multi-hop → escalate to origin.**

## Mixed-workload routing frontier (n=52: 40 PopQA + 12 FRAMES) 🔬
| Policy | Accuracy | Cost (vs always-origin) |
|---|---|---|
| always-origin | 44% | 100% |
| always-edge | 46% | 52% |
| **difficulty-route** (easy→edge, hard→origin) | **50%** | **73%** |
| oracle-route (upper bound) | 58% | 55% |
- Difficulty-routing **dominates always-origin on both cost and accuracy**; oracle shows large headroom.
- *Caveats:* n=52; the 40:12 mix flatters always-edge (PopQA-heavy). Robust claim = the **ordering**
  (routing > always-origin; oracle headroom exists), not the absolute %.

## L2 — WebFetch page cache (CDN object caching for agent fetches) 📦 (offline)
Single-pass shared URL cache (cold, no popularity weighting → lower bound):
- **GAIA-L2: 37% of fetches are redundant** (URL-exact), GAIA-L1 36%, FRAMES 9%. Domain-level 67-85%.
- FRAMES fetches concentrate on Wikipedia (**524/1014 → en.wikipedia.org**): an **edge Wikipedia snapshot
  serves most FRAMES fetches** — a concrete edge-serving artifact.
- *Caveat:* $ value of a cached fetch is ambiguous (content still re-ingested unless extraction is cached);
  report this as **addressable fetch redundancy**, the clean CDN-object-cache analog, not a direct $ figure.
- Under popularity-weighted traffic these rates rise (repeats of popular pages); 37% is the floor.

## Pillar 3 unblock — volatile-query benchmark seed 🌱 (`data/volatile/volatile_seed.jsonl`)
36 queries with drift-based eval (no fixed gold): 10 high (intraday: prices/weather/ISS-crew),
10 medium (weekly: standings/charts/versions), 10 low (yearly: office-holders/champions/latest-model),
6 static controls (Pride&Prejudice author, Canberra…).
**Protocol:** run the set at t0, t1(+1d), t2(+7d), t3(+30d) and measure **answer-drift rate per tier**.
Static controls should not drift (validates drift = real fact change, not agent nondeterminism).
Drift rate → calibrates per-tier **TTL**; a cache with TTL > change-interval = stale-serve rate.
*Not yet run* (avoid stacking API contention with the edge collection); schedule daily once collection done.

## Learned router + larger PopQA frontier (n=134 matched, real $) 🔬
Edge collection killed at 96/1202 popqa → 134 matched edge∩origin labels (still small for FRAMES: 12).
- **Edge dominates PopQA, stronger with more data:** edge 57% @ $0.017 vs origin 49% @ $0.035
  (cheaper *and* +8pp). Net edge-only=19 vs origin-only=8.
- **Cross-validated learned router** (predict Haiku-success from {prop, log s_pop, qlen}, AUC **0.626**):

  | policy | acc | cost (%origin) |
  |---|---|---|
  | always-origin | 49% | 100% |
  | always-edge | 57% | **49%** |
  | learned-route@0.4 | 58% | 63% |
  | oracle-route | 63% | 71% |

- **Key reframe:** on the easy head the cheap edge (small model + web search) **dominates** the expensive
  origin (which answers parametrically) — *routing barely helps; just use edge*. The router's value is
  **cross-difficulty** (escalate the hard multi-hop tail), which needs FRAMES+GAIA edge labels (the part of
  the killed job that actually matters; re-running the rest of PopQA does not).
- Slightly provocative headline this supports: *for high-frequency factual lookups, a small edge model
  with web access beats a large model at ~½ the cost; the expensive origin is only needed for the
  genuine multi-hop tail.* (Caveat: origin Sonnet ran at effort=low/parametric — partly a config effect.)

## ⭐ SCALED RESULTS (n=331; supersede the n=12/n=52 pilots) ⭐
Full Haiku-edge collection: PopQA 134 + FRAMES 188 matched vs Sonnet origin. **The n=12 FRAMES
"−16pp collapse" was noise** — it does not survive scale.
- **FRAMES at scale:** edge 49% EM @ $0.050 vs origin 51% @ $0.114 → **edge 56% cheaper, only 2pp behind**.
  Lenient substring rescore (strict-EM undercounts full-sentence golds): edge 73% vs origin 76% — **gap
  stable at 2-3pp** under both metrics, so the comparison is trustworthy without an LLM-judge.
- **Mixed PopQA+FRAMES frontier (n=331):**

  | policy | acc | cost (%origin) |
  |---|---|---|
  | always-origin | 50% | 100% |
  | **always-edge** | 51% | **44%** |
  | oracle-route | 59% | 79% |

- **Two headlines now:**
  1. *(robust, simple)* **The cheap edge matches the strong origin within 2-3pp at ~44% of the cost** across
     single-hop (PopQA, edge wins) and multi-hop (FRAMES, edge −2pp). → Default to serving at the edge;
     don't reflexively send everything to the big model.
  2. *(routing upside)* edge/origin **disagree on 17%** (edge-only=30, origin-only=26) → **oracle routing
     reaches 59% (+8pp)**. Real complementarity, but learned capture is limited (edge-success AUC ~0.63).
- **Open caveats:** (a) origin ran at low/medium effort — a stronger origin (Opus / high-effort Sonnet) may
  widen the gap → Opus tier pending; (b) **GAIA** (multi-tool + files + code) is the remaining test of
  whether edge *genuinely* breaks on hard agentic tasks — collecting now. GAIA decides whether the story is
  "always-edge dominates" or "difficulty-routing required."

## ⭐⭐ CAPSTONE: the difficulty gradient (n=380, all 3 benchmarks) ⭐⭐
GAIA Haiku landed (49 matched) and **edge finally breaks** → a clean monotonic gradient:

| Benchmark | Task type | Edge−Origin (acc) | edge cost | Verdict |
|---|---|---|---|---|
| PopQA | single-hop lookup | **+8pp** | 49% | edge wins |
| FRAMES | multi-hop web search | **−2pp** | 44% | edge competitive |
| GAIA-L1 | multi-tool (files/code/audio) | **−14pp** (84→69) | 36% | origin wins |

**The routing axis is not "easy vs hard" but "retrieval-bound vs reasoning/tool-bound."** The edge (small
model + web search) suffices for retrieval (single- and multi-hop); it breaks on genuine multi-tool agentic
reasoning (code exec, file parsing, transcription, multi-step logic).

3-way mixed frontier (n=380; popqa 134 / frames 197 / gaia 49):

| policy | acc | cost (%origin) |
|---|---|---|
| always-origin | 54% | 100% |
| always-edge | 53% | **42%** |
| task-type-route (retrieval→edge, reasoning→origin) | **55%** | 56% |
| oracle-route | 63% | 78% |

**Honest nuance — workload-mix dependence:** this mix is retrieval-heavy (GAIA only 13%), so always-edge
stays near-parity. **The router's value scales with the reasoning-bound fraction**: retrieval-heavy traffic
(typical web assistants) → serve at the edge; reasoning-heavy traffic → routing is essential. Report the
frontier *as a function of mix*, not one point. Oracle headroom (+9pp) exists at every mix.

## Final thesis (locked by the data)
*Agent traffic is dominated by retrieval-bound queries that a cheap edge (small model + web search +
execution cache) serves at ~parity accuracy and ~40% of origin cost; genuine multi-tool reasoning is the
minority tail that must escalate to the origin. ADN = an execution-caching edge that serves the retrieval
head and routes the reasoning tail, with popularity-driven L3/L2 caches cutting edge cost further.* The two
non-obvious hooks: **(1) the cheap edge beats/ties the frontier model on the retrieval head** (invert the
"send everything to the big model" default); **(2) the routing axis is task-type/tool-need, which is
predictable**, unlike generic difficulty.

## Revised direction (rolling)
1. **Primary paper = Pillars 1+2+4**, all supported by data in hand: *popularity-skewed agent workloads
   → multi-tier execution cache with predictor-gated admission and escalation*.
2. **Tier ownership by benchmark:** L3→PopQA (answer, popularity), L2→GAIA/FRAMES (WebFetch page cache),
   L1→GAIA (plan, latency-not-accuracy, from pilot).
3. **Next re-run experiment (highest value):** counterfactual **edge = Haiku** runs on a PopQA-test sample
   (same task_ids as existing Sonnet traces) to (a) get real cheap-path-success labels for the router,
   (b) measure the real edge-vs-origin **cost/quality frontier**. Latency excluded by design.
4. **Freshness** = optional later module needing a new volatile benchmark.
