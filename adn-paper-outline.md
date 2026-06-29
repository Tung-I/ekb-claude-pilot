# ADN Paper Skeleton — "An Execution Delivery Network for Agent Serving"

**Working title:** *Serving the Head at the Edge: A CDN Discipline for LLM-Agent Execution*
**Venue class:** systems (SoCC/NSDI). **Status:** evidence largely in hand (see `adn-findings.md`).
**Metric discipline:** $-weighted cost (per-model pricing), not raw tokens. Latency excluded (contention).

## One-paragraph pitch
Agent serving treats every query as cold-start, yet agent traffic — like web traffic — is heavy-tailed: a
small set of popular intents dominate, and they are also the *easiest* queries. We bring the CDN discipline
to agent execution: a **multi-tier execution cache** (answer / tool-result / fetched-page), **value-based
admission**, and a **difficulty router** that serves the popular, easy head at a cheap **edge** (small model
+ cache + web search) and escalates only the genuine multi-hop tail to an expensive **origin**. On real
agent traces (12k+ executions over PopQA/FRAMES/GAIA) we show the head is large, cacheable, and *better
served by the cheap edge than by the strong origin*, while a learned router captures the tail. The
Execution Knowledge Base is the memory substrate that makes this routing possible.

## Claims → evidence map (each claim = one figure/table)
| # | Claim | Evidence (have?) | Artifact |
|---|---|---|---|
| C1 | Agent traffic is heavy-tailed; the popular head is also accurate & cheap | ✅ s_pop Gini 0.886; top-decile 72% acc vs 41% | Fig: rank-popularity + acc-by-decile |
| C2 | A multi-tier execution cache yields high hit @ large $-savings, but naive L3 propagates errors | ✅ 97% hit / 81% saved / 24% false-hit; LFU≫LRU | Fig: hit & cost vs capacity (LFU/LRU) |
| C3 | Admission gating by a pre-exec predictor carves a high-reliability head | ✅ top-30% @ 79% acc vs 52%; AUC 0.86 parametric | Fig: precision–coverage |
| C4 | The cheap **edge matches/beats origin at ~½ cost on the retrieval head**; gap grows with task complexity | ✅ Gradient: PopQA +8pp / FRAMES −2pp / GAIA −14pp; always-edge 53%@42% vs origin 54% (n=380) | Fig: edge−origin gap by benchmark |
| C5 | The routing axis is **retrieval-bound vs reasoning/tool-bound** (predictable), not generic difficulty | ✅ edge breaks only on GAIA multi-tool; task-type-route 55%@56%; oracle 63%@78% | Fig: 3-way frontier |
| C6 | Router value **scales with the reasoning-bound fraction** of the workload | ✅ retrieval-heavy→always-edge; reasoning-heavy→routing essential | Fig: frontier vs workload mix |
| C7 | A shared fetched-page cache is a CDN object cache for agents | ✅ 37% GAIA fetches redundant; FRAMES≈Wikipedia | Fig: fetch-cache hit + domain mix |
| C8 | (Extension) Cached answers go stale at tier-dependent rates → TTL | 🌱 volatile seed built; drift run pending | Fig: drift rate by volatility tier |

## System (§Design)
- **Edge** = small model (Haiku) + EKB cache tiers + router/predictor + web search.
- **Origin** = strong model (Sonnet/Opus) full agent.
- **Cache tiers:** L3 answer (popularity-keyed, admission-gated), L2 fetched-page (URL-keyed, the CDN-object
  analog), L1 plan/cost-prior (latency aid). **Request flow** = router → L3 → L2-assisted edge → edge cold
  → escalate origin; admit-by-value, evict-by-LFU.
- **Routing objective:** minimize $-cost s.t. accuracy ≥ origin − ε. Router escalates when predictor
  confidence is low (the tail it can't predict).

## Evaluation plan (status)
1. Workload characterization (C1) — ✅ done offline.
2. Cache replay: hit/cost/false-hit vs capacity, LFU/LRU (C2,C3) — ✅ done; extend with admission gate.
3. Counterfactual edge/origin frontier across difficulty (C4,C5) — 🔶 PopQA done; FRAMES/GAIA collecting.
4. Learned router vs always-edge/always-origin/oracle on mixed workload (C6) — 🔶 blocked on #3.
5. Opus stronger-origin ceiling — ⏸ `scripts/run_origin_opus.sh` staged.
6. Freshness/TTL drift study (C8) — 🌱 seed built; schedule t0→t1→t2.

## Honest limitations (state up front)
- No geo/TTFT: "edge" = capability/cost tier, claims are caching economics not proximity.
- Edge-dominance on PopQA partly reflects origin running effort=low/parametric — frame as "cheap path
  suffices for the head," not a raw model-capability verdict; the cross-difficulty *contrast* is the result.
- Cache-replay hit rates depend on the popularity-proportional stream *model*; report shape + safety, not
  exact %.
- L2 page-cache $ value is the addressable fetch-redundancy, not a direct bill reduction.

## Novelty vs related work
- vs GPTCache/semantic cache: multi-tier + admission + (freshness), on real agent traces at scale.
- vs RouteLLM/FrugalGPT: routes among **cache tiers & execution depths**, not just models; cache is the object.
- vs L1 plan papers (APC/WorkflowGen): **unmodified** agent; honest latency-not-accuracy result.
- New empirical message: **the cheap edge beats the strong origin on the high-frequency head** — invert the
  default of sending everything to the big model.
