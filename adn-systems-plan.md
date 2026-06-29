# ADN: A Content-Delivery-Network Discipline for Agent Execution

**Status:** direction proposal (2026-06-27). Supersedes the "plan-template reuse" framing.
**Target:** systems venue (SoCC / NSDI class).
**Constraints locked:** emulated edge/origin (small model = edge, full model = origin); large counterfactual collection budget OK.

---

## 1. Thesis

> Agent workloads, like web traffic, have a **cacheable head and a novel tail**. We bring the
> CDN discipline — a **multi-tier execution cache** (answer / tool-result / plan), **volatility-aware
> freshness**, **value-based admission/eviction**, and **tier routing** — to agent serving. The
> **Execution Knowledge Base (EKB)** is the cache substrate; the **edge/PoP** is where it lives.
> The novel tail escalates to the cloud **origin** (full agent). This keeps the system honest:
> ADN never claims to answer everything cheaply — it claims to serve the head cheaply and route
> the tail correctly.

The pilot's findings are reframed as **characterization**, not failures:
- Head/tail structure (paraphrase variance 0.2–0.3× population; FRAMES decoupled) → *the workload is a CDN candidate, but only the head is cacheable.*
- L1 reuse = latency-not-accuracy, concentrated on the expensive tail → *plans are a tier, not a silver bullet.*
- Value tracks cost not similarity → *admit/route by predicted cost, not by similarity score.*
- FRAMES collapse → *the motivation for a router: you must detect cacheability before reusing.*

## 2. Honest scoping of "edge" (read this first)

We do **not** have geo-distributed PoPs or TTFT instrumentation, so we make **no empirical claim
about network proximity**. Every measured number is **caching economics**: hit rate, redundant-
execution avoided, freshness/staleness, and the cost/quality/latency frontier. "Edge" is the
**deployment narrative** (the cache+router lives in a CDN PoP — infra the lab already owns) and is
realized in experiments as a **model/capability tier**:

| Role | Realization | Why it's a faithful proxy |
|---|---|---|
| **Edge executor** | `claude --model haiku --effort low` (+ optional L1/L2 assist) | cheap, fast, capacity-limited — the cost profile of an edge node |
| **Origin executor** | `claude --model sonnet` (or `opus` for a stronger origin) | expensive, capable — the cost profile of central cloud |
| **Edge cache** | EKB tiers L3/L2/L1 served from the local store | the CDN cache at the PoP |
| **Router/predictor** | lightweight head over query embedding + KB-neighborhood features | the request-routing logic at the PoP |

This framing is fully supported by the data we can collect and avoids the proximity/TTFT
overclaim in the external deep-research report.

## 3. System architecture (ADN)

Request flow at the edge:

```
query q
  │
  ▼
[1] Router/predictor  ── features: emb(q), top-k KB sim, neighborhood density,
  │                                 cost-envelope estimate, volatility class
  ▼
[2] L3 answer cache?  ──hit & fresh & confident──▶ serve cached answer (opt. cheap revalidation)
  │ miss / stale
  ▼
[3] L2 tool-result cache?  ──hit──▶ edge small model reuses tool results, finishes locally
  │ miss
  ▼
[4] predicted cheap-path success high?  ──yes──▶ edge small model (± L1 plan), verify + fallback
  │ no / verify-fail
  ▼
[5] ORIGIN full agent (± soft L1 on expensive tail)
  │
  ▼
[6] Admission: write execution back to EKB if predicted (reuse × savings) clears a bar;
    Eviction: under capacity, drop by LFU/TinyLFU on execution objects.
```

Novel pieces vs prior work:
- **Multi-tier execution cache** (most semantic-cache work is single-tier, answer-only).
- **Volatility-aware freshness** — the agent analog of HTTP TTL / `If-Modified-Since`. *No semantic-cache paper handles staleness.* This is the highest-novelty component.
- **Value-based admission/eviction** on execution objects under bounded edge capacity.
- **Tier routing** (vs RouteLLM/FrugalGPT which route between *models*, not *cache tiers*).
- **Unmodified agent** throughout (Claude Code as-is) — clean, reproducible measurement.

## 4. Counterfactual data collection (the enabling asset)

For an evaluation set per benchmark, run each query under **every serving path** so we can build
an offline oracle, train the router, and replay any policy without re-running:

| Path id | Command | Purpose |
|---|---|---|
| `origin` | `--model sonnet` (have most already) | strong reference quality + cost |
| `origin_opus` | `--model opus` (optional) | stronger origin ceiling |
| `edge_cold` | `--model haiku --effort low` | cheap-path success labels |
| `edge_L1` | `run_claude_task_w_plan_reuse.py --model haiku` | does a plan prior rescue the cheap path? |
| `origin_softL1` | plan-reuse runner, soft framing, `--model sonnet` | expensive-tail latency win |
| L3 / L2 | **replay** test↔KB, no new runs | answer-cache & tool-result-cache hit/savings/false-hit |

Freshness sub-collection: re-run a **volatility-stratified KB sample now (t1)** and diff answers
vs original (t0) → measures answer drift → calibrates per-class TTL.

Everything is the existing runner with different flags — **no new runner needed** (the cheap-path
collection is the single biggest new spend; budget approved).

## 5. Experiment phases

### Phase 0 — Offline characterization (no new runs; do first, ~free)
- Tier hit-rate vs KB-size curves (L3/L2/L1) per benchmark.
- Template/intent skew (PopQA: plan fan-out per question-type); FRAMES tail.
- Multi-tier replay: per-tier hit rate, token/latency savings, **false-hit rate**.
- **Deliverable:** the motivation figure ("agent execution is cacheable, head-heavy, tier-dependent").

### Phase 1 — Freshness / staleness (novelty core; bounded re-runs)
- Volatility taxonomy (static fact / slow-changing / time-sensitive).
- t0→t1 answer-drift study on stratified sample.
- TTL + cheap-revalidation policy; measure stale-serve rate vs cost saved.
- **Deliverable:** "the freshness model" — the thing no prior semantic cache has.

### Phase 2 — Predictor + admission (the brain; needs counterfactual matrix)
- Train: cacheability (per tier), cost envelope (steps/tokens/latency), cheap-path success prob.
- Admission policy from predicted (reuse × savings).
- **Deliverable:** the router, evaluated as oracle vs learned vs threshold.

### Phase 3 — End-to-end ADN cascade (the systems result)
- Full L3→L2→L1→cold cascade with freshness + admission/eviction.
- **Baselines:** always-origin, always-edge, threshold-similarity cache, RouteLLM-style model router.
- **Metrics:** quality (LLM-judge + EM/any-match), $/token cost, end-to-end latency + p95,
  per-tier hit rate, escalation rate, **stale-serve rate**, hit-rate-vs-capacity.
- **Headline criterion:** minimize cost s.t. quality ≥ origin − ε (recognizable to the routing
  community), plus a freshness SLA the baselines violate.
- **Workload sensitivity:** PopQA (head-heavy, ADN wins big) vs FRAMES (tail-heavy, ADN degrades
  gracefully to always-origin) vs GAIA (mixed) — shows the policy is safe, not just lucky.

## 6. Differentiators (defensible against likely reviewer pushback)
- vs GPTCache / semantic cache: multi-tier + **freshness** + admission, on real agent traces at scale.
- vs RouteLLM / FrugalGPT / Hybrid-LLM: routes among **cache tiers and execution depths**, not just models; the cache is the contribution.
- vs L1 plan papers (APC / WorkflowGen): **unmodified** agent, honest latency-not-accuracy result, plans are one tier.
- Caveats stated up front: no geo/TTFT claims; entity-level Zipf argued from external logs, not PopQA (PopQA gives *template-level* reuse).

## 7. Open decisions
- Origin = Sonnet only, or include Opus as a stronger ceiling?
- Eviction/capacity study: include (stronger systems story) or defer (scope risk)?
- LLM-judge rescore: needed to fix strict-EM undercount before quality claims — when?
