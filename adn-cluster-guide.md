# ADN Experiments — HPC Cluster Execution Guide

**Read this first if you are a Claude session launched on the HPC cluster (single V100 16 GB) to run
the ADN pilot experiments.** The authoritative experiment design is **`adn-pilot-roadmap.md`** — read
it in full before doing anything. This guide tells you *how to operate on the cluster*: environment,
what is/ isn't in git, execution order, and pitfalls. The repo's top-level `CLAUDE.md` is about a
*different* task (GAIA benchmark collection) — it does **not** govern this work; this guide does.

---

## 0. What you are doing (one paragraph)
Three experiments on an edge-assisted agent-serving pipeline where **edge = Claude Haiku** and
**origin = Claude Sonnet/Opus** (both via the Anthropic API). **Exp 1:** can query features predict
whether the edge will succeed (routing), *beyond* the trivial benchmark-identity shortcut? **Exp 2:**
under oracle routing, how much $/token/latency is saved vs cloud-only, as a function of workload mix?
**Exp 3:** can a lightweight, self-hostable discriminator score an edge answer's trustworthiness and
reroute low-confidence cases to cloud? The **V100 is for the ML side only** (fine-tune a DeBERTa-class
encoder; serve a small local LLM via vLLM as a lightweight judge). There is **no local edge agent** —
edge/origin are Claude API calls.

## 1. Repository layout (what arrives via `git pull`)
- `adn-pilot-roadmap.md` — **the plan** (hypotheses, designs, deliverables, kill criteria).
- `adn-cluster-guide.md` — this file.
- `adn-findings.md`, `adn-experiments-detailed.md` — prior results + exactly how earlier experiments
  were run (read these for context and to reuse method).
- `adn-systems-plan.md`, `adn-paper-outline.md` — framing / paper skeleton.
- `scratch_adn/` — analysis code + **`all_records.jsonl`** (12,179 consolidated per-task records:
  query, prop, s_pop, cost, n_ws/n_wf, confidence, pred, correctness) + `*_matched_ids.json`
  (paired edge↔origin task ids: PopQA 134, FRAMES 188) + the reference scripts `consolidate.py`,
  `sim_l3.py`, `pred.py`, `compare_edge.py`, `compare_frames.py`.
- `runners/` — `run_claude_task.py`, `run_claude_popqa.py`, `run_claude_task_frames.py`, etc. The
  agent harness (calls `claude` CLI). **Reuse `exact_match`/`any_match` from here** (importable,
  model-agnostic) — do not reimplement gold matching.
- `tools/extract-query-embedding.py` — all-mpnet-base-v2, 768-d (for PopQA embeddings, see §3).
- `data/` — benchmark task files + gold answers (`data/popqa/*.jsonl`, `data/frames/*`,
  `data/gaia/*`, `data/volatile/volatile_seed.jsonl`).
- `embeddings/claude_native/<run>/<tid>/query_embedding.npy` — 768-d query embeddings (force-added
  to git despite `.npy` ignore). **Covers FRAMES + GAIA only; PopQA embeddings must be regenerated.**
- `scripts/` — collection shell scripts (`run_edge_counterfactual_haiku.sh`, `run_origin_opus.sh`, …).

## 2. What is NOT in git (you must provide it)
- **`traces/` (≈970 MB)** — raw per-task `normalized_trace.json` with full tool context (search
  queries+results, fetched page text), final answer, confidence, usage, per-step latency. **Gitignored
  by design.** **Needed for Exp 3** (the discriminator's `[query ⊕ tool-context ⊕ answer]` input) and
  to (re)build `all_records.jsonl` / collect more. **Transfer it out-of-band**, e.g. from the Mac:
  ```
  rsync -avz --progress <mac-user>@<mac-host>:/Users/tungi/ekb-claude-pilot/traces/ \
        <cluster>:/path/to/ekb-claude-pilot/traces/
  ```
  Exp 1 and Exp 2 can run from `all_records.jsonl` + `embeddings/` alone; only Exp 3 (and any new
  collection) strictly needs `traces/`.
- **`.env`** — secrets. Create on the cluster with at least `ANTHROPIC_API_KEY=...` (plus whatever
  the runners read; see the prior local-setup notes). Required for any Claude call: new Haiku/Sonnet
  collection, self-consistency sampling, and the strong LLM-judge relabel (S1).
- **`checkpoints/`, `wandb/`, `*.pt/pth/ckpt`, `*.npy` (other than the committed embeddings)** —
  generated locally on the cluster.

## 3. Environment setup (do once)
- Python deps: `scikit-learn`, `numpy`, `pandas`, `sentence-transformers` (for §S2 embeddings),
  `torch` (CUDA 11.x/12.x for V100, sm_70), `transformers`, `datasets`, `accelerate`, `peft`,
  `vllm`, `anthropic`. A V100 needs CUDA-compatible builds; confirm `torch.cuda.is_available()`.
- Claude CLI: the runners shell out to `claude`. If you only run the *ML/analysis* steps you do not
  need the CLI; you need it for *new collection*. Confirm `claude --version` or use the API directly.
- **vLLM model for the lightweight judge (B3) / Exp 1 zero-shot router:** `Qwen2.5-7B-Instruct-AWQ`
  (4-bit, ≈6 GB) fits the 16 GB V100 with KV-cache headroom. Fallback if memory is tight:
  `Qwen2.5-3B-Instruct` (FP16). Serve with `vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000`
  (OpenAI-compatible at `http://localhost:8000/v1`).
- **GPU is 16 GB — sequence the phases:** (A) serve the 7B for judge/router inference, dump scores to
  disk; (B) stop vLLM, free the GPU, then fine-tune the DeBERTa encoder (small footprint). Do **not**
  run a 7B server and encoder training simultaneously.

## 4. Execution order
Build shared infra first, then experiments. Each step writes artifacts the next consumes.

**Shared infra**
- **S1 — Strong-judge relabel.** Run Claude **Sonnet/Opus** (API) over every (question, gold,
  edge-answer) and (…, origin-answer) → binary `judged_correct`. This fixes strict-EM undercount
  (esp. FRAMES/GAIA). The strong judge is for *label quality* only; it is **not** the deployable
  discriminator. Report judge↔string-match agreement; hand-check ~20 disagreements.
- **S2 — PopQA embeddings.** `python tools/extract-query-embedding.py` over PopQA queries (it's
  missing; FRAMES+GAIA already shipped). Uniform 768-d features across all three benchmarks.
- **S3 — Latency cleaning.** Fit `latency ~ tokens + steps` per run; flag large positive residuals
  (high latency unexplained by work = contention/idle). Re-run only those few via the runners. Then
  decompose `total_latency = tool_latency + model_latency` (inter-step gaps). NOTE: the sanity check
  already showed latency tracks work (corr 0.73–0.95), so expect only a handful of re-runs.
- **S4 — Paired table.** Extend `all_records.jsonl` into a per-`query_id` table with base_task_id
  (GAIA paraphrase-safe splits), edge/origin `{judged_correct, cost_real$, latency_s, n_ws, n_wf,
  confidence, answer_text, tool_context_text}`, and embedding ref. This single table feeds Exp 1–3.

**Experiments** (see roadmap for full design + kill criteria)
- **Exp 1 — routing predictability.** Label = edge `judged_correct`. Run: (i) benchmark-id-only
  baseline, (ii) **within-each-benchmark** predictor (the scientific core), (iii) **leave-one-
  benchmark-out** transfer, (iv) feature ablation {benchmark-id, mpnet emb, interpretable,
  self-consistency over k=5 Haiku samples}. Report the `id-only → +query-features` AUC decomposition
  per benchmark; **never report pooled AUC alone**. Split strictly by base_task_id. **No token-
  logprobs exist** (Anthropic doesn't expose them) — uncertainty features are self-report +
  self-consistency only.
- **Exp 2 — savings ceiling.** Replay policies over S4: cloud-only, edge-only, oracle-route,
  predicted-route (Exp 1), and **cascade** (edge→discriminator→escalate). Metrics: real $ (price
  tables in `compare_edge.py`), token breakdown, latency mean+p95 (post-S3), accuracy. Stacked
  savings decomposition (L3 cache-hit / edge-served / origin-escalated); frontier **as a curve over
  the reasoning-bound fraction**; ε ∈ {0,2,5} pp. Honest: edge is cheaper but **slower** on PopQA
  (tool-round-trip-bound) — cost is the primary claim, latency supporting.
- **Exp 3 — discriminator.** Input `[query ⊕ tool-context ⊕ answer]` (needs `traces/`). Signals
  cheap→expensive: B0 self-confidence (baseline), B1 self-consistency, **B2 fine-tuned DeBERTa
  encoder (core deliverable, V100)**, B3 small local-LLM judge (vLLM). Metrics: risk-coverage AUROC,
  calibration (ECE), and the system payoff — sweep reroute threshold, does B2 **close the
  always-edge→oracle gap from Exp 2**? Plus cost-of-discriminator and PopQA+FRAMES→GAIA transfer.

## 5. Open decision to confirm with the human before scaling
The **reasoning-bound (hard) class is scarce**: only ~161 unique GAIA tasks (51 L1 / 84 L2 / 26 L3),
which bounds Exp 1's within-GAIA/LOBO power and Exp 3's transfer test. Recommended default: **collect
more** — run Haiku edge on GAIA L2+L3 (pair with origin). Alternatives: use the x3 paraphrases (split
by base_task_id) with small-n CIs, or carve a hard multi-hop FRAMES subset. PopQA can be scaled
cheaply (~1,686 untouched test tasks); FRAMES is saturated.

## 6. Pitfalls (learned the hard way)
- **Cost metric:** edge↔origin comparisons use **real per-model $/M-token** (Haiku ≠ Sonnet price);
  same-model/offline replay uses Sonnet-ratio $-weighted tokens. Never apply one model's price to the
  other (that bug produced a false "edge costlier" result).
- **Labels:** always use S1 `judged_correct`, not raw strict-EM, for training/eval.
- **Leakage:** GAIA `__paraN` paraphrases share a base task — split by base_task_id everywhere.
- **Pooled AUC lies:** it's inflated by easy cross-benchmark separation; within-benchmark + LOBO is
  the real test.
- **GPU memory:** sequence vLLM-serving and encoder-training; don't co-locate.
- **Don't reimplement** gold matching, consolidation, or the cost functions — import/reuse from
  `runners/` and `scratch_adn/`.

## 7. Deliverables
Per-benchmark+LOBO AUC table (Exp 1); policy×{cost,latency,accuracy} frontier + mix curves + savings
decomposition (Exp 2); risk-coverage/calibration plots + threshold-swept accuracy/cost vs
always-edge/oracle (Exp 3). End-to-end: one script regenerates the paired table → all three figures.
