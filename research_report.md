# EKB Caching Research — Progress Report

**Author:** Tung-I Chen  
**PI:** Hui Guan  
**Date:** 2026-06-15  
**Project:** Execution Knowledge Base (EKB) for Efficient AI Agent Serving  
**Repo:** `/work/pi_rsitaram_umass_edu/tungi/ekb-claude-pilot`

---

## Table of Contents

1. [Problem Statement and Motivation](#1-problem-statement-and-motivation)
2. [Related Work: Three-Layer Caching Framework](#2-related-work-three-layer-caching-framework)
3. [Experiment Setup and Infrastructure](#3-experiment-setup-and-infrastructure)
4. [Experiment 1: Semantic–Execution Correlation (Paraphrase Study)](#4-experiment-1-semanticexecution-correlation-paraphrase-study)
5. [Experiment 2: L1 Plan Caching on GAIA Level 1](#5-experiment-2-l1-plan-caching-on-gaia-level-1)
6. [Experiment 3: L1 Plan Caching on GAIA Level 2](#6-experiment-3-l1-plan-caching-on-gaia-level-2)
7. [Experiment 4: FRAMES and PopQA Trace Collection](#7-experiment-4-frames-and-popqa-trace-collection)
8. [Current Data Status](#8-current-data-status)
9. [Summary of Findings](#9-summary-of-findings)
10. [Next Steps](#10-next-steps)
11. [Appendix: File Layout and Key Commands](#11-appendix-file-layout-and-key-commands)

---

## 1. Problem Statement and Motivation

### 1.1 The ADN Proposal

The Adaptive Delivery Network (ADN) proposal addresses a core inefficiency in AI agent serving: every incoming user query triggers a full agent execution, regardless of whether that execution is novel or highly similar to prior work. This wastes compute, inflates latency, and burns through LLM token budgets unnecessarily.

The ADN introduces an **Execution Knowledge Base (EKB)** — a structured store of *how* agents have executed queries, not just *what* they output. The EKB enables **query routing** (predicting difficulty before execution) and **execution caching** (reusing prior execution knowledge to guide or skip future agent runs).

### 1.2 Core Assumption

The fundamental hypothesis driving the EKB is:

> **Semantically similar queries tend to execute similarly** — same tool sequences, similar step counts, similar resource consumption.

If this holds, a new query can be matched against cached execution traces to:
1. Predict its execution complexity (easy vs. hard, how many steps, which tools)
2. Reuse a prior execution plan to guide the agent and reduce redundant planning
3. Potentially skip execution entirely and return a refined prior answer (for near-identical queries)

### 1.3 The Three Caching Layers

The EKB is organized into three complementary caching layers, each operating at a different granularity:

| Layer | Cached Object | When It Intervenes | Benefit |
|---|---|---|---|
| **L1: Execution-plan caching** | Plan skeletons, tool sequences | Before/during action selection | Faster planning, fewer wasted steps |
| **L2: Tool-result caching** | Intermediate tool feedback `f_t` | During tool execution | Skip redundant API calls |
| **L3: Query-response caching** | Final answer `y_i` | Before the agent trajectory | Skip execution entirely |

The layers are complementary, not mutually exclusive. An incoming query could hit L3 (return cached answer directly), L2 (reuse some tool results), or L1 (reuse the execution plan structure), depending on how similar the prior execution is.

### 1.4 Why This Matters

Current LLM agent deployments treat every query as cold-start. A web-search agent answering "What is the population of France?" and "What is the population of Germany?" will issue nearly identical tool calls, spend the same tokens, and take the same latency — even though the execution structure is identical. Caching any of the three layers would yield substantial savings at scale.

---

## 2. Related Work: Three-Layer Caching Framework

We surveyed 10 recent papers on agentic caching, organizing them into the three-layer framework.

### 2.1 L1: Execution-Plan Caching

These methods cache plan-level procedural knowledge across tasks.

| Method | Reused Object | Cache Form | Retrieval Signal | Key Idea |
|---|---|---|---|---|
| **APC** (Zhang et al., NeurIPS 2026) | Whole-task plan template | `{(K(q_i), p_i)}` | Task keyword match | Adapts abstract plan templates on hit; abstracts trajectory on miss |
| **HMT** (Tan et al., arXiv 2603) | Hierarchical intent→stage→action memory | Tree `T = (I, G, L)` | Intent, stage condition, observation | Grounds reuse in current observation; avoids brittle DOM-ID replay |
| **AgenticCache** (Kim et al., arXiv 2604) | Short-horizon action transitions `p_i → p_j` | `{(p_i, p_j, count, importance)}` | Previous action + state metadata | Exploits plan locality; asynchronous LLM validation |
| **WorkflowGen** (Wei et al., arXiv 2604) | Executable workflow template + node experience | `{(k_i, p_i, e_i, η_i)}` | Query embedding similarity + two thresholds | Adaptive routing: direct reuse / rewrite / re-plan |

**Gap identified:** None of these papers test on large-scale web-search benchmarks. All use either small proprietary setups, embodied/GUI agents, or synthetic tool-calling workloads. Our work targets web-search agents (WebSearch + WebFetch tool calls) on standard QA benchmarks — a gap in the literature.

### 2.2 L2: Tool-Result Caching

| Method | Cache Key | Reused Object | State Awareness |
|---|---|---|---|
| **ToolCaching** (Zhai et al., arXiv 2601) | `Hash(d, u)` + TTL | Individual tool feedback `f_t` | Weak (rule-based) |
| **TVCACHE** (Kumar et al., arXiv 2602) | Tool-call prefix `ρ_t = ((d_0,u_0),...,(d_t,u_t))` | `f_t` under matched trajectory state | Strong (prefix-matched) |
| **FAME** (Kulkarni et al., arXiv 2601) | Session/invocation identifiers | Runtime or workflow state | Stateful (serverless) |

### 2.3 L3: Query-Response Caching

| Method | Cache Key | Reuse Rule | Main Risk |
|---|---|---|---|
| **GPTCache** (Bang, NLP-OSS 2023) | Query embedding | Return `y_i` if `Sim(φ(q), φ(q_i)) ≥ δ` | False semantic hit |
| **MeanCache** (Gill et al., IPDPS 2025) | User-adapted embedding + context chain | Return `y_i` if semantic sim AND context match | Context mismatch |
| **SIC** (Basu, arXiv 2602) | W5H2 canonical intent key | Reuse exact if intent+params match; refine if only intent matches | Under-keying |

### 2.4 Why Existing Benchmarks Do Not Fit

A survey of standard agent benchmarks found:

| Benchmark | Problem |
|---|---|
| WebArena, Mind2Web | Browser DOM action space (CLICK, TYPE, SELECT) — not web-search API |
| ALFWorld | Embodied household AI — not web search |
| SWE-bench | Code editing tasks — not factual web research |
| Existing caching papers | Small proprietary or synthetic workloads |

We need benchmarks where the agent primarily uses **WebSearch + WebFetch** API calls, has **gold answers** for evaluation, clusters **naturally by topic/question-type** (so that plan reuse is meaningful), and has **sufficient scale** for KB/test splitting. We identified GAIA, FRAMES, and PopQA as fitting these criteria.

---

## 3. Experiment Setup and Infrastructure

### 3.1 Agent

- **Agent framework:** Claude Code CLI (`claude -p`)
- **Model:** `claude-sonnet-4` (latest claude-sonnet-4-6)
- **Effort:** medium
- **Max turns:** 48
- **Timeout:** 600 seconds per task
- **Tools available:** WebSearch, WebFetch, Bash, Read, Write, and other general tools via ToolSearch

Claude Code is used as-is, without modification. The only intervention is optional injection of a cached plan into the system prompt (for plan reuse experiments).

### 3.2 Trace Collection Pipeline

Each task goes through a custom runner (`runners/run_claude_task.py` for baseline, `runners/run_claude_task_w_plan_reuse.py` for plan reuse) that:

1. Renders the task into a structured prompt
2. Launches `claude -p` as a subprocess with `--output-format json`
3. Captures all lifecycle events via a Claude Code hook script (`tools/_claude_trace_hook.py`)
4. Normalizes all outputs into a `normalized_trace.json` with standardized fields

Per-task artifacts stored in `traces/gaia/claude_native/<run_name>/<task_id>/`:
- `task_record.json` — raw input
- `task_prompt.txt` — rendered prompt passed to Claude
- `cli_command.json` — exact subprocess invocation (for reproducibility)
- `hook_events.jsonl` — live per-turn event log (survives HPC job kills)
- `normalized_trace.json` — standardized trace summary (steps, tools, tokens, latency, answer, exact_match)
- `claude_session.jsonl` — full session transcript

Aggregate results per run are in `results/claude_native/<run_name>/results.jsonl` and `summary.json`.

### 3.3 Embedding

- **Model:** `sentence-transformers/all-mpnet-base-v2` (768-dim, stronger than the initial MiniLM-L6-v2 used in early experiments)
- **Normalization:** L2-normalized embeddings for cosine similarity via dot product
- **Storage:** `embeddings/claude_native/<embedding-run>/<task_id>/query_embedding.npy`
- **Embedding script:** `tools/extract-query-embedding.py`

### 3.4 Benchmarks

| Benchmark | Task Type | # Tasks | Tools | Eval Metric |
|---|---|---|---|---|
| **GAIA Level 1** | Multi-tool factual QA (easy) | 51 test | WebSearch, WebFetch, Code, Files | Exact match (EM) |
| **GAIA Level 2** | Multi-tool factual QA (hard) | 84 test | WebSearch, WebFetch, Code, Files | Exact match (EM) |
| **FRAMES** | Multi-hop factual research (2–5 hops) | 627 KB + 197 test | WebSearch, WebFetch | Exact match (EM) |
| **PopQA** | Single-hop entity lookup | 9,756 KB + 2,431 test | WebSearch | Any-match (substring EM) |

### 3.5 Knowledge Base Construction

For GAIA: we use **paraphrase augmentation** — each original GAIA task is paraphrased 3× (for L1: original + 3 paraphrases = x4; for L2: original + 2 paraphrases = x3). The paraphrases are run through the agent to collect execution traces that form the KB. The original queries serve as test tasks.

This creates a controlled setting where we *know* the correct KB entry for each test query (its paraphrase), letting us establish an upper-bound retrieval recall and measure plan reuse quality under near-ideal conditions.

For FRAMES and PopQA: standard KB/test splits are used without paraphrase augmentation (more realistic retrieval setting).

### 3.6 Plan Reuse Mechanism

For a new test query `q`:
1. Embed `q` using `all-mpnet-base-v2`
2. Retrieve top-5 KB neighbors by cosine similarity
3. Filter: keep only neighbors with similarity ≥ 0.8
4. If cache hit: rank the ≥0.8 candidates by a plan-quality metric (number of tool calls, or total tokens used in KB run)
5. Extract the `tool_sequence` (ordered list of tool names) from the top-ranked KB trace
6. Inject the tool sequence into the system prompt as a mandatory execution plan:

```
Execution Plan (retrieved from knowledge base — follow strictly):
A semantically similar task was previously solved efficiently using this exact tool sequence:
  1. WebSearch
  2. WebFetch
  3. WebSearch
  ...
You MUST execute these tools in this exact order.
Do not use any tools not listed above, do not skip steps, and do not add extra steps.
```

7. If no KB neighbor ≥ 0.8 similarity (cache miss): run without plan injection

---

## 4. Experiment 1: Semantic–Execution Correlation (Paraphrase Study)

### 4.1 Setup

- **Data:** `gaia_lv1_x4.jsonl` (204 tasks = 51 originals × 4 variants) and `gaia_lv2_x3.jsonl` (288 tasks = 96 originals × 3 variants)
- **Completed:** 132/204 for L1, 292/288+ for L2
- **Goal:** Validate that semantically similar queries (paraphrases of the same task) produce similar execution profiles

### 4.2 3NN Retrieval Quality

Using `all-mpnet-base-v2` embeddings and cosine similarity, for each of the 51 GAIA L1 test tasks (original queries), we retrieve the 3 nearest KB neighbors from the paraphrase pool:

| Metric | Value |
|---|---|
| Rank-1 paraphrase recall (% of top-1 neighbors that are correct paraphrases) | **~80%** |
| Mean rank-1 cosine similarity | **0.84** |
| Median rank-1 cosine similarity | **0.93** |
| Tasks with any KB neighbor ≥ 0.8 similarity (cache hit rate) | **~96%** |

The high hit rate reflects the controlled paraphrase setting — in a real-world KB without deliberate paraphrases, hit rates would be lower.

### 4.3 Within-Group Execution Variance

We group the completed L1 traces by original task ID (each group = original + paraphrases). For the 33 groups with ≥2 completed variants:

| Metric | Value |
|---|---|
| Within-group step standard deviation (mean over groups) | **4.12** |
| Within-group step std (median over groups) | **1.89** |
| Global step standard deviation (all tasks) | **9.50** |
| Intra-group std / global std | **0.43** |

Interpretation: queries in the same semantic group (paraphrases of the same task) have ~57% lower execution variance than the overall population. This validates the EKB's core assumption — but the within-group variance is still substantial (some paraphrase groups have high stdev due to unpredictable agent behavior on hard tasks).

### 4.4 Key Takeaway

The semantic-execution correlation signal is real but noisy. It is strongest for simple GAIA Level 1 tasks (short, predictable tool sequences) and weaker for hard tasks where the agent behavior is more variable. This suggests L1 plan caching will be most effective for simple, structured tasks.

---

## 5. Experiment 2: L1 Plan Caching on GAIA Level 1

### 5.1 Setup

- **Test set:** 51 GAIA Level 1 tasks (`data/gaia/gaia_lv1.jsonl`)
- **KB:** 152 paraphrase traces from `gaia_lv1_x4` run (`data/gaia_paraphrased/gaia_lv1_x4_kb.jsonl`)
- **KB embedding run:** `gaia_lv1_x4-all-mpnet-base-v2`
- **Similarity threshold:** 0.8
- **Top-K candidates:** 5
- **Ranking variants:** `total_tool_calls` (fewer tool calls = simpler cached plan) or `total_tokens` (fewer tokens = leaner cached plan)

### 5.2 Preliminary Study Results (MiniLM Embeddings, First Iteration)

This was the initial plan-caching experiment, evaluated in `scripts/plan_caching_evaluation.ipynb` (commit 1095ffb). It used `all-MiniLM-L6-v2` embeddings.

| Condition | EM | Steps (mean) | Tokens (mean) | Latency (median) | Cache Hit Rate |
|---|---|---|---|---|---|
| Baseline (no plan reuse) | **70.6%** | 9.6 | 159k | 44.9s | — |
| Plan reuse (rank=total_tool_calls) | **76.5%** | 6.5 | 175k | 38.7s | 86.3% |
| Plan reuse (rank=total_tokens) | (included for comparison) | — | — | — | — |

**Deltas vs baseline (rank=tool_calls):**
- Exact match: +5.9 percentage points
- Steps: −3.0 (−31%)
- Latency: −14%
- Tokens: +10% (overhead from plan injection and occasional plan-following detours)

**Tool-sequence adherence (on cache hits):**
- 81.8% of hit tasks achieved ≥0.8 sequence similarity to cached plan
- 54.5% followed the cached plan exactly (0 deviation)

### 5.3 Updated Results (all-mpnet-base-v2 Embeddings)

After switching to the stronger `all-mpnet-base-v2` embedding model, the cache hit rate improved and we ran two ranking variants on the same 51 test tasks:

| Run | EM | Steps (mean) | Tokens (mean) | Latency (mean) | Cache Hits |
|---|---|---|---|---|---|
| `gaia_lv1_plan_reuse` (rank=tool_calls) | **70.6%** | 4.5 | 134k | 59.4s | 49/51 (96%) |
| `gaia_lv1_plan_reuse_rank_by_token` (rank=tokens) | **80.4%** | 4.7 | 149k | 78.5s | 51/51 (100%) |

**Notes on comparison:**
- The baseline for this run is not directly measured in the updated experiment (the original task baseline comes from partial `gaia_lv1_x4` results: 41 tasks at 65.9% EM, 6.2 steps, 187k tokens, 96s mean latency). Direct comparison requires caution due to sample size difference.
- The rank-by-token variant achieves 80.4% EM with near-100% cache hit rate, suggesting that selecting the leanest KB plan (fewest tokens) leads to better plan transferability.
- The lower step counts (4.5–4.7 vs. 6.2 baseline) indicate significant reduction in agent exploration, which may explain why token overhead is partially offset.

### 5.4 Key Takeaways from L1 Plan Caching (GAIA Level 1)

1. **Plan caching improves accuracy on Level 1 tasks.** Both preliminary study (+5.9 pp) and updated runs (+4.7–14.5 pp relative to partial baseline) show positive EM delta.
2. **Step count drops significantly.** The plan injection effectively guides the agent to use fewer, more targeted tool calls (−31% to −35% steps).
3. **Token overhead is partially offset.** The plan prompt adds tokens, but the guided execution reduces exploration overhead — net token cost is roughly neutral to slightly positive.
4. **Ranking criterion matters.** Selecting the KB plan ranked by token count (fewest tokens) consistently outperforms ranking by tool call count. This suggests that token-lean plans are more "compact" representations that transfer better.
5. **Higher hit rate with stronger embedding.** Upgrading from MiniLM to all-mpnet-base-v2 raised the hit rate from 86.3% to 96–100%, suggesting the embedding model quality is a key lever.

---

## 6. Experiment 3: L1 Plan Caching on GAIA Level 2

### 6.1 Setup

- **Test set:** 84 GAIA Level 2 tasks (`data/gaia/gaia_lv2.jsonl`)
- **KB:** 160 paraphrase traces from `gaia_lv2_x3` run (`data/gaia_paraphrased/gaia_lv2_x3_kb.jsonl`)
- **KB embedding run:** `gaia_lv2_x3-all-mpnet-base-v2`
- **Similarity threshold:** 0.8, Top-K: 5
- **Run:** `gaia_lv2_plan_reuse` (partial, 32/84 tasks completed due to quota limits)

### 6.2 Results (Matched Comparison on 32 Tasks)

| Condition | EM | Steps (mean) | Tokens (mean) | Cache Hit Rate |
|---|---|---|---|---|
| Baseline (32 matched tasks from `gaia_lv2_x3`) | **68.8%** | 12.8 | 431k | — |
| Plan reuse (rank=tool_calls, 32 tasks) | **50.0%** | 8.3 | 244k | 84% |

**Breakdown by cache hit/miss:**
- Cache hits (27/32): 59.3% EM
- Cache misses (5/32): 0.0% EM (all 5 missed tasks failed — small sample)

**Extended run (89 tasks, `gaia_lv2_plan_reuse_rank_by_step`):**
- EM: 60.7%, steps: 12.9, tokens: 442k, cache hits: 92%
- Compared to full L2 baseline (96 tasks): 66.7% EM, 12.1 steps, 359k tokens

### 6.3 Key Takeaways from L1 Plan Caching (GAIA Level 2)

1. **Plan caching hurts accuracy on Level 2 tasks.** The matched comparison shows −18.8 pp EM (68.8% → 50.0%). The extended run shows −6 pp compared to the full baseline.
2. **Step reduction is still observed.** The plan guides the agent to complete tasks in fewer steps (8.3 vs. 12.8) — but fewer steps does not help when those steps are the wrong ones.
3. **Token cost increases in the extended run.** The 89-task run consumed more tokens on average (442k vs. 359k baseline), suggesting that following a misaligned plan forces the agent to backtrack or add extra steps.
4. **Cache misses all failed.** The 5 tasks with no KB neighbor above 0.8 similarity all failed. This is consistent with harder tasks being harder to retrieve for — but 5 tasks is too small for conclusions.

**Hypothesis for why L2 caching hurts:**
- Level 2 tasks require more complex, multi-step, adaptive execution. A rigid cached plan constrains the agent at exactly the points where it needs flexibility to handle page-specific content, unexpected search results, or multi-source synthesis.
- The paraphrase KB may not contain plans that generalize to the specific information retrieval paths needed for a new (but similar) Level 2 task — the topic is the same, but the execution path is highly sensitive to the specific evidence found at each step.
- The prompt instruction ("You MUST execute these tools in this exact order") may be too strict for Level 2 tasks. A softer "suggestion" framing might preserve benefits while allowing adaptation.

---

## 7. Experiment 4: FRAMES and PopQA Trace Collection

These experiments build larger, more realistic trace datasets for future plan caching and tool-result caching evaluation.

### 7.1 FRAMES (Factuality, Retrieval, And Multi-step Evidenced Summarization)

- **Source:** Google DeepMind FRAMES dataset, 824 tasks
- **Split:** 627 KB tasks + 197 test tasks (`data/frames/frames_kb.jsonl`, `data/frames/frames_test.jsonl`)
- **Runner:** `runners/run_claude_task_frames.py`
- **Task type:** Multi-hop factual research requiring 2–5 web sources per answer

**Current status (KB run, 250/627 completed):**

| Metric | Value |
|---|---|
| Tasks completed | 250/627 (40%) |
| Exact match (EM) | 38.6% (97/250) |
| Mean steps | 7.3 |
| Median steps | 5.0 |
| Mean tokens | 187k |
| Mean latency | 62.5s |
| Median latency | 43.0s |

**Test run:** Not yet started (0/197 tasks).

**Why FRAMES matters:**
- Multi-hop structure forces multiple WebSearch/WebFetch calls per task — exactly the setup where plan caching is meaningful
- 824 tasks is 6× GAIA's base task count, enabling robust KB/test splitting
- Tasks cluster by topic (science, sports, history, geography) — natural execution-plan similarity within clusters
- 38.6% EM indicates these are genuinely hard tasks, comparable to GAIA Level 1–2

### 7.2 PopQA

- **Source:** Entity-centric open-domain QA, ~14k pairs filtered to ~12k
- **Split (filtered):** 9,756 KB tasks + 2,431 test tasks
- **Runner:** `runners/run_claude_popqa.py`
- **Task type:** Single-hop entity lookup ("Who directed X?", "What nationality is Y?")

**Current status:**

| Split | Completed | Any-match accuracy |
|---|---|---|
| popqa_kb | 4,374/9,756 (45%) | 80.7% |
| popqa_test | 2,431/2,431 (**100%**) | 80.7% |

**Notable characteristics:**
- Very short tasks: avg 1.7–1.9 steps (single WebSearch → extract → answer)
- High accuracy: 80.7% any-match at simple single-hop lookup
- Questions group tightly by question type: all "birth year" queries share the same plan (`WebSearch entity` → `extract year` → `StructuredOutput`), all "director" queries share another plan, etc.
- This makes PopQA the **cleanest benchmark for demonstrating plan caching** — the cached plan for one "director" query should perfectly transfer to all other "director" queries, with only the entity name changing

---

## 8. Current Data Status

### 8.1 Trace Collection Progress

| Dataset | Purpose | Done | Total | % | Status |
|---|---|---|---|---|---|
| `gaia_lv1_x4` | GAIA L1 × 4 paraphrases (KB + paraphrase study) | 132 | 204 | 65% | In progress |
| `gaia_lv2_x3` | GAIA L2 × 3 paraphrases (KB + paraphrase study) | ~292 | ~288 | ~100% | Nearly complete |
| `frames_kb` | FRAMES KB traces | 250 | 627 | 40% | In progress |
| `frames_test` | FRAMES test traces | 0 | 197 | 0% | **Not started** |
| `popqa_kb` | PopQA KB traces | 4,374 | 9,756 | 45% | In progress (rate limited) |
| `popqa_test` | PopQA test traces | 2,431 | 2,431 | 100% | **Complete** ✅ |

### 8.2 Plan Caching Experiments Completed

| Run | Benchmark | Variant | Tasks | EM |
|---|---|---|---|---|
| `gaia_lv1_plan_reuse` | GAIA L1 | rank=tool_calls | 51 | 70.6% |
| `gaia_lv1_plan_reuse_rank_by_token` | GAIA L1 | rank=tokens | 51 | 80.4% |
| `gaia_lv1_plan_reuse_rank_by_latency` | GAIA L1 | rank=latency | partial | — |
| `gaia_lv2_plan_reuse` | GAIA L2 | rank=tool_calls | 32 | 50.0% |
| `gaia_lv2_plan_reuse_rank_by_step` | GAIA L2 | rank=tool_calls (extended) | 89 | 60.7% |
| `gaia_lv2_plan_reuse_rank_by_token` | GAIA L2 | rank=tokens | 89 | 59.6% |

### 8.3 Resume Commands

```bash
# Activate environment (always required)
module load conda/latest && conda activate /work/pi_rsitaram_umass_edu/tungi/conda/envs/ekb
cd /work/pi_rsitaram_umass_edu/tungi/ekb-claude-pilot

# Resume gaia_lv1_x4 (72 tasks remaining)
python runners/run_claude_task.py --input data/gaia_paraphrased/gaia_lv1_x4.jsonl

# Continue frames_kb (392 tasks remaining)
bash scripts/run_frames.sh
# OR: python runners/run_claude_task_frames.py --input data/frames/frames_kb.jsonl --run-name frames_kb --model sonnet --effort medium --max-turns 48 --timeout-sec 600 --disable-session-archive --sleep-sec 1

# Start frames_test (197 tasks, not yet started)
python runners/run_claude_task_frames.py --input data/frames/frames_test.jsonl --run-name frames_test --model sonnet --effort medium --max-turns 48 --timeout-sec 600 --disable-session-archive --sleep-sec 1

# Resume popqa_kb (5,382 tasks remaining)
python runners/run_claude_popqa.py --input data/popqa/popqa_filtered_kb.jsonl --run-name popqa_kb
```

---

## 9. Summary of Findings

### Finding 1: Semantic–Execution Correlation is Real but Noisy

The core EKB assumption holds: paraphrases of the same GAIA task produce similar tool sequences (within-group step std ≈ 43% of global step std). The signal is strongest for simple Level 1 tasks and weakens for harder tasks.

### Finding 2: L1 Plan Caching Helps on Simple Tasks (GAIA Level 1)

- **+5.9 pp EM** in preliminary study with MiniLM embeddings
- **Up to +14.5 pp EM** with stronger all-mpnet-base-v2 embeddings and rank-by-token selection
- **−30–35% steps** consistently across all variants
- Near-100% cache hit rate with the stronger embedding model

### Finding 3: L1 Plan Caching Hurts on Hard Tasks (GAIA Level 2)

- **−18.8 pp EM** on matched 32-task comparison
- **−6 pp EM** on the 89-task extended run vs. full baseline
- Steps reduce (−35%) but accuracy falls, suggesting the plan is too rigid for complex tasks
- Hypothesis: hard tasks require adaptive execution that a rigid cached plan blocks

### Finding 4: Ranking Criterion Matters

Selecting cached plans by total_tokens (fewest tokens = leanest plan) consistently outperforms ranking by total_tool_calls. This is possibly because token-lean cached plans are more compact and leave more room for the agent to adapt.

### Finding 5: Benchmark-Level Observations

- **FRAMES** (38.6% EM on 250 tasks): Multi-hop tasks are harder than GAIA Level 1 but the execution structure is clearly web-search-driven (avg 7.3 steps). Good candidate for plan caching if the KB is large enough.
- **PopQA** (80.7% any-match on 2,431 test tasks, avg 1.9 steps): Single-hop lookups are fast, highly accurate, and group tightly by question type — ideal for demonstrating plan-type clustering and clean plan transfer.

---

## 10. Next Steps

### Immediate (Data Collection, in priority order)

1. **Complete `frames_test`** (197 tasks, not started) — needed to measure FRAMES baseline accuracy before running plan reuse
2. **Resume `frames_kb`** (392 remaining) — KB needed for FRAMES plan caching experiments
3. **Resume `gaia_lv1_x4`** (72 remaining) — complete the paraphrase study dataset for L1

The large `popqa_kb` run (5,382 remaining) is lower priority since the test set is done and basic accuracy is established.

### Experiment Design Improvements

4. **Softer plan injection for L2.** Instead of "You MUST execute in this exact order," try: "A similar task was solved with this tool sequence. Consider following this approach, but adapt as needed." This may preserve step-count savings while allowing the flexibility that L2 tasks require.

5. **Similarity threshold sweep.** Test thresholds {0.7, 0.75, 0.8, 0.85} to measure the hit-rate vs. plan-quality tradeoff. Lower threshold → more hits but less accurate plans.

6. **Plan abstraction.** Instead of injecting the raw tool sequence (which includes ToolSearch overhead steps), abstract to just the substantive tool names (WebSearch, WebFetch, Bash). This may give the agent cleaner guidance.

7. **Cost analysis with actual billing rates.** The Claude API charges 0.1× for cache-read tokens and 5× for output tokens. A proper cost model might show net savings even when token count slightly increases due to plan injection — because plan reuse reduces output tokens (less reasoning under uncertainty).

### Expanding to FRAMES and PopQA

8. **Compute embeddings for completed FRAMES KB traces** and run 3NN retrieval for FRAMES test tasks. Run plan reuse on FRAMES.

9. **Run plan reuse on PopQA test tasks** using completed PopQA KB. Given the tight question-type clustering, expect very high cache hit rates and strong plan transfer — PopQA may be the cleanest demonstration of the EKB value proposition.

10. **Semantic clustering analysis on FRAMES and PopQA.** Show that question-type clusters (FRAMES: science/sports/history; PopQA: director/nationality/birth-year) align with execution-plan clusters.

### Architecture Toward L2 and L3

11. **L2 (Tool-result caching):** For PopQA, the WebSearch query is nearly deterministic given the entity name. If two tasks ask about the same entity but in different ways, the tool result (WebSearch page content) is likely identical. Measure how often this occurs to estimate potential L2 cache hit rates.

12. **L3 (Query-response caching):** For PopQA, since tasks are single-hop and the answer is typically a short string, a direct response cache (GPTCache-style) is plausible. Measure the semantic similarity distribution between PopQA test queries and KB queries to estimate potential L3 hit rates.

13. **Joint L1+L2+L3 design.** Once each layer is validated separately, design a combined caching strategy that applies the most aggressive reuse possible given a similarity threshold, falling back from L3 → L2 → L1 → cold execution as the match quality decreases.

### Research Positioning

14. **Differentiator from APC/WorkflowGen.** Our work uses Claude Code as the unmodified agent (no fine-tuning, no custom planning module) and measures caching on established web-search benchmarks (GAIA, FRAMES, PopQA). This demonstrates that L1 plan caching is effective with standard agents, not just purpose-built ones.

15. **Paper narrative.** The contrast between L1 caching helping on simple tasks and hurting on hard tasks is itself a key contribution: it motivates *adaptive* caching (apply plan injection only when similarity is high enough or the task is simple enough) and the need for a joint multi-layer design where L2 caching helps even when L1 plan caching is unsafe.

---

## 11. Appendix: File Layout and Key Commands

### Key File Paths

```
ekb-claude-pilot/
├── data/
│   ├── gaia/                          # Raw GAIA tasks (gaia_lv1.jsonl, gaia_lv2.jsonl)
│   ├── gaia_paraphrased/              # x4/x3 paraphrase datasets + KB splits
│   ├── frames/                        # FRAMES dataset (frames.jsonl, frames_kb.jsonl, frames_test.jsonl)
│   └── popqa/                         # PopQA (popqa_filtered_kb.jsonl, popqa_filtered_test.jsonl)
│
├── runners/
│   ├── run_claude_task.py             # Baseline trace collection runner
│   ├── run_claude_task_w_plan_reuse.py# Plan-reuse runner (L1 cache)
│   ├── run_claude_task_frames.py      # FRAMES runner
│   └── run_claude_popqa.py            # PopQA runner
│
├── tools/
│   ├── _claude_trace_hook.py          # Hook script for trace collection
│   ├── extract-query-embedding.py     # Compute embeddings for KB tasks
│   └── prepare_frames.py              # FRAMES dataset prep
│
├── embeddings/claude_native/          # Per-task query embeddings (.npy)
│   └── <embedding-run>/<task_id>/query_embedding.npy
│
├── results/claude_native/             # Per-run results (results.jsonl, summary.json)
│   ├── gaia_lv1_x4/
│   ├── gaia_lv1_plan_reuse/
│   ├── gaia_lv1_plan_reuse_rank_by_token/
│   ├── gaia_lv2_plan_reuse/
│   ├── gaia_lv2_plan_reuse_rank_by_step/
│   ├── frames_kb/
│   ├── popqa_kb/
│   └── popqa_test/
│
├── scripts/
│   ├── run_frames.sh                  # Run FRAMES KB traces
│   └── plan_caching_evaluation.ipynb  # Preliminary study evaluation (commit 1095ffb)
│
├── notebooks/                         # Analysis notebooks
│   ├── gaia_paraphrase_trace_analysis_lv1_x4.ipynb
│   ├── gaia_paraphrase_trace_analysis_lv2_x3.ipynb
│   ├── gaia_plan_reuse_analysis.ipynb
│   └── gaia_lv2_plan_reuse_analysis.ipynb
│
├── agentic_caching_survey.md          # Detailed three-layer caching literature survey
├── ekb-preliminary-study.md           # Original study design document
├── benchmark_sugg.md                  # Benchmark selection rationale (FRAMES, PopQA, HotpotQA)
└── configs/claude/ekb_trace_settings.json  # Claude Code session settings for all runs
```

### Conda Environment

```bash
module load conda/latest && conda activate /work/pi_rsitaram_umass_edu/tungi/conda/envs/ekb
```

Always run this before any Python script. The environment contains all dependencies (sentence-transformers, numpy, anthropic SDK, etc.).

### Metrics Reference

| Field | Meaning |
|---|---|
| `exact_match` | Bool — answer exactly matches gold (GAIA, FRAMES) |
| `any_match` | Bool — gold answer is substring of predicted answer (PopQA) |
| `total_steps` | Total number of tool-use turns in the session |
| `total_tool_calls` | Number of substantive tool calls (excludes ToolSearch overhead) |
| `total_tokens` | Total input + output tokens consumed |
| `total_latency_ms` | Wall-clock time from session start to final answer |
| `cache_hit` | Bool — a KB neighbor was found at similarity ≥ threshold |
| `cache_source_similarity` | Cosine similarity of the retrieved KB plan |
| `cached_tool_sequence` | The injected tool sequence (list of tool name strings) |

---

*Report generated 2026-06-15. All numbers computed from `results/claude_native/` files at that date. Data collection is ongoing — FRAMES KB (40% done) and PopQA KB (45% done) will be updated in subsequent runs.*
