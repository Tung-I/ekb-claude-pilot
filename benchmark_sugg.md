# Benchmark Suggestions for EKB Web-Search Experiments

**Date**: 2026-06-13  
**Context**: We are looking for benchmarks beyond GAIA (Level 1: ~51 tasks, Level 2: ~84 tasks, total ~135) for future EKB caching experiments. The constraint is that we focus on **web-search applications** (agent primarily uses WebSearch/WebFetch tools), and we need sufficient scale for meaningful caching evaluation.

---

## 1. Why GAIA Alone Is Insufficient

The paraphrase augmentation study (gaia_lv1_x4, gaia_lv2_x3) served as a preliminary study to establish the semantic–execution correlation signal. However, in a realistic caching scenario the knowledge base is unlikely to contain near-duplicate (paraphrase) queries for the current task. Without paraphrases, GAIA reduces to ~135 base tasks — too small for reliable cache hit rate estimation, KB/test splitting, or statistical significance in retrieval experiments.

---

## 2. Why Most Caching-Paper Benchmarks Are Not Suitable

A survey of the related caching papers (APC, HMT, AgenticCache, WorkflowGen, ToolCaching, TVCACHE, SIC) revealed that none used a large-scale, web-search-native benchmark:

| Paper | Benchmark(s) Used |
|---|---|
| AgenticCache | 4 unnamed embodied (non-web) benchmarks |
| TVCACHE | Terminal tasks, SQL generation, video understanding |
| ToolCaching | Synthetic tool-calling workloads |
| HMT | Unspecified web-task benchmarks |
| APC / WorkflowGen | Unspecified "real-world agent applications" |
| SIC | Intent classification datasets (MASSIVE, BANKING77, CLINC150) |

Most caching papers either use small proprietary setups, embodied/GUI agent benchmarks, or intent classification datasets. **This is a gap our work can fill.**

The "Agentic Web" paper surveys the community's go-to web-agent benchmarks: WebArena, Mind2Web, BrowseComp, WebShop, Mini-WoB++, SWE-bench. However, **WebArena and Mind2Web operate through browser DOM actions** (CLICK, TYPE, SELECT on page elements), not a WebSearch/WebFetch API. Integrating them would require building a browser-control layer on top of Claude Code — a non-trivial engineering detour that would pull focus from the caching research itself.

---

## 3. Candidate Benchmarks

### 3.1 FRAMES *(primary recommendation)*

> Factuality, Retrieval, And Multi-step Evidenced Summarization  
> Released by Google DeepMind.

- **Scale**: ~824 tasks
- **Task type**: Multi-hop factual research. Each task requires finding 2–5 pieces of evidence from different web sources and synthesizing them into a final answer.
- **Why it fits EKB**:
  - Forces multiple WebSearch/WebFetch calls per task — matches our existing trace setup exactly.
  - Tasks cluster naturally by topic (science, sports, history, geography), so tasks within a cluster share execution skeletons. This is the structure that makes plan caching meaningful.
  - Has gold answers for exact-match evaluation.
  - Scale is 6× GAIA's base task count.
- **Limitation**: Multi-hop tasks have longer and more variable execution traces, which raises the bar for plan reuse.

---

### 3.2 PopQA *(secondary recommendation)*

> Entity-centric open-domain QA dataset.

- **Scale**: ~14,000 QA pairs (a stratified 1–2k subset is practical)
- **Task type**: Single-hop factual lookup about named entities (e.g., "Who directed X?", "What nationality is Y?", "When was Z founded?").
- **Why it fits EKB**:
  - Questions group naturally by *question type* (birth year, nationality, director, etc.). Tasks of the same type share nearly identical execution plans: WebSearch entity → extract fact → StructuredOutput.
  - This makes the caching story maximally interpretable: a plan cached from one "birth year" query should transfer cleanly to a new "birth year" query about a different entity.
  - Large enough to construct a meaningful KB/test split.
  - Single-hop structure keeps execution traces short and easy to analyze.
- **Limitation**: Tasks are simpler than GAIA Level 2; caching benefits are likely an upper bound rather than a realistic estimate for hard agentic tasks.

---

### 3.3 HotpotQA *(large-scale option)*

> Multi-hop QA over Wikipedia. Full-wiki setting requires the agent to find supporting documents without gold context — effectively a web search problem.

- **Scale**: ~113k total; 7.4k dev questions are practical
- **Task type**: Two-hop reasoning (bridge questions: "Find A's property, then use it to find B's property"; comparison questions: "Which of A and B has more X?").
- **Why it fits EKB**:
  - The hop structure creates natural execution-plan clusters: bridge questions share a "search → pivot → search → synthesize" skeleton.
  - Huge scale enables large KB construction and robust cache hit rate estimation.
  - Well-established benchmark with strong community adoption.
- **Limitation**: Wikipedia-centric tasks may have lower web-search diversity than FRAMES. Also needs filtering to remove questions answerable from model memory without search.

---

### 3.4 BrowseComp *(qualitatively harder option)*

> OpenAI's benchmark for deep web research tasks.

- **Scale**: ~100–300 tasks
- **Task type**: Hard, multi-step web research that requires backtracking, cross-referencing multiple sources, and extended browsing.
- **Why it fits EKB**:
  - Closest in spirit to GAIA Level 2–3 difficulty.
  - Genuinely impossible to answer from model memory — web search is mandatory.
  - Good for testing where caching helps on hard tasks and characterizing its upper bound.
- **Limitation**: Scale is still small; not ideal as a primary evaluation benchmark. Better suited as a secondary hard-task probe.

---

## 4. Recommendation

**FRAMES + PopQA** is the recommended pairing for initial EKB experiments:

| Property | FRAMES | PopQA (subset) |
|---|---|---|
| Scale | ~824 tasks | ~1–2k tasks |
| Task type | Multi-hop research | Single-hop entity lookup |
| Primary tools | WebSearch, WebFetch | WebSearch |
| Execution plan variety | High | Low (clusters tightly by question type) |
| Caching interpretability | Moderate | High |
| Evaluation metric | Exact match / F1 | Exact match |
| Difficulty | GAIA Level 1–2 | Easier than GAIA |

- **FRAMES** gives research-style tasks with natural topic clustering that mirror the GAIA spirit at larger scale.
- **PopQA** gives entity-lookup tasks where execution-plan similarity is almost perfectly predicted by question type — the cleanest setting to demonstrate and measure EKB caching benefit.
- Together they cover **two distinct difficulty and structure regimes**, enabling a richer analysis of *when* and *how much* plan caching helps.

If a third benchmark is added for completeness, use a **1k stratified sample of HotpotQA (full-wiki)** to cover the two-hop bridge/comparison structure that falls between PopQA's simplicity and FRAMES's complexity.

---

## 5. Compatibility Notes

All three recommended benchmarks can be run with the existing `runners/run_claude_task.py` pipeline with minimal changes:

- Tasks need to be converted to the JSONL format used by `data/gaia/`.
- Ground-truth answers need to be mapped to the `gold_answer` field in `normalized_trace.json`.
- No browser automation is required — WebSearch/WebFetch suffice.

WebArena and Mind2Web are **not recommended** for the current setup: they require browser DOM action spaces (CLICK, TYPE, SELECT) that are outside the current Claude Code tool repertoire.
