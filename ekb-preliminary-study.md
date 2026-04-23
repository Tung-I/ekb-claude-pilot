# EKB Preliminary Study: Execution Knowledge Base Validation

**Goal:** Collect preliminary results that validate the Execution Knowledge Base (EKB) concept for the ADN proposal. These results address two specific reviewer criticisms and one TODO in the proposal.

**Timeline:** Results needed before proposal submission. Prioritize Study 1 (most critical).

**Contact:** [PI Hui Guan]

---

## Background (Read This First)

The ADN proposal argues that agent workloads (multi-step LLM + tool-use workflows) can be served more efficiently by building an **Execution Knowledge Base** — a structured store of *how* agents execute queries, not just *what* they output. The EKB has two layers:

- **Layer 1 (Result Cache):** Caches final agent responses. Semantically similar queries get a cached answer with zero computation.
- **Layer 2 (Execution Knowledge):** Stores execution traces — tool call sequences, step counts, resource consumption, tier success rates. These are aggregated into **probabilistic execution graphs** that predict how a new query will execute.

The EKB enables **query routing**: given a new query, predict whether it's simple (serve at the edge with a 7B model) or complex (route to cloud with a 70B+ model), without executing it first.

**The core assumption we need to validate:** Queries that are semantically similar also tend to execute similarly (same tools, similar step count, similar resource needs). If this assumption holds, we can use query embeddings to look up execution knowledge and make routing decisions.

---

## Study 1: Semantic-Execution Similarity (CRITICAL)

**Reviewer concern:** "A core mechanism of the system — the ability to match incoming queries to relevant execution knowledge — rests on an unvalidated assumption."

### Research Question
Do semantically similar agent queries produce similar execution profiles? Specifically, can we predict execution characteristics (step count, tool sequence, tier suitability) from query embeddings?

### Method

**Step 1: Collect execution traces from an agent benchmark.**

Pick benchmarks and agent setup from the options below. Ideally, collect traces from **two benchmarks** across different domains to show the EKB generalizes.

**Benchmark options (aligned with the proposal's three use cases):**

| Benchmark | Domain | # Tasks | Tools/Actions | Why it fits |
|-----------|--------|---------|---------------|-------------|
| **WebArena** | Conversational web | 812 | Browser nav, form fill, click, type, search | Directly matches Use Case 1 (shopping, customer support). Self-hostable (e-commerce, forum, GitLab, CMS sites). Natural difficulty range: simple lookups (2-3 steps) vs. complex multi-page comparisons (15+ steps). |
| **GAIA** | General assistant + web | 466 | Web browsing, calculator, code exec, API calls | Real-world questions with 3 difficulty levels. Excellent variability: easy questions need 1-2 tool calls, hard ones need 10+. Well-cited, clean dataset. |
| **ALFWorld** | Embodied AI | 200+ | Navigate, pick up, put down, open, examine | Directly matches Use Case 2 (embodied agents). Text-based embodied tasks in simulated household. Simple fetch tasks (3 steps) vs. complex multi-room planning (15+ steps). Easy to run (Python, no GPU needed). |
| **τ-bench** | Customer service | 200+ | Tool calls under domain policies, user interaction | Conversational agents with strict policy compliance. Captures real enterprise agent workload patterns. |
| **AgentBench** | Multi-domain | 1000+ | Code exec, web browsing, retrieval, reasoning | Covers multiple environments. Good for showing EKB works across domains. |

**Recommended combination:** WebArena (conversational web) + ALFWorld (embodied AI). This covers two of the three proposal use cases directly, and the two benchmarks have very different tool vocabularies, which tests whether execution knowledge transfers across domains.

**Agent setup options (pick one):**

*Option A: Claude Code as the agent (Recommended — fixed cost, richest traces).*
Use the Claude Code CLI (`claude`) with the Max subscription. Claude Code is itself a tool-using agent that plans, reads, writes, runs commands, and iterates — producing naturally rich execution traces. It works well for **WebArena-style tasks** (give it a web task description and let it use browser tools) and **general assistant tasks** (GAIA). Claude Code's conversation logs (JSONL in `~/.claude/projects/`) contain detailed per-turn records of every tool call.

Setup:
```bash
# For each benchmark task, run Claude Code non-interactively
claude --print "<task description>" --output-format json
```
The `--print` flag runs non-interactively. Parse the JSON output for tool calls (Read, Edit, Bash, Grep, etc.), step counts, and latencies. Alternatively, use the Claude Code SDK (`@anthropic-ai/claude-code`) to run programmatically and capture structured output in a batch loop.

Advantages: fixed cost (subscription), detailed tool-use traces, realistic multi-step agent behavior, no infrastructure setup.

*Option B: Self-hosted open-source agents (free, multi-model).*
Run a local LLM via vLLM (e.g., Llama-3.1-8B and/or Llama-3.1-70B on the Unity cluster) with an agent framework on top:
- **For WebArena:** Use the official WebArena agent harness with a local model backend via vLLM
- **For ALFWorld:** Use a ReAct agent (LangChain/LangGraph) — ALFWorld is lightweight and runs on CPU
- **For GAIA:** LangChain ReAct agent with web browsing tools
- General: **OpenHands** supports multiple benchmarks and logs detailed traces

Advantages: free compute (use Unity GPUs), can run multiple model sizes (8B vs 70B) to get **multi-tier data** directly — this is especially valuable because it lets us compare execution traces across model sizes, directly simulating the ADN's outer-edge vs. cloud tiers.

*Option C: Hybrid — Claude Code + local small model (Best for the proposal).*
Run the same benchmark tasks with both Claude Code (simulating cloud-tier 70B+ model) and a local 8B model via vLLM (simulating edge-tier). This directly produces multi-tier execution data: same query, two different execution traces, showing how execution profiles differ by model capability. For example, on a WebArena shopping task, Claude might complete it in 5 steps while the 8B model needs 15 steps or fails entirely — exactly the kind of data that motivates the ADN's tier-based routing.

**What to log per task:**
- The input query/task description (verbatim text)
- The full execution trace: ordered list of (action_type, action_detail, latency) tuples
  - action_type: LLM_call, tool_call, file_read, file_write, bash_command, retrieval, etc.
  - action_detail: which tool, which API, which model
- Total step count (number of agent turns/actions)
- Total wall-clock time
- Total tokens consumed (input + output)
- Whether the task succeeded or failed
- Which model was used

**Minimum scale:** 200+ tasks with traces per benchmark. WebArena has 812 tasks (start with a 200-300 subset), ALFWorld has 200+ tasks (run all). More is better.

**Step 2: Compute query embeddings.**

For each task's input query, compute embeddings using:
- A pretrained sentence encoder (e.g., `all-MiniLM-L6-v2` or `E5-large-v2`)
- Optionally also try `text-embedding-3-small` from OpenAI for comparison

**Step 3: Analyze semantic-execution correlation.**

Produce the following analyses:

**(a) Clustering analysis.** Cluster queries by their semantic embeddings (k-means or HDBSCAN). For each cluster, compute the variance of execution features (step count, tool set, latency). Compare within-cluster variance to overall variance. **We want to show:** queries in the same semantic cluster have more similar execution profiles than random pairs.

Metric: *intra-cluster execution variance / total execution variance* — lower is better. A ratio significantly below 1.0 validates the assumption.

**(b) Execution prediction from embeddings.** Train a simple classifier/regressor:
- Input: query embedding
- Output: predicted step count (regression), predicted tool set (multi-label classification), predicted difficulty tier (easy/medium/hard classification)

Use 80/20 train/test split. Report accuracy/F1/MAE.

**We want to show:** Even a simple model can predict execution characteristics from query embeddings with reasonable accuracy (e.g., >70% tier classification accuracy).

**(c) Visualization.** Produce a 2D t-SNE or UMAP plot of query embeddings, colored by:
- Step count (gradient: green=few steps, red=many steps)
- Difficulty tier (easy/medium/hard)

**We want to show:** Visually, queries that cluster together tend to have similar colors — i.e., semantic similarity correlates with execution similarity.

### Deliverables for the Proposal
- One figure: t-SNE/UMAP visualization (colored by step count or difficulty)
- One table or figure: clustering analysis results (within-cluster vs total variance)
- One table: prediction accuracy (step count MAE, tier classification accuracy)
- 1-2 sentences describing the finding for the proposal text

---

## Study 2: EKB Cold-Start Convergence (IMPORTANT)

**Reviewer concern:** "How long does it take for the EKB to become useful? What is the performance penalty during the bootstrapping phase?"

### Research Question
How many execution traces per query class does the EKB need before its routing predictions become accurate?

### Method

**Option A: Use the data from Study 1.**

Simulate the EKB bootstrapping process:
1. Order the execution traces chronologically (or randomly shuffle to simulate arrival order).
2. Simulate the EKB accumulating traces one by one.
3. After every N traces (e.g., N=10, 20, 50, 100, 200), evaluate routing accuracy:
   - For each held-out query, use the EKB's accumulated knowledge to predict the best tier (or step count, or difficulty).
   - Compare prediction to ground truth.
4. Plot a **convergence curve**: x-axis = number of traces seen, y-axis = routing accuracy.

**We want to show:** Routing accuracy rises quickly (e.g., reaches 80% within 50-100 traces) and plateaus, demonstrating that the EKB becomes useful fast.

**Option B: Use the multi-tier VLM study data (if available).**

If traces from the PIs' existing multi-tier VLM study are available, use those instead. This has the advantage of being real multi-tier data (edge vs. cloud execution) rather than simulated. Same analysis as above.

### Deliverables for the Proposal
- One figure: convergence curve (traces seen vs. routing accuracy)
- One number: "The EKB achieves X% routing accuracy after only Y traces" for the proposal text

---

## Study 3: Execution Variability Characterization (NICE-TO-HAVE)

This study isn't addressing a specific reviewer concern but provides useful context for the intro and motivation sections.

### Research Question
How variable are agent execution profiles across queries? This motivates the EKB: if all queries executed identically, we wouldn't need execution knowledge.

### Method

Using the traces collected in Study 1:
1. Compute basic statistics: distribution of step counts, distribution of tool calls per query, distribution of latency.
2. Compute the **cost amplification factor**: ratio of internal model calls to external user queries. (The proposal mentions this as an evaluation metric.)
3. Identify the "easy" vs "hard" split: what fraction of queries could be served by a small model in <3 steps vs require 10+ steps?

### Deliverables for the Proposal
- One histogram: distribution of step counts across queries (showing high variability)
- One number: "X% of queries complete in ≤3 steps, while Y% require 10+ steps"
- One number: average cost amplification factor

---

## Practical Notes

**Trace format (minimum per query):**
```json
{
  "query_id": "webarena_042",
  "query_text": "Find the cheapest wireless mouse on the shopping site and add it to cart",
  "benchmark": "webarena",
  "agent": "claude-code",
  "model": "claude-sonnet-4",
  "steps": [
    {"step": 1, "type": "web_navigate", "tool": "browse", "target": "shopping.example.com", "latency_ms": 1500},
    {"step": 2, "type": "llm_call", "tool": "reasoning", "latency_ms": 900, "tokens_in": 3000, "tokens_out": 200},
    {"step": 3, "type": "web_click", "tool": "click", "target": "search_bar", "latency_ms": 200},
    {"step": 4, "type": "web_type", "tool": "type", "target": "wireless mouse", "latency_ms": 150},
    {"step": 5, "type": "llm_call", "tool": "reasoning", "latency_ms": 1100, "tokens_in": 5000, "tokens_out": 300},
    {"step": 6, "type": "web_click", "tool": "click", "target": "sort_by_price", "latency_ms": 200},
    {"step": 7, "type": "web_click", "tool": "click", "target": "add_to_cart", "latency_ms": 200}
  ],
  "total_steps": 7,
  "total_llm_calls": 2,
  "total_tool_calls": 5,
  "total_latency_ms": 4250,
  "total_tokens": 8600,
  "success": true,
  "tools_used": ["browse", "click", "type", "reasoning"]
}
```

**Compute/cost estimate:**
- **Claude Code (Max subscription):** Fixed cost. Run 400-800 tasks across WebArena + ALFWorld. Main constraint is rate limits — may take 2-3 days of sequential runs.
- **Self-hosted (Unity cluster):** Free. Need 1x A100 for 8B model, 2-4x A100 for 70B. WebArena needs the self-hosted web environment (Docker). ALFWorld is CPU-only.
- **Hybrid:** Combine both. ~3-5 days total.
- **Embedding computation:** Negligible (runs on CPU in seconds).
- **Analysis:** Runs on a laptop.

**Parsing Claude Code traces:**
Claude Code stores conversation logs as JSONL files in `~/.claude/projects/<project-hash>/`. Each line is a message with role, content, and tool use blocks. A simple Python script can extract:
- All tool calls (type=`tool_use`) with their names and inputs
- Timing between messages (approximate latency)
- Token counts from usage metadata
- Success/failure from final assistant message

**Priority order:**
1. Study 1 (semantic-execution similarity) — this is the most critical reviewer concern
2. Study 2 (cold-start convergence) — second most important
3. Study 3 (variability characterization) — nice to have, and mostly falls out of Study 1 data

Studies 2 and 3 reuse data from Study 1, so the main effort is collecting the traces.

**Recommended approach:** Start with **WebArena + Claude Code** (Option A) — it directly matches the proposal's conversational web use case and produces rich traces at fixed cost. Then add **ALFWorld** (lightweight, CPU-only) for the embodied AI domain. If time permits, run Option C (same WebArena tasks with a local 8B model) to get multi-tier comparison data — this directly demonstrates the ADN's edge-vs-cloud execution difference and would be the strongest possible preliminary result.
