# Literature Survey Notes: Three-Level Caching for Multi-Tool LLM Agents

## 1. Positioning the Three Caching Levels

| Level                          | Cached / Reused Object                                                    | Main Question                                                                                    | Representative Work                                     |
| ------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| **L1: Execution-plan caching** | Plan skeletons, workflow templates, action transitions, procedural memory | Can we reuse prior execution structure to guide future tool use?                                 | APC, HMT, AgenticCache, WorkflowGen                     |
| **L2: Tool-result caching**    | Intermediate tool feedback/results (f_t)                                  | Can we avoid re-executing a tool call whose result is already known under an equivalent context? | ToolCaching, TVCACHE, FAME                              |
| **L3: Query-response caching** | Final answer (y), or a reusable response draft/intent                     | Can we directly return or refine a previous answer without running the full agent trajectory?    | GPTCache, MeanCache, Structured Intent Canonicalization |
---

## 2. Notation

The notation below follows the formulation in:

> Xu, Haoyuan, et al. “The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration.” arXiv preprint arXiv:2603.22862, 2026.

| Symbol                       | Meaning                                                 |
| ---------------------------- | ------------------------------------------------------- |
| $q\in\mathcal Q$             | User query / task instruction                           |
| $s_t\in\mathcal S$           | True environment state at step $t$                      |
| $o_t\in\mathcal O$           | Agent-visible observation at step $t$                   |
| $\mathcal D={d_i}_{i=1}^{N}$ | Tool space                                              |
| $d_t\in\mathcal D$           | Tool selected at step $t$                               |
| $u_t\in\mathcal U_{d_t}$     | Valid argument for tool $d_t$                           |
| $a_t=(d_t,u_t)$              | Tool-call action at step $t$                            |
| $f_t\in\mathcal F$           | Tool feedback, e.g., returned JSON, text, error message |
| $y\in\mathcal Y$             | Final answer                                            |
| $h_t$                        | Interaction history within the current task             |
| $\tau$                       | Execution trajectory of the current task                |
| $m_t$                        | Agent runtime memory / working state                    |
| $\pi_\theta$                 | Agent policy                                            |
| $\text{Exec}$                | Tool execution function                                 |
| $\mathcal C$                 | Generic cache                                           |
| $\eta_i$                     | Metadata associated with a cached entry                 |
| $\chi_t\in{0,1}$             | Cache-hit indicator                                     |

### Action Space

The multi-tool action space is

$$
\mathcal A := (\mathcal D\times\mathcal U)\cup {\text{Submit}(y)}.
$$

A tool-call action is

$$
a_t=(d_t,u_t).
$$

The agent samples the next action from a policy conditioned on the current history and runtime memory:

$$
a_t\sim \pi_\theta(\cdot\mid h_t,m_t).
$$

### Tool Execution

If (a_t=(d_t,u_t)), the tool is executed as

$$
(f_t,s_{t+1})\leftarrow \text{Exec}(d_t,u_t,s_t).
$$

The history is updated as

$$
h_{t+1}=h_t\oplus(a_t,f_t,o_{t+1}).
$$

### Interaction History

The interaction history up to step (t) is a within-task record:

$$
h_t := (q,o_0,(a_0,f_0,o_1),(a_1,f_1,o_2),\ldots,o_t).
$$

It records the current task’s observed execution prefix.

### Execution Trajectory

The execution trajectory is

$$
\tau := ((a_0,f_0),(a_1,f_1),\ldots,(a_{T-1},f_{T-1}),\text{Submit}(y)).
$$

Compared with $h_t$, the trajectory is a compact action-feedback trace, usually used for evaluation, cost accounting, or future cache construction.

### Cost-Aware Objective

A generic cost-aware objective is

$$
\max_\theta \mathbb E\left[
\text{Reward}(q,y,\tau)-\lambda\cdot \text{Cost}(\tau)
\right].
$$

---

# 3. L1: Execution-Plan Caching

Execution-plan caching reuses prior execution structure to guide future agent behavior. The cached object is not a tool result $f_t$ or final response $y$, but some form of reusable procedural knowledge.

A generic plan cache can be written as

$$
\mathcal C^{\text{plan}}={(k_i,p_i,\eta_i)}_{i=1}^{M},
$$

where $k_i$ is the retrieval key, $p_i\in\mathcal P$ is a reusable plan object, and $\eta_i$ stores metadata.

For a new query $q$, the agent retrieves or generates a plan:

$$
p_q =
\begin{cases}
\text{Adapt}(p_i,q), & \text{if a valid cached plan is retrieved},\\
\text{Plan}(q), & \text{otherwise}.
\end{cases}
$$

The policy becomes

$$
a_t\sim \pi_\theta(\cdot\mid h_t,m_t,p_q).
$$

The key point is that $p_q$ is cross-task execution knowledge, whereas $h_t$ only records the current task’s history.

---

## 3.1 Example of Plan Caching

Consider two related queries:

$$
q=\text{“Find the latest CUDA version supported by PyTorch and summarize compatibility.”}
$$

$$
q'=\text{“Find the latest TensorRT version and summarize CUDA compatibility.”}
$$

A reusable plan skeleton may be

$$
p=
\text{Search official docs}
\rightarrow
\text{Open compatibility table}
\rightarrow
\text{Extract version constraints}
\rightarrow
\text{Cross-check release notes}
\rightarrow
\text{Summarize}.
$$

The exact entities differ, but the execution structure is reusable.

---

## 3.2 Agentic Plan Caching (APC): Plan-Template Reuse

> Zhang, Qizheng, Michael Wornow, and Kunle Olukotun. “Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents.” NeurIPS 2026.

APC caches abstract plan templates extracted from previous successful executions.

Define the APC cache as

$$
\mathcal C_{\text{APC}}={(k_i,p_i)}_{i=1}^{M},
$$

where (k_i) is a task-level keyword and (p_i) is a reusable plan template.

For a new query, APC first computes

$$
k=K(q),
$$

where $K(\cdot)$ is a lightweight keyword extractor.

Then

$$
p_q=
\begin{cases}
\text{Adapt}_{\text{LLM}}(p_i,q), & \text{if } k=k_i,\\
\text{Plan}_{\text{LLM}}(q), & \text{otherwise}.
\end{cases}
$$

The agent acts with the retrieved/adapted plan as additional conditioning:

$$
a_t\sim \pi_\theta(\cdot\mid h_t,m_t,p_q).
$$

If a cache miss occurs and the trajectory succeeds, APC abstracts the trajectory into a reusable template:

$$
p_{\text{new}}=\text{Abstract}(\tau),
$$

and inserts it into the cache:

$$
\mathcal C_{\text{APC}}
\leftarrow
\mathcal C_{\text{APC}}\cup {(K(q),p_{\text{new}})}.
$$

APC’s reuse pipeline is

$$
q
\rightarrow
K(q)
\rightarrow
\text{retrieve plan template}
\rightarrow
\text{adapt template}
\rightarrow
\text{execute}.
$$

---

## 3.3 Hierarchical Memory Tree (HMT): Hierarchical Web-Procedural Memory

> Tan, Yunteng, Zhi Gao, and Xinxiao Wu. “Enhancing Web Agents with a Hierarchical Memory Tree.” arXiv preprint arXiv:2603.07024, 2026.

HMT builds a hierarchical memory tree from successful web-agent trajectories.

The memory object is

$$
\mathcal T_{\text{HMT}}=(\mathcal I,\mathcal G,\mathcal L),
$$

where $I\in\mathcal I$ are intent nodes, $g\in\mathcal G$ are stage/subgoal nodes, and $\ell\in\mathcal L$ are action nodes.

A compact representation is

$$
I=(\text{intent},\text{constraints}),
$$

$$
g=(\text{stage name},\text{pre}(g),\text{post}(g)),
$$

$$
\ell=(\rho,\delta),
$$

where $\rho$ is an abstract action pattern and (\delta) is a semantic element description.

At test time, HMT retrieves memory relevant to the current query, history, and observation:

$$
P_t=\text{RetrieveHMT}(q,h_t,o_t,\mathcal T_{\text{HMT}}).
$$

The planner selects the current stage:

$$
g_t\sim \pi_{\text{plan}}(\cdot\mid q,h_t,o_t,P_t).
$$

The actor grounds the abstract action into a concrete web action:

$$
a_t\sim \pi_{\text{act}}(\cdot\mid q,h_t,o_t,g_t,P_t).
$$

For a web agent,

$$
a_t=(d_t,u_t),
$$

where $d_t$ may be CLICK, TYPE, SELECT, etc., and $u_t$ contains the target UI element and optional argument.

### Example

Query:

```text
I want to fly to New York.
```

Intent level:

```text
Intent: Book a flight
Constraints: destination = NYC
```

Stage level:

```text
Search Flight
Filter Results
Select Flight
Enter Passenger Info
Confirm Booking
```

A stage node may store:

```text
Stage: Select Flight

Pre-condition:
  - result list is visible
  - flight cards are visible
  - price/sort/filter options are visible

Post-condition:
  - summary page is visible
  - selected flight details are visible
```

Action level:

A raw trajectory may store:

```text
CLICK element_id="#search-button-483"
```

HMT instead stores:

```text
Action pattern:
  CLICK

Semantic element description:
  role = button
  text = "Search" or "Find"
  position = bottom of form
```

Thus, HMT avoids brittle replay of raw DOM IDs and instead reuses semantic web-action patterns.

Compared with APC, HMT does not simply retrieve a whole-task plan skeleton. It retrieves stage-aligned web-procedural memory and grounds it in the current observation.

---

## 3.4 AgenticCache: Transition-Level Plan/Action Caching

> Kim, Hojoon, Yuheng Wu, and Thierry Tambe. “AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents.” arXiv preprint arXiv:2604.24039, 2026.

AgenticCache uses the term “plan,” but its plan object is closer to a short-horizon high-level action or macro-action than a full workflow.

Its main empirical claim is **plan locality**:

$$
\text{the next plan/action is often predictable from the current plan/action.}
$$

Instead of caching a whole plan template, AgenticCache caches transitions:

$$
p_i\rightarrow p_j.
$$

The cache is

$$
\mathcal C_{\text{AC}}=

{(p_i,p_j,z_{ij},C_{ij},I_j)},
$$

where $z_{ij}$ is the valid state-metadata range, $C_{ij}$ is the observed transition count, and $I_j$ is the reliability or importance of $p_j$, estimated by a background LLM.

Candidate transitions are scored as

$$
S(p_i\rightarrow p_j)=C_{ij}\cdot I_j.
$$

At step (t), the next plan/action is selected by

$$
p_t=
\arg\max_{p_j\in\mathcal F(p_{t-1},s_t)}
S(p_{t-1}\rightarrow p_j),
$$

where $\mathcal F(p_{t-1},s_t)$ contains transitions consistent with the current state metadata.

Then the agent acts immediately while the LLM validates asynchronously:

$$
(p_{t-1},s_t)
\rightarrow
\text{retrieve next }p_t
\rightarrow
\text{act immediately}.
$$

AgenticCache is best viewed as **transition-level plan/action caching**, not tool-result caching, because it caches the decision of what to execute next, not the feedback $f_t$ returned by execution.

---

## 3.5 WorkflowGen: Trajectory-Driven Workflow Template Reuse

> Wei, Ruocan, Shufeng Wang, and Ziwei Shi. “WorkflowGen: An Adaptive Workflow Generation Mechanism Driven by Trajectory Experience.” arXiv preprint arXiv:2604.19756, 2026.

WorkflowGen caches richer workflow experience than APC. It extracts both workflow-level templates and node-level execution experience from historical trajectories.

Compared with APC:

$$
\text{APC: }
\mathcal C_{\text{APC}}={(K(q_i),p_i)}.
$$

$$
\text{WorkflowGen: }
\mathcal C_{\text{WG}}={(k_i,p_i,e_i,\eta_i)},
$$

where $p_i$ is an executable workflow template, $e_i$ contains node-level experience, and $\eta_i$ contains trajectory metadata.

WorkflowGen uses adaptive routing. Let

$$
s=\max_i \text{Sim}(\phi(q),k_i).
$$

where $\phi(q)$ is the query embedding and $k_i$ is the cached embedding.

Then

$$
p_q=
\begin{cases}
p_i, & \text{if } s>\delta_{\text{high}},\\
\text{Rewrite}(p_i,q,e_i), & \text{if } \delta_{\text{low}}<s\le \delta_{\text{high}},\\
\text{Plan}_{\text{LLM}}(q), & \text{if } s\le \delta_{\text{low}}.
\end{cases}
$$

WorkflowGen’s main reuse object is lengthy, including action sequences, dependencies, fixed and variable nodes, context, metadata, and node-level success/failure experience.

---

## 3.6 Summary: Execution-Plan Caching Methods

| Method           | Reused Object                                     | Cache / Memory Form                             | Retrieval Signal                           | Main Difference                                                    |
| ---------------- | ------------------------------------------------- | ----------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| **APC**          | Whole-task plan template                          | ${(K(q_i),p_i)}$                                | Task keyword match                         | Reuses and adapts abstract plan templates                          |
| **HMT**          | Hierarchical web-procedural memory                | $\mathcal T=(\mathcal I,\mathcal G,\mathcal L)$ | Intent, stage condition, observation match | Reuses intent-stage-action memory and grounds actions semantically |
| **AgenticCache** | Short-horizon plan/action transition              | ${(p_i,p_j,z_{ij},C_{ij},I_j)}$                 | Previous plan/action and state metadata    | Caches local transitions, not whole workflows                      |
| **WorkflowGen**  | Executable workflow template plus node experience | ${(k_i,p_i,e_i,\eta_i)}$                        | Query similarity and adaptive thresholds   | Reuses, rewrites, or regenerates workflow templates                |

---

# 4. L2: Tool-Result Caching

Tool-result caching reuses intermediate tool feedback (f_t) when the same or equivalent tool call has already been executed under a valid context.

Without caching:

$$
(f_t,s_{t+1})\leftarrow \text{Exec}(d_t,u_t,s_t).
$$

With tool-result caching, execution is replaced by a cache-augmented executor:

$$
(f_t,s_{t+1},\chi_t)
\leftarrow
\text{Exec}_{\mathcal C}(d_t,u_t,s_t,h_t).
$$

Define the tool-result cache as

$$
\mathcal C^{\text{tool}}={(\kappa_i,f_i,\eta_i)}_{i=1}^{M},
$$

where $\kappa_i$ is the cache key, $f_i$ is cached feedback, and $\eta_i$ stores metadata such as timestamp, TTL, tool type, execution cost, provenance, or state signature.

The key design choices are the key extraction function $\Gamma$(.) and validation function $\text{Valid}$(.):

$$
\kappa_t=\Gamma(d_t,u_t,s_t,h_t),
$$

$$
\text{Valid}(\kappa_t,\eta_i,s_t,h_t)\in\{0,1\}.
$$

Then

$$
\text{Exec}_{\mathcal C}(d_t,u_t,s_t,h_t)=

\begin{cases}
(f_i,\widehat{s}_{t+1},\chi_t=1), & \text{if } \kappa_t=\kappa_i \text{ and } \text{Valid}(\kappa_t,\eta_i,s_t,h_t)=1,\\
(\text{Exec}(d_t,u_t,s_t),\chi_t=0), & \text{otherwise}.
\end{cases}
$$

The core question is:

$$
\text{What context must match before a previous } f_i \text{ can safely replace fresh execution?}
$$

---

## 4.1 ToolCaching: Individual Tool-Request Caching

> Zhai, Yi, et al. “ToolCaching: Towards Efficient Caching for LLM Tool-calling.” arXiv preprint arXiv:2601.15335, 2026.

ToolCaching caches the result of individual tool-call requests if they are cacheable and valuable.

Its key function is approximately

$$
\Gamma_{\text{TC}}(d_t,u_t,s_t,h_t)
=
\text{Hash}(d_t,u_t).
$$

Thus, ToolCaching is mostly request-level and weakly state-aware.

A cache hit occurs when

$$
\text{Hash}(d_t,u_t)\in \mathcal C^{\text{tool}}
$$

and the entry passes metadata-based validation.

ToolCaching emphasizes two decisions.

1) Cacheability: Some tool calls should not be cached, such as state-mutating commands or results with very short freshness windows.

2) Admission and Eviction: ToolCaching estimates a value score:
$$
\text{Value}(d_t,u_t)
=
v(\text{latency},\text{cost},\text{result size},\text{TTL}).
$$

The system prioritizes caching outputs that are expensive to compute, frequently reused, small enough to store, and sufficiently stable.

---

## 4.2 TVCACHE: Stateful Tool-Value Caching

> Kumar, Abhishek Vijaya, et al. “TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents.” arXiv preprint arXiv:2602.10986, 2026.

TVCACHE is designed for stateful agent rollouts, where a tool result may depend on the preceding tool-call sequence.

Define the current tool-call prefix as

$$
\rho_t=((d_0,u_0),(d_1,u_1),\ldots,(d_t,u_t)).
$$

TVCACHE uses a trajectory-aware key:

$$
\Gamma_{\text{TV}}(d_t,u_t,s_t,h_t)=\rho_t.
$$

It stores a Tool Call Graph:

$$
G_q=(V_q,E_q),
$$

where each node corresponds to a tool call and stores

$$
v=((d_t,u_t),f_t,\text{snapshot}_t).
$$

A cache hit occurs when the current prefix matches a cached path:

$$
\rho_t\in G_q.
$$

If the full prefix exists, TVCACHE returns the cached result. If only a partial prefix exists, it can restore the sandbox snapshot at the longest matched prefix and execute only the unmatched suffix.

TVCACHE’s strategy is:

$$
\text{reuse } f_t \text{ only under a matched state-producing tool-call prefix}.
$$

This is safer for stateful tools but more restrictive than request-level caching.

---

## 4.3 FAME: Runtime State Reuse for MCP-Enabled FaaS Agents

> Kulkarni, Varad, et al. “Optimizing FaaS Platforms for MCP-enabled Agentic Workflows.” arXiv preprint arXiv:2601.14735, 2026.

FAME focuses on optimizing serverless execution for MCP-enabled agentic workflows.

It is related to tool-result caching, but it is better described as **runtime state reuse** rather than a pure key-value cache for $f_t$.

At a high level, FAME tries to preserve or restore useful workflow/runtime state across serverless invocations, reducing cold-start and repeated initialization overhead.

A simplified representation is

$$
\mathcal C_{\text{FAME}}=
{(\text{session id},\text{invocation state},\eta_i)}.
$$

Unlike ToolCaching and TVCACHE, the cached object is not simply an individual tool feedback $f_t$. It may include execution context, memory, loaded tool state, or workflow state needed to continue an agentic execution efficiently.

---

## 4.4 Summary: Tool-Result Caching Methods

| Method          | Cache Key                                              | Reused Object                                | State Awareness   | Main Difference                                                 |
| --------------- | ------------------------------------------------------ | -------------------------------------------- | ----------------- | --------------------------------------------------------------- |
| **ToolCaching** | $\text{Hash}(d,u)$ plus metadata/TTL                   | Individual tool feedback $f_t$               | Weak / rule-based | Focuses on cacheability, admission, and eviction                |
| **TVCACHE**     | Tool-call prefix $\rho_t=((d_0,u_0),\ldots,(d_t,u_t))$ | Tool feedback under matched trajectory state | Stronger          | Reuses results only under matched execution prefixes            |
| **FAME**        | Session/invocation/runtime identifiers                 | Runtime or workflow state                    | Stateful          | Optimizes serverless runtime reuse rather than pure $f_t$ reuse |

---

# 5. L3: Query-Response Caching

Query-response caching reuses a previous final answer $y_i$, or a refined version of it, before launching the full agent trajectory.

Define a query-response cache:

$$
\mathcal C^{\text{resp}}
=
{(\kappa_i,q_i,y_i,\eta_i)}_{i=1}^{M},
$$

where $\kappa_i$ is the response-cache key, $q_i$ is the original query, $y_i$ is the cached final answer, and $\eta_i$ stores context, timestamp, evidence, trajectory, freshness, or user metadata.

Define a response key:

$$
\kappa_q=\Lambda(q,o_0,m_0).
$$

Then

$$
y=
\begin{cases}
y_i, & \text{if a valid direct response hit is found},\\
\text{Refine}(q,y_i,\eta_i), & \text{if a partially reusable response is found},\\
\text{Agent}(q), & \text{otherwise}.
\end{cases}
$$

This layer is more aggressive than plan or tool-result caching because it may skip the entire trajectory:

$$
\tau^{\mathcal C}=(\text{Submit}(y_i)).
$$

The main risk is unsafe response reuse: two queries may be semantically similar but require different answers.

---

## 5.1 GPTCache: Semantic Cache for LLM Applications

> Bang, Fu. “GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings.” NLP-OSS, 2023.

GPTCache is a general-purpose semantic cache for LLM applications.

Its pipeline is:

```text
Pre-Processor
→ Embedding Generator
→ Cache Manager
→ Similarity Evaluator
→ Post-Processor
```

In our notation, GPTCache can be written as

$$
\mathcal C_{\text{GPTCache}}
=
{(\phi(q_i),q_i,y_i,\eta_i)}_{i=1}^{M}.
$$

For a new query,

$$
i^\star=\arg\max_i \text{Sim}(\phi(q),\phi(q_i)).
$$

If

$$
\text{Sim}(\phi(q),\phi(q_{i^\star}))\ge \delta,
$$

then GPTCache returns or post-processes the cached response:

$$
y=y_{i^\star}
\quad \text{or} \quad
y=\text{PostProcess}(q,y_{i^\star}).
$$

GPTCache is mainly designed for LLM request-response caching, not multi-step agent trajectories.

---

## 5.2 MeanCache: User-Centric Semantic Query-Response Reuse

> Gill, Waris, et al. “MeanCache: User-Centric Semantic Caching for LLM Web Services.” IPDPS, 2025.

MeanCache stores user-specific query-response pairs and uses semantic similarity plus context-chain matching.

Its cache can be written as

$$
\mathcal C_{\text{MC}}
=

{(e_i,q_i,y_i,c_i)}_{i=1}^{M},
$$

where

$$
e_i=\phi_u(q_i)
$$

is a user-adapted query embedding, and (c_i) is the context chain associated with (q_i).

For a new query,

$$
e_q=\phi_u(q),
$$

$$
i^\star=\arg\max_i \text{Sim}(e_q,e_i).
$$

A cache hit occurs if

$$
\text{Sim}(e_q,e_{i^\star})\ge \delta
$$

and

$$
\text{ContextMatch}(h_0,c_{i^\star})=1.
$$

Then

$$
y=y_{i^\star}.
$$

MeanCache emphasizes that semantic similarity alone is insufficient because the same query can require different answers in different conversational contexts.

---

## 5.3 Structured Intent Canonicalization (SIC): Safer Cache-Key Construction

> Basu, Abhinaba. “Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning.” arXiv preprint arXiv:2602.18922, 2026.

This paper argues that agent caching often fails because embedding similarity does not reliably preserve action intent. For example, two semantically similar queries may require different tool calls or different final answers.

The proposed solution is W5H2 structured intent canonicalization:

$$
\text{Who},\text{ What},\text{ When},\text{ Where},\text{ Why},\text{ How},\text{ How Much}.
$$

Define

$$
z_q=(\text{What}(q),\text{Where}(q)),
$$

$$
r_q=(\text{Who}(q),\text{When}(q),\text{How}(q),\text{How Much}(q),\ldots).
$$

Here $z_q$ is the canonical intent key, and (r_q) stores response- or execution-determining parameters.

For query-response caching, a safe reuse rule is

$$
y=
\begin{cases}
y_i, & \text{if } (z_q,r_q)=(z_i,r_i),\
\text{Refine}(q,y_i,r_q), & \text{if } z_q=z_i \text{ but } r_q\ne r_i,\\
\text{Agent}(q), & \text{otherwise}.
\end{cases}
$$

Example:

```text
Query: "Check email from Alice about this meeting."
```

Canonical intent:

```text
What = retrieve
Where = email
```

Parameters:

```text
Who = Alice
Why = meeting
```

---

## 5.4 Summary: Query-Response Caching Methods

| Method        | Cache Key                                              | Cached Object                                                    | Reuse Rule                                                                  | Main Risk                              |
| ------------- | ------------------------------------------------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------- |
| **GPTCache**  | Query embedding, optionally with preprocessing/context | Final response $y_i$                                             | Return or post-process most similar cached response                         | False semantic hit                     |
| **MeanCache** | User-adapted query embedding plus context chain        | Final response $y_i$                                             | Return $y_i$ if semantic similarity and context match                       | Context mismatch or low hit rate       |
| **SIC**       | Canonical intent $z_q$ plus parameters $r_q$           | Intent key, response, or reusable template depending on use case | Reuse exactly if intent and parameters match; refine if only intent matches | Under-keying if parameters are ignored |

---

# 6. Overall Comparison Across the Three Levels

| Level                          | What Is Reused?                                                  | Where It Intervenes                | Example Methods                     |
| ------------------------------ | ---------------------------------------------------------------- | ---------------------------------- | ----------------------------------- |
| **L1: Execution-plan caching** | Plan skeletons, workflows, action transitions, procedural memory | Before or during action selection  | APC, HMT, AgenticCache, WorkflowGen |
| **L2: Tool-result caching**    | Tool feedback $f_t$ or runtime state                             | During tool execution              | ToolCaching, TVCACHE, FAME          |
| **L3: Query-response caching** | Final answer $y_i$, response draft, or canonical intent          | Before the agent trajectory starts | GPTCache, MeanCache, SIC            |

The three levels are complementary but distinct:

$$
\text{L1 modifies } \pi_\theta(\cdot\mid h_t,m_t).
$$

$$
\text{L2 modifies } \text{Exec}(d_t,u_t,s_t).
$$

$$
\text{L3 modifies whether the full trajectory is needed.}
$$
