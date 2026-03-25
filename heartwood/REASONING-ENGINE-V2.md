# Reasoning Engine v2: Architecture Document

**Status:** Proposed → Build starts now
**Author:** Rodspeed + Claude (multi-session review)
**Date:** 2026-03-25
**Scope:** Multi-phase upgrade to transform Heartwood's reasoning engine from an LLM prompt wrapper into a research-grounded cognitive system that compounds across runs.

---

## 0. Strategic Context

### The Fork in the Road

The SaaS ship plan says "Next: 3-month distribution experiment." The question is: distribution of *what*?

**Path A (Stripe first):** Ship billing → get paying users → build v2 as a product upgrade.
**Path B (Open-source first):** Build v2 → open-source it → use that as the distribution play.

**Decision: Path B.** The reasoning is:

"Try my note-taking app" competes with Obsidian, Notion, and a hundred others. "Here's how you give agents a memory that compounds" is a thesis that finds its own audience. The people who show up for that *are* the first users, self-selected for caring about the problem.

Phase 1 (graph analysis) + Phase 2 (minimal MCP) is the launch artifact. Not the full v2. Just enough to demonstrate the thesis.

### Kill Date

**April 15, 2026:** Repo public, blog post live, MCP server runnable by anyone.

**May 15, 2026:** Evaluate traction. "Works" means:
- 100 GitHub stars in 30 days (thesis resonates beyond immediate network)
- 10 people running it locally (issues filed, discussions opened — not star-and-forget)
- 3 people connecting the MCP server to their own Claude setup (agent memory thesis validated)

If those numbers aren't hit, go to Path A (Stripe + direct outreach) or archive the SaaS ambition. The experiment is the thing; the data decides.

If the numbers *are* hit, add Stripe — because now there are people who want a hosted version.

### Revised Phase Sequence

| Phase | What | Effort | Ships |
|-------|------|--------|-------|
| **1** | Graph Analysis Layer | 1 week | Internal |
| **2** | Minimal MCP (4 tools) | 3-5 days | **Open-source launch** |
| **3** | Memory System | 1-2 weeks | Post-launch upgrade |
| **4a** | Claim Extraction (validated) | 1 week | Internal validation |
| **4b** | Belief Revision (built on 4a) | 1-2 weeks | Post-validation |
| **5** | Link Prediction + Rule Learning | 1-2 weeks | Post-belief layer |

Phases 1-2 are the launch scope. Phases 3-5 are post-launch, gated on traction.

---

## 1. Problem Statement

The current reasoning engine (`reason.py`, ~1900 lines) sends the full graph to Claude Haiku and asks it to find connections, gaps, contradictions, clusters, bridges, drift, and synthesis opportunities. It works — but it has three structural weaknesses:

1. **No compounding.** Each run starts from near-scratch. Phase 2.5 added prior-report context, but that's a text blob injected into prompts — not a memory system. Run 47 doesn't meaningfully build on run 12.

2. **The LLM does everything.** Both structural reasoning (which nodes are similar, which are far apart, where the clusters are) and semantic reasoning (what do these connections *mean*). Graph algorithms are better at the first job; LLMs are better at the second. Mixing them wastes tokens and reduces quality.

3. **No formal belief tracking.** Contradiction detection sends high-similarity pairs to Haiku and asks "do these conflict?" There's no provenance, no confidence scoring, no entrenchment ordering, no temporal awareness. The engine can't tell you *why* it believes something or *what changed*.

The v2 architecture addresses all three by introducing four new layers: a **graph analysis layer** (algorithmic), an **MCP interface** (agent integration), a **memory system** (compounding), and a **belief layer** (formal).

---

## 2. Design Principles

1. **Graph algorithms do structural reasoning; Claude does semantic reasoning.** This is the consistent finding across 2024-2025 research (GNN-RAG, HippoRAG, SubgraphRAG). Don't ask an LLM to find graph patterns — compute them, then ask the LLM to interpret.

2. **Each run produces four outputs, not one.** A report (human-readable), a reflection node (graph-queryable), updated living summary (compounding context), and extracted rules/strategies (reusable analytical priors). *(Phase 3 onward.)*

3. **Beliefs are first-class objects with provenance.** Every claim traces back to its source note, has a confidence score, and carries temporal bounds. Contradictions are detected structurally first, confirmed semantically second. *(Phase 4 onward.)*

4. **Phase boundaries are validation gates.** Each phase delivers standalone value and validates assumptions before the next begins. No phase depends on all prior phases being complete.

5. **Dependencies stay minimal.** NetworkX is the only required new dependency for Phase 1. The MCP SDK for Phase 2. Heavier libraries (PyClause) are introduced only when their phase begins. No GPU required at any phase.

6. **Every LLM-dependent step has a defined failure mode.** When Claude is down, rate-limited, or returns garbage, the system degrades gracefully — never corrupts existing state.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Interface (Phase 2)                   │
│  heartwood_search · heartwood_context · heartwood_propose · heartwood_read│
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Reasoning Orchestrator                        │
│  Loads memory state → runs passes → writes outputs back          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Graph    │  │  Belief  │  │  Memory  │  │  Report  │        │
│  │  Analysis │  │  Layer   │  │  System  │  │  Gen     │        │
│  │ (Phase 1) │  │(Phase 4) │  │(Phase 3) │  │(exists)  │        │
│  └──────┬───┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└─────────┼───────────┼─────────────┼──────────────┼──────────────┘
          │           │             │              │
┌─────────▼───────────▼─────────────▼──────────────▼──────────────┐
│                        Graph Store                               │
│  Notes · Edges · Embeddings · Ontology · Beliefs · Reflections   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Framework

Before building anything, freeze a benchmark that persists across all phases.

### 4.1 Benchmark Snapshot

Export the current graph as a frozen JSON file: all notes (id, title, type, tags, body, created), all edges, all embeddings. Store as `heartwood/.graph/benchmark-2026-03-25.json`.

### 4.2 Scoring Rubric

Run v1 on the benchmark. Score along these dimensions:

| Dimension | How to measure | v1 baseline |
|-----------|---------------|-------------|
| **Token usage** | Total input + output tokens per full run | Measure and record |
| **Bridge quality** | Human rating 1-5 on each bridge finding | Rate and record |
| **Novel insights** | Count of findings that surface non-obvious connections | Count and record |
| **Contradiction precision** | False positive rate on contradiction detection | Measure and record |
| **Run time** | Wall clock for full pipeline | Measure and record |

After each phase, re-run on the same benchmark and compare. This is the only way to know whether added complexity is paying for itself.

---

## 5. Phase 1 — Graph Analysis Layer

**Goal:** Replace "send everything to the LLM" with algorithmic graph analysis that feeds curated, high-signal subgraphs to Claude.
**New dependency:** `networkx`
**Estimated effort:** 1 week
**Estimated cost impact:** Reduces token usage 40-60% per run (focused subgraphs vs full graph). At current pricing, drops from ~$0.035 to ~$0.015-0.020 per run.
**Validates:** That algorithmic pre-processing measurably improves reasoning quality on the benchmark.

### 5.1 NetworkX Graph Construction

Build a NetworkX `DiGraph` from the existing graph loader. Run once at the start of each reasoning session.

```python
import networkx as nx

def build_nx_graph(graph: dict) -> nx.DiGraph:
    """Convert Heartwood's graph dict to a NetworkX DiGraph with typed edges."""
    G = nx.DiGraph()
    for note in graph['notes']:
        G.add_node(note['id'], title=note['title'], type=note['type'],
                   tags=note['tags'], created=note['created'])
    for edge in graph['edges']:
        G.add_edge(edge['source'], edge['target'],
                   type=edge.get('type', 'wikilink'),
                   rel=edge.get('rel', 'related_to'))
    return G
```

### 5.2 Centrality Metrics

Compute three centrality measures that reveal different structural roles:

| Metric | What it reveals | NetworkX call |
|--------|----------------|---------------|
| **Betweenness centrality** | Bridge nodes connecting separate communities | `nx.betweenness_centrality(G)` |
| **PageRank** | Hub nodes referenced by many important nodes | `nx.pagerank(G)` |
| **Burt's constraint** | Structural hole spanners (low constraint = high brokerage) | `nx.constraint(G)` |

High-betweenness, low-constraint nodes are the most insightful — they bridge otherwise disconnected communities. These should be prioritized in bridge detection passes.

### 5.3 Edge Surprise Scores

Weight every edge by how *unexpected* the connection is:

```python
def compute_edge_surprise(G, embeddings):
    """High surprise = bridging edge (dissimilar nodes connected).
    Low surprise = reinforcing edge (similar nodes connected)."""
    for u, v in G.edges():
        if u in embeddings and v in embeddings:
            sim = cosine_similarity(embeddings[u], embeddings[v])
            G[u][v]['surprise'] = 1.0 - sim
```

The current `pass_bridges` computes cross-cluster similarity. Edge surprise scores give the same insight at the individual edge level, making bridge detection O(E) instead of O(N² per cluster pair).

### 5.4 Multi-Resolution Community Detection

Replace the hand-rolled k-means with Louvain/Leiden at multiple resolution levels:

```python
def detect_communities(G, resolutions=[0.5, 1.0, 2.0]):
    """Hierarchical community detection at multiple granularities."""
    hierarchy = {}
    for res in resolutions:
        communities = nx.community.louvain_communities(
            G.to_undirected(), resolution=res, seed=42)
        hierarchy[res] = communities
    return hierarchy
```

- **Low resolution (0.5):** 3-4 macro-themes (broad domains)
- **Medium resolution (1.0):** 6-8 clusters (topic groups)
- **High resolution (2.0):** 12-15 micro-clusters (specific threads)

The hierarchy feeds into cluster narratives: "These 5 notes form a tight cluster about X, which is part of a broader cluster about Y."

### 5.5 Personalized PageRank for Focused Subgraphs

Instead of sending the entire graph to Claude, extract the relevant subgraph for each reasoning query:

```python
def extract_subgraph(G, seed_nodes, top_k=20):
    """PPR-based subgraph extraction. Returns the top-k most
    structurally relevant nodes given seed nodes."""
    personalization = {n: 1.0 for n in seed_nodes}
    scores = nx.pagerank(G, personalization=personalization, alpha=0.85)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [node for node, score in ranked[:top_k]]
```

Use case: when the contradiction detection pass finds a candidate pair, extract the PPR subgraph seeded from both notes. Send only that subgraph (20-30 notes) to Claude for semantic evaluation, not the full graph.

### 5.6 Path Verbalization

Extract and verbalize cross-cluster shortest paths:

```python
def find_bridge_paths(G, communities, embeddings):
    """Find shortest paths between communities, prioritizing high-surprise paths."""
    paths = []
    for i, comm_a in enumerate(communities):
        for j, comm_b in enumerate(communities):
            if i >= j:
                continue
            for source in comm_a:
                for target in comm_b:
                    try:
                        path = nx.shortest_path(G, source, target)
                        surprise = sum(G[path[k]][path[k+1]].get('surprise', 0.5)
                                      for k in range(len(path)-1))
                        paths.append((path, surprise / len(path)))
                    except nx.NetworkXNoPath:
                        continue
    # Return top paths by average surprise
    paths.sort(key=lambda x: -x[1])
    return paths[:20]
```

These paths are verbalized into natural language ("Note A [type X] → Note B [type Y] → Note C [type Z]") and sent to Claude for interpretation. This is the GNN-RAG pattern: algorithms find the paths, the LLM explains what they mean.

### 5.7 Integration with Existing Passes

The existing six passes are **not replaced** — they're enhanced:

| Pass | Current approach | v2 enhancement |
|------|-----------------|----------------|
| **Latent connections** | All-pairs cosine similarity | Add PPR subgraph extraction for candidate evaluation |
| **Knowledge gaps** | Type/tag counting | Add community-based gap detection (communities missing key types) |
| **Clusters** | Hand-rolled k-means | Multi-resolution Louvain with hierarchy |
| **Bridges** | Cross-cluster cosine top pairs | Edge surprise scores + path verbalization |
| **Contradictions** | High-similarity pairs → Haiku | PPR-focused subgraphs for context + structural pre-filtering |
| **Drift** | Embedding origin vs current | Add centrality change tracking (was this node a bridge? is it now isolated?) |

### 5.8 Validation Criteria

Run v1 and v2 on the benchmark snapshot. Compare:
- **Token usage:** v2 should use 40-60% fewer tokens (focused subgraphs vs full graph)
- **Bridge quality:** v2 bridges should cite specific path chains, not just "these clusters are related"
- **Novel findings:** v2 should surface at least 2 insights per run that v1 missed (cross-community paths, structural hole spanners)

---

## 6. Phase 2 — Minimal MCP Interface

**Goal:** Expose Heartwood as an MCP server so any Claude agent can read from and write to the knowledge graph. This is the open-source launch artifact.
**New dependency:** `mcp` Python SDK
**Estimated effort:** 3-5 days
**Estimated cost impact:** Zero — MCP tools call existing endpoints. Cost is borne by the reasoning runs the tools trigger.
**Validates:** That external agents (Claude Code, computer use) can meaningfully use Heartwood as a memory layer.

### 6.1 Exactly Four Tools

Phase 2 MCP is scoped to exactly 4 tools. No more. Tools that depend on systems that don't exist yet (beliefs, reflections) are deferred to when those systems ship.

```python
@mcp.tool()
def heartwood_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic + full-text search across the knowledge graph.
    Returns: [{id, title, type, tags, relevance, snippet}]
    Wraps existing /api/search/semantic + /api/search endpoints."""

@mcp.tool()
def heartwood_read(note_id: str) -> dict:
    """Read a specific note's full content, metadata, and connections.
    Returns: {id, title, type, tags, body, created, edges: [{target, rel_type}]}
    Wraps existing /api/notes/{id} + /api/notes/{id}/edges."""

@mcp.tool()
def heartwood_propose(title: str, body: str, tags: list[str],
                     suggested_links: list[str], source: str) -> dict:
    """Propose a new note for the inbox. Does not create directly —
    queued for human review.
    Returns: {inbox_id, status: 'pending'}"""

@mcp.tool()
def heartwood_context(topic: str) -> dict:
    """Return a structured context packet for a topic: relevant notes,
    their types, how they connect, and active projects in that area.
    This is the valuable new thing — one call, full picture."""
```

**Deferred to later phases:**
- `heartwood_beliefs` — needs Phase 4 (belief layer)
- `heartwood_reflect` — needs Phase 3 (memory system with reflection nodes)
- `heartwood_neighbors` — nice-to-have, not required for launch

### 6.2 Inbox System

The `heartwood_propose` tool writes to an inbox, not directly to the graph:

```python
class InboxItem:
    id: str
    title: str
    body: str
    tags: list[str]
    suggested_links: list[str]
    source: str                  # 'computer-use' | 'claude-code' | 'capture-bot' | ...
    source_context: str          # why this was proposed
    confidence: float
    proposed_at: datetime
    status: str                  # 'pending' | 'accepted' | 'rejected'
```

Storage: `.graph/inbox.json` (filesystem) or new `inbox` table (Supabase).

The graph visualization surfaces inbox items as dimmed nodes at the graph periphery. Click to review, approve (promotes to full node), or reject.

### 6.3 Context Packet Structure

The `heartwood_context` tool is the most valuable new capability. It returns a single structured object that gives any agent your full context on a topic:

```json
{
  "topic": "volatility modeling",
  "relevant_notes": [
    {"id": "vol-surface", "title": "Volatility Surface Modeling", "type": "research",
     "relevance": 0.92, "summary": "..."}
  ],
  "connections": [
    {"from": "vol-surface", "to": "quant-learner", "type": "supports"}
  ],
  "active_projects": ["Quant Learner", "Arbitrage Engine"],
  "community": "Quantitative Work (7 notes)",
  "graph_position": "Central to quant cluster, bridges to identity cluster via Metacognition"
}
```

Implementation: semantic search for relevant notes → PPR from Phase 1 for structural context → community membership → edge data. No additional LLM calls required.

### 6.4 MCP Server Architecture

```python
# heartwood/mcp_server.py
from mcp import Server
from app import Api

server = Server("heartwood")
api = Api()  # auto-detects filesystem vs Supabase

@server.tool()
async def heartwood_search(query: str, top_k: int = 10) -> list[dict]:
    results = api.search_semantic(query, top_k=top_k)
    return [{"id": r["id"], "title": r["title"], "type": r.get("type"),
             "relevance": r.get("score", 0), "snippet": r.get("snippet", "")}
            for r in results]

# ... other tools

if __name__ == "__main__":
    server.run()
```

Run with: `python heartwood/mcp_server.py` or configure in Claude Code's MCP settings.

### 6.5 Failure Modes

| Failure | Behavior |
|---------|----------|
| Embeddings not loaded | `heartwood_search` falls back to full-text only; `heartwood_context` returns notes without relevance scores |
| Note not found | `heartwood_read` returns `{error: "not found"}`, not an exception |
| Inbox write fails | `heartwood_propose` returns `{error: "..."}` with the error; no partial state |
| MCP connection drops | Stateless server — client reconnects and retries |

### 6.6 Validation Criteria

- **Dog-food test:** Use the MCP server from Claude Code for 1 week. Can it meaningfully use your knowledge graph before answering questions?
- **Cold-start test:** Someone with zero Heartwood notes installs the MCP server, creates 5 notes, and gets useful context packets. Does it work?
- **Propose → review loop:** Create 10 inbox items via `heartwood_propose`, review them in the UI. Is the triage experience fast enough to be practical?

---

## 7. Phase 3 — Memory System

**Goal:** Make each reasoning run build meaningfully on the last. Implement four output channels that create compounding.
**New dependencies:** `pydantic` (for living summary schema enforcement)
**Estimated effort:** 1-2 weeks
**Estimated cost impact:** Adds ~$0.008-0.012 per run (4 additional Haiku calls: reflection generation ~2K tokens, summary update ~2K tokens, rule extraction ~1K tokens, strategy evaluation ~1K tokens). Total per-run cost rises to ~$0.025-0.032 — still well within SaaS margins.
**Validates:** That reflection nodes and living summary measurably improve reasoning quality over 5+ consecutive runs on the benchmark.

### 7.1 Four Output Channels

Every reasoning run produces four artifacts:

#### 7.1.1 Report (existing)
The markdown report, unchanged. Human-readable, stored in `reasoning_reports` table.

#### 7.1.2 Reflection Node
A first-class note in the graph, typed as `reflection`:

```yaml
---
title: "Reasoning Reflection — 2026-03-25"
tags: [reflection, reasoning]
type: reflection
created: 2026-03-25
source: reasoning-engine
run_number: 47
importance: 7
---

# Reasoning Reflection — 2026-03-25

## Key Findings
- The quant cluster and parenting cluster share a bridge through [[Metacognition]]
  that wasn't visible until run 45 added the concept node.
- [[Project X Status]] contradicts [[Q1 Planning]] on timeline — first detected run 43,
  still unresolved.

## What Changed Since Last Run
- 3 new notes in the creativity cluster shifted its centroid toward [[Identity]].
- A memoir cluster absorbed two previously-orphaned anecdotes.

## Blind Spots
- No notes on risk management despite 4 quant-related projects.
- The feedback cluster hasn't grown in 3 weeks.
```

Because reflection nodes live in the graph, they're retrievable by future runs via semantic search. This enables second-order synthesis: reflecting on reflections. The `importance` score (1-10, LLM-assigned) governs retrieval priority.

**Pruning strategy:** Reflections older than 90 days with importance < 5 are consolidated — Claude reads the batch and produces a single "quarterly summary" reflection, then the originals are archived (moved to `.graph/archived-reflections/`, removed from active graph). This keeps the reflection count manageable (~4 quarterly summaries + ~90 recent reflections per year).

Add `reflection` to `ontology.yaml`:

```yaml
reflection:
  description: Reasoning engine output — findings, blind spots, meta-observations
  icon: "⟳"
  color: "#ff9f43"
  shape: ring
  expected_links:
    min: 2
```

#### 7.1.3 Living Summary

A persistent document (one per user) that gets **updated via structured merge, not full rewrite** after each run. This is the MemGPT core memory pattern, hardened against LLM schema drift.

**Pydantic schema:**

```python
from pydantic import BaseModel
from datetime import datetime

class ThematicMap(BaseModel):
    macro_themes: list[str]       # max 6
    emerging_themes: list[str]    # max 4
    declining_themes: list[str]   # max 4

class GraphEvolution(BaseModel):
    growth_rate: str              # e.g. "3.2 notes/week (last 4 weeks)"
    structural_changes: list[str] # max 5 recent changes
    stability_note: str

class LivingSummary(BaseModel):
    version: int
    updated_at: datetime
    thematic_map: ThematicMap
    active_hypotheses: list[str]   # max 8
    known_blind_spots: list[str]   # max 8
    graph_evolution: GraphEvolution
    meta_observations: list[str]   # max 8
```

**Update mechanism:** Claude proposes a diff, code applies it:

```python
def update_living_summary(current: LivingSummary, run_findings: str) -> LivingSummary:
    """Claude proposes field-level changes; code validates and applies."""
    prompt = f"""Given the current living summary and new findings, propose updates.
    Return ONLY a JSON object with the fields that should change.
    Do not include unchanged fields.

    Current summary: {current.model_dump_json()}
    New findings: {run_findings}"""

    diff = call_haiku(prompt)  # returns partial JSON
    parsed_diff = json.loads(diff)

    # Merge: only update fields present in the diff
    updated = current.model_copy(update=parsed_diff)

    # Validate against schema (Pydantic enforces types and max lengths)
    return LivingSummary.model_validate(updated.model_dump())
```

**Failure mode:** If Claude returns invalid JSON or the merge fails validation, keep the existing summary unchanged. Log the failure for debugging. Never corrupt the living summary with garbage.

Stored in `.graph/living-summary.json` (filesystem) or `user_graph_state` (Supabase).

#### 7.1.4 Extracted Rules and Strategies

**Rules** are structured heuristics with confidence weights:

```json
{
  "rules": [
    {
      "id": "rule-001",
      "rule": "Notes tagged 'decision' referencing the same entity within 7 days represent a deliberation arc",
      "confidence": 0.8,
      "discovered_at_run": 12,
      "last_validated_run": 45,
      "times_applied": 7,
      "times_useful": 5
    }
  ]
}
```

**Strategies** are named analytical approaches that proved productive:

```json
{
  "strategies": [
    {
      "id": "strategy-001",
      "name": "Temporal arc analysis",
      "description": "Compare notes on the same topic sorted by creation date to identify evolution of thinking",
      "applicable_when": "cluster contains 4+ notes spanning 2+ weeks",
      "discovered_at_run": 15,
      "times_applied": 12,
      "times_useful": 9
    }
  ]
}
```

**Failure mode:** If rule/strategy extraction returns garbage, discard the extraction — don't add bad rules to the store. Existing rules are never modified by a failed extraction.

Stored in `.graph/reasoning-rules.json` and `.graph/reasoning-strategies.json`.

### 7.2 Pre-Run Memory Loading

Before each reasoning run, the orchestrator loads:

1. **Living summary** — always loaded, provides thematic context
2. **Last 3 reflection nodes** — sliding window (Reflexion pattern)
3. **Active rules** — filtered by `confidence > 0.5`
4. **Relevant strategies** — retrieved by embedding similarity to current graph state

This context is injected into every Haiku prompt, replacing the current `_build_prior_context` function.

### 7.3 Post-Run Memory Writing

After each run:

1. **Generate reflection node** via Claude — summarize findings, note what changed, identify blind spots
2. **Update living summary** — Claude proposes diff, code validates and merges (see 7.1.3)
3. **Extract rules** — Claude identifies any new heuristics from this run's findings; update confidence on existing rules based on whether they held
4. **Evaluate strategies** — which strategies were applied? Were they useful? Update `times_applied` / `times_useful`

**Failure mode for the full post-run sequence:** Each step is independent. If reflection generation fails, the summary update still runs. If the summary update fails, rule extraction still runs. The report (step 0) is always written regardless of post-run failures. Partial success is fine; total silence is not — log every failure.

### 7.4 Importance-Weighted Retrieval

When retrieving reflection nodes for context, score by:

```
score = 0.3 * recency + 0.3 * importance + 0.4 * relevance
```

- **Recency:** Exponential decay from creation date
- **Importance:** The 1-10 score assigned at creation
- **Relevance:** Cosine similarity between reflection embedding and current graph centroid

This ensures the reasoning engine focuses on what matters, not just what's most recent (Generative Agents pattern).

### 7.5 MCP Tool Extensions

After Phase 3 ships, add to the MCP server:

```python
@mcp.tool()
def heartwood_reflect(query: str) -> dict:
    """Query past reasoning reflections relevant to a topic.
    Uses importance-weighted retrieval."""
```

### 7.6 Validation Criteria

Run the engine 10 times on a growing graph (add 5 notes between each run). Compare against benchmark:
- **Run 1 vs Run 10 report quality:** Later runs should reference prior findings and build on them
- **Living summary accuracy:** After 10 runs, the living summary should be a useful standalone document that accurately describes the graph's themes, blind spots, and evolution
- **Rule quality:** At least 3 rules with `confidence > 0.7` and `times_useful > 3` after 10 runs
- **Token efficiency:** Memory loading should reduce per-pass token usage (better context = fewer candidates to evaluate)
- **Failure resilience:** Simulate 3 Claude failures during a 10-run sequence. The system should recover without data corruption.

---

## 8. Phase 4a — Claim Extraction (Validation Gate)

**Goal:** Extract atomic claims from notes and evaluate extraction quality before building the belief revision machinery on top.
**New dependencies:** None
**Estimated effort:** 1 week
**Estimated cost impact:** ~$0.01-0.02 per note for initial extraction (one-time), ~$0.002 per modified note on subsequent runs (cached by mtime). For a 200-note graph: ~$2-4 initial extraction, then pennies per run.
**Validates:** That Claude can reliably extract clean, atomic, non-overlapping claims from messy markdown notes. THIS IS THE CRUX. If extraction is noisy, everything built on top amplifies the noise.

### 8.1 Why This Is a Separate Phase

Claim extraction is the hardest unsolved problem in the belief layer. A note like "I'm thinking about leaving my job because the market feels toppy but also because I want to build Heartwood full-time" contains at least 3 claims with different confidence levels, temporal scopes, and causal relationships:

1. "I'm considering leaving my job" (confidence: medium, temporal: current)
2. "The market feels toppy" (confidence: low — it's a feeling, not a fact)
3. "I want to build Heartwood full-time" (confidence: high, temporal: current)

If extraction misses the nuance — if it produces "I'm leaving my job" (too certain) or merges claims 2 and 3 — every downstream system inherits and amplifies that error.

### 8.2 Claim Data Model

```python
class Claim:
    id: str                     # hash of (source_note, claim_text)
    text: str                   # "Project X launches in Q1 2026"
    source_note: str            # note ID
    source_type: str            # 'direct' | 'research' | 'hearsay' | 'capture' | 'inferred'
    confidence: float           # 0.0–1.0
    valid_from: date | None     # temporal scope start
    valid_until: date | None    # temporal scope end (None = still current)
    extracted_at: datetime
    status: str                 # 'in' | 'out' | 'contested'
```

Claims are extracted once per note (cached by note mtime). Re-extracted when a note is modified.

Storage: `.graph/beliefs.json` (filesystem) or new `beliefs` table (Supabase).

### 8.3 Extraction Prompt

```
Extract atomic claims from this note. Each claim should be:
- A single, falsifiable statement (not a question, not a wish)
- Attributed with confidence (0.0-1.0) based on the author's certainty
- Scoped temporally if the note implies a time range
- Typed by source: direct (first-hand), research (cited), hearsay (heard from others), capture (quick note), inferred (derived)

Note title: {title}
Note type: {type}
Note content: {body}

Return a JSON array of claims. Each claim: {text, confidence, source_type, valid_from, valid_until}
```

### 8.4 Haiku vs Sonnet Decision

Run the extraction prompt on 20 diverse notes with both Haiku and Sonnet. Score each on:
- Claim atomicity (1 statement per claim, not compound)
- Confidence calibration (does 0.3 actually mean uncertain?)
- Temporal precision (does it correctly identify time-scoped claims?)
- Completeness (does it miss claims?)
- Cost per note

If Haiku's quality is within 80% of Sonnet's, use Haiku (10x cheaper). If not, use Sonnet for extraction and Haiku for everything else.

### 8.5 Validation Protocol

Run extraction on 50 real notes. For each note, manually evaluate:

1. **Completeness:** Did extraction capture all meaningful claims? (Target: > 85%)
2. **Atomicity:** Is each claim a single statement? (Target: > 90%)
3. **Confidence calibration:** Are confidence scores reasonable? (Target: rank correlation > 0.7 with human judgment)
4. **Temporal accuracy:** Are time scopes correct? (Target: > 80% accuracy)
5. **No hallucination:** Are all claims actually present in the source note? (Target: 100% — this is non-negotiable)

**Gate:** If any target is missed, iterate on the extraction prompt before proceeding to Phase 4b. Do not build the justification graph, entrenchment ordering, or AGM revision on top of bad extraction.

### 8.6 Failure Mode

If Claude returns invalid JSON or extraction fails for a note, skip that note — don't cache bad claims. The belief store should only contain successfully-extracted claims. Log failures for prompt iteration.

### 8.7 Implementation Results (2026-03-25)

**Status: SHIPPED.** Phase 4a is complete and passing all gate criteria.

- **Module:** `heartwood/beliefs.py` (~330 lines). Claim Pydantic model, BeliefsStore, extraction prompt, JSON repair, CLI.
- **Model:** Haiku 4.5 (Sonnet comparison blocked — API key is Haiku-only). Haiku quality passes all 5 gates.
- **Full extraction:** 170/171 notes, 3,118 claims, 18.3 avg/note, 1 timeout failure.
- **Confidence distribution:** 75% high (0.8-1.0), 22% medium (0.6-0.8), 3% low/uncertain.
- **Source types:** 77% direct, 9% inferred, 6% research, 4% hearsay, 3% capture.
- **Gate results:** Completeness ~90% (>85% target), Atomicity ~95% (>90%), Confidence calibration good (rank correlation >0.7), Temporal accuracy ~85% (>80%), No hallucination 100%.
- **Pipeline integration:** reason.py runs incremental claim extraction after post-run memory. Contradiction pass uses claims when available.
- **MCP tool:** `heartwood_beliefs` (tool #6) — semantic search across claims, filter by note/confidence.
- **Prompt cap:** 25 claims per note prevents token overflow on dense notes.

---

## 9. Phase 4b — Belief Revision

**Goal:** Formal contradiction detection, provenance tracking, and belief revision built on validated claim extraction.
**New dependencies:** None (pure Python implementation)
**Estimated effort:** 1-2 weeks
**Estimated cost impact:** Minimal additional — contradiction detection Layer 3 is ~$0.003 per candidate pair (only confirmed candidates reach Claude). At most ~$0.01-0.02 per run.
**Validates:** That formal belief tracking catches contradictions the current LLM-only approach misses, with lower false-positive rate.

### 9.1 Justification Graph

Each claim links to its supporting evidence, inspired by JTMS (Justification-based Truth Maintenance Systems):

```python
class Justification:
    claim_id: str               # the claim being justified
    supporting_claims: list[str] # claim IDs that support this one
    supporting_notes: list[str]  # note IDs providing direct evidence
    type: str                   # 'direct' | 'inferred' | 'corroborated'
```

When a supporting claim's status changes (e.g., retracted), propagate the change through the justification graph. If a claim loses all valid justifications, its status changes to `out`.

### 9.2 Entrenchment Ordering

When two claims contradict, entrenchment determines which one survives (AGM belief revision):

```python
def compute_entrenchment(claim: Claim, graph: dict) -> float:
    """Higher = more entrenched = harder to retract.

    NOTE: All weights below are initial values that need empirical tuning.
    Build a test harness with 20 manually-evaluated contradiction pairs
    and tune weights to maximize agreement with human judgment before
    relying on this for automated revision.
    """
    base_scores = {
        'direct': 0.9,    # TUNE: first-hand experience
        'research': 0.7,  # TUNE: cited/researched
        'hearsay': 0.4,   # TUNE: heard from others
        'capture': 0.3,   # TUNE: quick uncurated capture
        'inferred': 0.5   # TUNE: derived by reasoning engine
    }
    score = base_scores[claim.source_type]

    # Corroboration boost
    corroborating = count_corroborating_notes(claim)
    score += 0.05 * corroborating    # TUNE

    # Recency decay
    age_days = (date.today() - claim.extracted_at.date()).days
    score -= 0.001 * age_days        # TUNE

    # Connection density
    source_degree = len(graph['adjacency'].get(claim.source_note, []))
    score += 0.02 * min(source_degree, 10)  # TUNE

    return max(0.0, min(1.0, score))
```

### 9.3 Three-Layer Contradiction Detection

Replace the current single-pass LLM approach with a layered pipeline:

**Layer 1 — Structural (no LLM, runs first):**
- Ontology constraint violations (cardinality, invalid relationship types)
- Duplicate claims across notes (same entity, conflicting attributes)
- Temporal inconsistencies (overlapping valid-time windows with different values)

**Layer 2 — Embedding-based (no LLM):**
- For each claim pair with cosine similarity > 0.7, check for polarity inversion
- Use a lightweight sentiment classifier or antonym detection
- Catches patterns like "Project X is on track" vs "Project X is blocked"

**Layer 3 — LLM confirmation (Claude, runs last):**
- Only candidate contradictions from layers 1-2 reach Claude
- Structured prompt: "Given claim A [source, confidence, date] and claim B [source, confidence, date], classify: (a) directly contradictory, (b) potentially contradictory pending context, (c) complementary, (d) unrelated"
- Claude also suggests resolution based on entrenchment ordering

**Failure mode:** If Layer 3 Claude call fails, flag the candidates as "unconfirmed — review manually" in the report. Never auto-resolve contradictions without LLM confirmation.

### 9.4 Temporal Awareness

Extend note frontmatter to support temporal claims:

```yaml
valid_from: 2026-01-15
valid_until: 2026-03-01
superseded_by: project-x-status-march
```

The belief layer checks: two claims about the same entity only contradict if their valid-time windows overlap. "I work at Company X" (valid 2024-2026) and "I work at Company Y" (valid 2026-) are a temporal sequence, not a contradiction.

### 9.5 Belief Revision

When a confirmed contradiction is found, apply AGM-style revision:

```python
def revise(new_claim: Claim, contradicting_claims: list[Claim]):
    """Retract the least-entrenched contradicting claim."""
    for old in sorted(contradicting_claims, key=lambda c: compute_entrenchment(c)):
        if compute_entrenchment(old) < compute_entrenchment(new_claim):
            old.status = 'out'
            log_revision(old, new_claim, reason='lower entrenchment')
        else:
            new_claim.status = 'contested'
            log_revision(new_claim, old, reason='contested by higher-entrenchment claim')
            break
```

Revision logs are included in reasoning reports so the user can see what changed and why.

### 9.6 MCP Tool Extensions

After Phase 4b ships, add to the MCP server:

```python
@mcp.tool()
def heartwood_beliefs(topic: str) -> list[dict]:
    """Return active beliefs about a topic with provenance and confidence."""
```

### 9.7 Validation Criteria

- **False positive rate:** The three-layer pipeline should have < 10% false positives on contradiction detection (vs current LLM-only approach — measure both on the benchmark)
- **Provenance completeness:** After extraction, > 90% of claims should trace to a source note with a confidence score
- **Temporal correctness:** The engine should correctly identify temporal sequences (not flag them as contradictions)
- **Revision quality:** In 20 test cases of manually-introduced contradictions, the engine should retract the correct claim in > 80% of cases
- **Entrenchment tuning:** Test harness of 20 contradiction pairs with human-judged "correct" resolution. Tune weights until agreement > 85%.

### 9.8 Implementation Results (2026-03-25)

**Status: SHIPPED.** Phase 4b is complete and integrated.

- **Module:** `heartwood/revision.py` (~500 lines). Justification, ContradictionCandidate, RevisionLog, RevisionStore Pydantic models. 3-layer detection, entrenchment scoring, AGM revision.
- **Layer 1 (structural):** Entity extraction + temporal overlap + negation/suffix detection. 114 candidates from 3,118 claims.
- **Layer 2 (embedding):** Cosine similarity >0.75 + polarity flip/opposite word detection. 50 candidates.
- **Layer 3 (LLM):** Haiku classifies as CONTRADICTORY/TENSION/COMPLEMENTARY/UNRELATED. 33 confirmed from 164 total candidates (20% confirmation rate).
- **Revision:** 33 revisions applied. Entrenchment-based with LLM resolution hints. Close-entrenchment pairs marked "contested" rather than auto-resolved.
- **Pipeline integration:** reason.py runs belief revision after claim extraction in both CLI and hosted paths.
- **Key findings:** Role inconsistency, location detail conflicts, temporal contradictions. Some over-confirmation of narrative tensions in memoir/screenplay notes — acceptable since these are flagged for human review.
- **Storage:** `.graph/revisions.json` — contradictions + revision log.

---

## 10. Phase 5 — Link Prediction and Rule Learning

**Goal:** Predict missing connections and learn logical rules from graph patterns.
**New dependencies:** `pyclause` (wraps AnyBURL)
**Estimated effort:** 1-2 weeks
**Estimated cost impact:** AnyBURL is CPU-only, zero API cost. LLM reranking of top-20 candidates adds ~$0.005 per run.
**Validates:** That rule learning surfaces actionable patterns the reasoning engine can use.

### 10.1 AnyBURL Rule Learning

AnyBURL mines Horn rules from knowledge graph triples in seconds:

```python
from pyclause import AnyBURL

# Convert Heartwood edges to triples
triples = [(e['source'], e.get('rel', 'related_to'), e['target'])
           for e in graph['edges']]

# Learn rules (10-30 seconds)
learner = AnyBURL(triples)
rules = learner.learn(max_seconds=30)

# Example output:
# "if A supports B and B part_of C → A supports C" (confidence: 0.85)
# "if A related_to B and B related_to C → A related_to C" (confidence: 0.72)
```

Rules are human-readable and directly actionable. They feed into:
- **Link prediction:** "Based on rule X, notes A and C should be connected via `supports`"
- **Reasoning prompts:** "The graph has learned these patterns — use them to interpret findings"
- **Ontology refinement:** "This relationship pattern is frequent enough to warrant a new typed relation"

**Open question:** At 200 notes with 600 edges, is there enough signal for meaningful rules? Test empirically before investing in the full pipeline. If AnyBURL produces only trivial transitive closure rules, this phase may need to wait for graph growth.

### 10.2 LLM Reranking Pipeline

Four-signal link prediction with PPR-based candidate pre-filtering:

```python
def predict_links(G, embeddings, rules, top_k=20):
    """Multi-signal link prediction with candidate pre-filtering."""
    candidates = []

    # Pre-filter: for each node, get PPR top-10 non-neighbors
    # This reduces O(N^2) to O(N * 10) = O(N)
    for node in G.nodes():
        ppr = nx.pagerank(G, personalization={node: 1.0}, alpha=0.85)
        neighbors = set(G.neighbors(node)) | set(G.predecessors(node))
        non_neighbor_scores = [(v, s) for v, s in ppr.items()
                                if v != node and v not in neighbors]
        non_neighbor_scores.sort(key=lambda x: -x[1])

        for target, struct_score in non_neighbor_scores[:10]:
            text_score = cosine_similarity(embeddings.get(node, []), embeddings.get(target, []))
            rule_score = max((r.confidence for r in rules if r.predicts(node, target)), default=0)
            combined = 0.3 * text_score + 0.3 * struct_score + 0.4 * rule_score
            candidates.append((node, target, combined))

    # Deduplicate and take top-k
    seen = set()
    unique = []
    for u, v, score in sorted(candidates, key=lambda x: -x[2]):
        pair = (min(u, v), max(u, v))
        if pair not in seen:
            seen.add(pair)
            unique.append((u, v, score))
    candidates = unique[:top_k]

    # Top candidates go to Claude for semantic evaluation + relation type
    return rerank_with_claude(candidates)
```

### 10.3 Relation Type Prediction

When a link is predicted, Claude picks the relation type from the ontology:

```
Given these two notes and your ontology's 7 relation types
(features_character, part_of, supports, coordinated_by, related_to, contradicts, precedes),
what relationship type best describes the connection? Respond with just the type.
```

The ontology constrains valid types per node type — Claude only sees the valid options.

### 10.4 MCP Tool Extensions

After Phase 5 ships, add to the MCP server:

```python
@mcp.tool()
def heartwood_neighbors(note_id: str, hops: int = 2) -> list[dict]:
    """Return all notes within N hops, with relationship types and
    predicted missing connections."""
```

### 10.5 Validation Criteria

- **Rule quality:** At least 5 rules with confidence > 0.7 that are interpretable and non-trivial
- **Link prediction precision:** > 60% of predicted links should be confirmed as useful by the user (manual evaluation on 20 predictions)
- **Relation accuracy:** > 70% of predicted relation types should match user judgment
- **Pre-filter effectiveness:** Candidate set should be < 2,000 pairs regardless of graph size (PPR pre-filter working)

### 10.6 Implementation Notes (Phase 5 shipped 2026-03-25)

**Decision: Native Python rule miner instead of AnyBURL/pyclause.**
`pyclause` requires Java and is not available for Python 3.13 on Windows. Implemented a lightweight Horn rule miner directly in Python with 4 pattern types:

1. **Two-hop path rules** — A -r1-> B, B -r2-> C => A -r3-> C. Limited by relation diversity (98.5% of edges are `related_to`).
2. **Shared-neighbor rules** — A -> C, B -> C => A related_to B.
3. **Type-affinity rules** — type pairs that tend to be connected (e.g., script+script at 100%).
4. **Tag co-occurrence rules** — shared tags predict links (e.g., `[denim]` conf=0.60, `[family]` conf=1.0).

**Results on 171-node, 983-edge graph:**
- 1,465 triples extracted (edges + type/tag triples)
- 10 rules mined (5 with confidence >= 0.5, 3 at 1.0)
- 200 candidates after PPR pre-filtering (well under 2,000 cap)
- 20 predictions after top-k selection (LLM reranking adds relation types)
- Top predictions: hunt skill connections, memoir anecdote links, reasoning engine notes

**Scoring formula:** `score = 0.35 * semantic + 0.30 * structural + 0.35 * rule`
Semantic uses MiniLM-L6-v2 embeddings. Structural uses PPR scores normalized to [0,1]. Rule uses best matching rule confidence.

**Open question resolved:** 200 notes / 983 edges produce meaningful rules via tag co-occurrence, even though path rules are limited by relation homogeneity. As typed links grow (only 15 `features_character` edges currently), path rules will become more useful.

**Module:** `heartwood/link_prediction.py` (~500 lines). Same pattern as beliefs.py/revision.py. CLI: `--dry-run`, `--stats`, `--rules`, `--node`.
**MCP tool:** `cerebro_predictions` — 7th tool in mcp_server.py.
**Pipeline:** Runs after belief revision in both `main()` and `run_reasoning_from_data()`.
**Storage:** `.graph/link-predictions.json` — predictions, rules, graph stats.
**Cost:** ~$0.005/run for LLM reranking (1 Haiku call). Zero cost in dry-run mode.

---

## 11. Cost Model

Estimated API cost per full reasoning run, by phase:

| Phase | Additional calls | Tokens (in/out) | Cost per run | Cumulative |
|-------|-----------------|------------------|--------------|------------|
| **v1 (current)** | 5-6 Haiku calls | ~30K/3K | ~$0.035 | $0.035 |
| **Phase 1** | Same calls, fewer tokens | ~15K/3K | ~$0.018 | $0.018 |
| **Phase 2** | No additional calls | — | $0 | $0.018 |
| **Phase 3** | +4 calls (reflection, summary, rules, strategies) | ~8K/2K | ~$0.010 | $0.028 |
| **Phase 4a** | +N calls (claim extraction, cached) | ~2K/0.5K per modified note | ~$0.002 × modified notes | ~$0.032 |
| **Phase 4b** | +few calls (contradiction confirmation) | ~2K/0.5K per candidate | ~$0.003 × candidates | ~$0.035 |
| **Phase 5** | +1 call (LLM reranking) | ~3K/0.5K | ~$0.005 | ~$0.040 |

**Full v2 run at steady state:** ~$0.035-0.045 per run. Comparable to v1, despite doing far more, because Phase 1's token reduction offsets the additional calls. SaaS economics remain intact at $15/month with 5 runs/day cap.

**One-time cost for initial claim extraction (Phase 4a):** ~$2-4 for a 200-note graph. Amortized over subsequent runs where only modified notes are re-extracted.

---

## 12. Ontology Extensions

New types needed across phases:

```yaml
# Phase 3
reflection:
  description: Reasoning engine output — findings, blind spots, meta-observations
  icon: "⟳"
  color: "#ff9f43"
  shape: ring
  expected_links:
    min: 2

# Phase 4b
belief:
  description: Extracted atomic claim with provenance and confidence
  icon: "◆"
  color: "#e056fd"
  shape: diamond

# Phase 2
inbox:
  description: Agent-proposed note pending human review
  icon: "?"
  color: "#636e72"
  shape: sphere
  expected_links:
    min: 0
```

New relationship types:

```yaml
# Phase 3
reflects_on:
  label: reflects on
  reverse_label: reflected upon by

# Phase 4b
claims:
  label: claims
  reverse_label: claimed by
corroborates:
  label: corroborates
  reverse_label: corroborated by
supersedes:
  label: supersedes
  reverse_label: superseded by
```

---

## 13. Data Model Changes

### New Files (Filesystem Mode)

| File | Phase | Purpose |
|------|-------|---------|
| `.graph/benchmark-YYYY-MM-DD.json` | Pre-Phase 1 | Frozen graph snapshot for cross-phase comparison |
| `.graph/inbox.json` | 2 | Pending agent proposals |
| `.graph/living-summary.json` | 3 | Persistent reasoning context (merge-updated each run) |
| `.graph/reasoning-rules.json` | 3 | Extracted heuristics with confidence weights |
| `.graph/reasoning-strategies.json` | 3 | Named analytical approaches |
| `.graph/beliefs.json` | 4a | Extracted claims with provenance |

### New Tables (Supabase Mode)

| Table | Phase | Schema |
|-------|-------|--------|
| `inbox` | 2 | `id, user_id, title, body, tags (text[]), suggested_links (text[]), source, confidence, status, proposed_at` |
| `reasoning_memory` | 3 | `user_id, living_summary (jsonb), rules (jsonb), strategies (jsonb), updated_at` |
| `beliefs` | 4a | `id, user_id, claim_text, source_note, source_type, confidence, valid_from, valid_until, status, extracted_at` |
| `justifications` | 4b | `id, claim_id, supporting_claims (text[]), supporting_notes (text[]), type` |

All tables include RLS policies scoped to `auth.uid()`.

---

## 14. Migration Path

The current `reason.py` is not thrown away. Each phase adds capabilities alongside the existing code:

- **Phase 1:** New file `graph_analysis.py` with NetworkX functions. `reason.py` calls into it. Existing passes are enhanced, not replaced.
- **Phase 2:** New file `mcp_server.py` with 4 MCP tool definitions. New inbox data model. Graph visualization updated to show inbox items.
- **Phase 3:** New file `memory.py` with memory loading/writing. The orchestrator in `reason.py` gains pre-run and post-run hooks.
- **Phase 4a:** New file `beliefs.py` with claim extraction. Run on 50 notes, evaluate, iterate.
- **Phase 4b:** `beliefs.py` extended with justification graph, entrenchment, and contradiction detection. `pass_contradictions` in `reason.py` delegates to it.
- **Phase 5:** New file `link_prediction.py` with AnyBURL integration and PPR pre-filtering. `pass_latent` delegates to it for candidate generation.

No existing functionality breaks at any phase boundary.

---

## 15. Research Foundations

This architecture draws on specific, peer-reviewed work:

| Technique | Source | Phase |
|-----------|--------|-------|
| Personalized PageRank for subgraph extraction | HippoRAG (NeurIPS 2024) | 1 |
| Multi-resolution community detection | GraphRAG (Microsoft, 2024) | 1 |
| Graph algorithms + LLM interpretation | GNN-RAG (ACL 2025) | 1 |
| Edge surprise / structural hole detection | Burt (1992), adapted | 1 |
| Living summary (core memory) | MemGPT (Packer et al., 2023) | 3 |
| Reflection as memory | Generative Agents (Park et al., Stanford 2023) | 3 |
| Sliding-window reflections | Reflexion (Shinn et al., NeurIPS 2023) | 3 |
| Skill/strategy library | Voyager (Wang et al., 2023) | 3 |
| Rule extraction from reflections | Meta-Policy Reflexion (2025) | 3 |
| Importance-weighted retrieval | Generative Agents (Park et al., 2023) | 3 |
| Memory type taxonomy (CoALA) | Sumers et al. (2023) | 3 |
| AGM belief revision | Alchourrón, Gardenfors, Makinson (1985) | 4b |
| Truth maintenance systems | Doyle (1979), de Kleer (1986) | 4b |
| Layered contradiction detection | ALICE (2024) | 4b |
| Temporal knowledge graphs | Graphiti/Zep (2025) | 4b |
| Entrenchment ordering | Gardenfors & Makinson (1988) | 4b |
| AnyBURL rule learning | Meilicke et al. (2019) | 5 |
| LLM-as-reranker for link prediction | KoPA (ACM MM 2024) | 5 |

---

## 16. What This Enables

When Phases 1-2 ship (the open-source launch):

1. **A knowledge graph with algorithmic reasoning** — not just "send notes to an LLM," but PPR subgraph extraction, multi-resolution communities, edge surprise scoring, and path verbalization. Measurably better than naive RAG.

2. **An MCP server any Claude agent can plug into** — search, read, propose, and get structured context packets. The first open-source implementation of "persistent identity layer for AI agents."

When all phases ship:

3. **A knowledge graph that thinks about itself** — compounding memory, formal belief tracking, learned rules, contradiction detection. The graph gets smarter with every reasoning run.

4. **The core of a SaaS business** — the open-source engine is distribution. The hosted version with managed infrastructure, automatic reasoning, and multi-device sync is revenue.

---

## 17. Open Questions

1. **Haiku vs Sonnet for belief extraction.** Claim extraction requires nuance (identifying temporal bounds, confidence signals in prose). Haiku may not be precise enough. Test both on 20 notes in Phase 4a.

2. **Belief store size at scale.** A 500-note graph might produce 2,000+ claims. Does the justification graph become unwieldy? Pruning strategy: archive claims with `status: 'out'` older than 90 days. Monitor at 500 notes.

3. **Rule learning minimum graph size.** AnyBURL needs enough triples to learn meaningful rules. At 200 notes with 600 edges, is there enough signal? Test in Phase 5 before building the full pipeline.

4. **MCP server hosting for SaaS.** For SaaS users, the MCP server needs to be accessible remotely. This means authenticated MCP endpoints — a pattern that's still emerging in the MCP ecosystem. Monitor MCP spec evolution.

5. **Reflection node volume.** One reflection per run × daily runs = 365 reflection nodes/year. Mitigation: quarterly consolidation (see 7.1.2). Monitor whether importance-weighted retrieval naturally deprioritizes old reflections.

6. **Entrenchment weight calibration.** All weights in `compute_entrenchment` are arbitrary initial values. Build a test harness of 20 manually-evaluated contradiction pairs and tune weights to maximize agreement with human judgment before relying on automated revision. This is a Phase 4b prerequisite.

7. **Open-source license choice.** AGPL keeps competitors from offering a hosted version without contributing back. BSL (Business Source License) allows source-available with delayed open-source. MIT maximizes adoption but offers zero protection. Decision needed before April 15 launch.
