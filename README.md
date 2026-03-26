# Heartwood

A personal knowledge graph with a reasoning engine that thinks. Not a note-taking app — a system that compounds understanding across every run.

Heartwood stores your knowledge as markdown notes linked by wikilinks, then runs a multi-phase reasoning pipeline that finds latent connections, detects contradictions, extracts beliefs, predicts missing links, and builds a compounding memory layer. Each run makes the next one smarter. It also works as a knowledge layer for AI agents — via MCP, any agent can search, read, and propose into your graph.

## Why Heartwood

Most knowledge tools are storage. They help you write things down and find them later. Heartwood does that — but it also **reasons over your graph**:

- **Graph algorithms handle structure** — centrality, community detection, edge surprise, PageRank. These are computed, not hallucinated.
- **An LLM handles meaning** — interpreting clusters, surfacing contradictions, generating synthesis. The LLM sees structural context computed by the graph layer, not raw text.
- **A memory system compounds across runs** — living summary, extracted reasoning rules, strategies, and reflection nodes. Run 47 meaningfully builds on run 12.
- **A belief layer tracks what you know** — atomic claims extracted from every note, with confidence, provenance, and AGM-style revision when contradictions are confirmed.

This separation — graph algorithms for structure, LLMs for semantics — is the core design principle. It's grounded in 2024-2025 research on GNN-RAG, HippoRAG, and SubgraphRAG.

## Self-Hosted vs. Hosted

This repo is the full local application. Everything runs on your machine — your notes stay on disk, embeddings are computed locally (MiniLM-L6-v2 on CPU), and the only external call is to the Anthropic API for the reasoning engine.

A hosted multi-tenant version with cloud sync, Telegram capture, and a Chrome extension is available separately.

## Architecture

```
Notes + Wikilinks + Ontology
          |
    Graph Analysis            NetworkX: centrality, communities, PPR, edge surprise
          |
    Reasoning Engine          6-pass pipeline (latent, gaps, clusters, bridges,
          |                   contradictions, drift, synthesis) via Claude
          |
    Memory System             Living summary + reasoning rules + strategies +
          |                   reflection nodes (MemGPT-inspired compounding)
          |
    Belief Extraction         Atomic claims from every note (Pydantic-validated,
          |                   confidence-scored, source-tracked)
          |
    Belief Revision           3-layer contradiction detection (structural ->
          |                   embedding -> LLM) + AGM-style entrenchment ordering
          |
    Link Prediction           Horn rule mining + PPR candidates + multi-signal
                              scoring + LLM reranking with relation types
```

Each phase produces standalone value. The reasoning engine works without the belief layer; the belief layer works without link prediction. Failures in any phase don't block downstream — the system degrades gracefully.

## Quick Start

```bash
git clone https://github.com/rodspeed/heartwood
cd heartwood
pip install -r requirements.txt
```

Set your Anthropic API key (required for the reasoning engine). Either:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Or configure it in the app (Settings > API Key), which writes to `heartwood/.graph/config.json`.

Launch the desktop app:
```bash
python heartwood/app.py
```

Or run the reasoning engine directly:
```bash
python heartwood/reason.py              # full pipeline (all 6 passes + memory + beliefs + revision + link prediction)
python heartwood/reason.py --quiet      # structural passes only (no API calls)
python heartwood/reason.py --narrate    # full pipeline + natural language briefing
python heartwood/reason.py --pass drift # single pass
python heartwood/reason.py --backup     # zip notes/ + .graph/ to timestamped backup
```

## MCP Server

Heartwood ships with a Model Context Protocol server, so Claude (or any MCP-compatible agent) can query your knowledge graph directly.

This turns Heartwood into long-term memory for your agents. Instead of starting every conversation from scratch, an agent connected to your graph can recall what you know, check what you believe, see what the reasoning engine has found, and propose new knowledge back into the graph. The agent doesn't just read your notes — it reads the structure, the beliefs, the contradictions, and the predictions that the reasoning engine has computed on top of them.

```bash
python heartwood/mcp_server.py
```

Seven tools are exposed:

| Tool | What it does |
|------|-------------|
| `heartwood_search` | Semantic + keyword search across all notes |
| `heartwood_read` | Read a specific note with full content |
| `heartwood_propose` | Propose a new note with wikilinks and type inference |
| `heartwood_context` | Generate a context packet for a topic |
| `heartwood_reflect` | Query reasoning reflections by topic |
| `heartwood_beliefs` | Search extracted beliefs by note, text, or confidence |
| `heartwood_predictions` | Get link predictions for a note |

To connect to Claude Code, add to your MCP config:
```json
{
  "mcpServers": {
    "heartwood": {
      "command": "python",
      "args": ["path/to/heartwood/mcp_server.py"]
    }
  }
}
```

## Reasoning Engine Deep Dive

### 6-Pass Analysis Pipeline

Each run executes six analysis passes, each informed by graph-computed structural context:

1. **Latent Connections** — Finds note pairs with high embedding similarity but no wikilink. Graph analysis adds edge surprise scores to surface non-obvious connections.

2. **Knowledge Gaps** — Identifies underlinked notes, missing hub connections, and type distribution imbalances. Uses betweenness centrality and degree distribution from NetworkX.

3. **Cluster Narratives** — Groups notes into communities via Louvain community detection on the graph structure. Names each cluster and describes what binds its members.

4. **Inter-Cluster Bridges** — Finds thematic bridges between clusters using cross-cluster edge surprise and path verbalization. These are the unexpected connections between seemingly unrelated domains.

5. **Contradiction Detection** — Surfaces conflicting information. When a belief layer exists, uses atomic claims instead of raw text for higher precision.

6. **Drift Alerts** — Detects notes whose content has diverged from their graph neighborhood, suggesting a title/tag/link update or a genuine change in thinking.

### Compounding Memory (MemGPT-inspired)

Every full run produces four memory outputs:

- **Reflection node** — A first-class graph note analyzing what the engine found. Stored in `notes/` with full frontmatter, retrievable by future runs via semantic search.
- **Living summary** — A structured JSON document updated via proposed diffs (not full rewrites). Pydantic-validated. Captures the evolving state of the graph.
- **Reasoning rules** — Extracted analytical patterns (e.g., "notes tagged [family] tend to contradict notes tagged [career]"). Confidence-tracked, validated across runs.
- **Reasoning strategies** — Higher-order approaches that worked (e.g., "checking temporal overlap before semantic similarity reduces false positives"). Self-improving analytical priors.

### Belief Extraction + Revision

**Extraction:** Every note is decomposed into atomic claims — Pydantic-validated, confidence-scored (0-1), source-typed (direct, research, inferred, hearsay, capture). The extraction prompt requests up to 25 claims per note; in practice most notes yield ~20.

**3-Layer Contradiction Detection:**
- **Layer 1 (Structural):** Entity overlap + temporal proximity + negation signals. No LLM cost. For a graph of ~3,100 claims, narrows ~4.8M potential pairs to ~100 candidates.
- **Layer 2 (Embedding):** Cosine similarity > 0.75 with polarity inversion detection. Adds candidates Layer 1 missed.
- **Layer 3 (LLM Confirmation):** Claude classifies each candidate as CONTRADICTORY, TENSION, COMPLEMENTARY, or UNRELATED. Only CONTRADICTORY pairs trigger revision.

**AGM-Style Revision:** Confirmed contradictions are resolved by entrenchment ordering. Each claim's entrenchment score combines source type (direct > research > inferred > hearsay > capture), corroboration from other notes, recency, and connection density of the source note — all multiplied by the claim's confidence. The least-entrenched claim is retracted. Close-entrenchment pairs are marked "contested" for human review rather than auto-resolved. LLM resolution hints override pure entrenchment when available. See `revision.py` for the full formula and tunable weights.

**Reversal:** When the detection pipeline improves, old revisions can be reversed — restoring incorrectly retracted claims with full audit trail.

### Link Prediction

**Triple Extraction:** The knowledge graph is converted to (subject, relation, object) triples from wikilinks and typed relationships defined in the ontology.

**Rule Mining:** A native Python Horn rule miner extracts four pattern types:
1. Two-hop path rules (A -r1-> B, B -r2-> C => A -r3-> C)
2. Shared-neighbor rules
3. Type-affinity rules (type pairs that tend to connect)
4. Tag co-occurrence rules (shared tags predict links)

**Candidate Generation:** Personalized PageRank pre-filters to top-15 non-neighbors per node (~200 candidates total).

**Multi-Signal Scoring:**
```
score = 0.35 * semantic + 0.30 * structural + 0.35 * rule
```
Semantic: MiniLM-L6-v2 cosine similarity. Structural: PPR scores. Rule: best matching rule confidence.

**LLM Reranking:** Top candidates go to Claude for YES/MAYBE/NO classification with relation type prediction from the ontology.

Early testing on 17 predictions showed 15 confirmed by the graph owner.

## Ontology

Heartwood uses a YAML-defined ontology (`ontology.yaml`) for type inference, validation, and relationship semantics:

- **Node types:** person, project, concept, skill, tool, event, place, media, reflection, and more
- **Relation types:** `features_character`, `part_of`, `supports`, `coordinated_by`, `related_to`, `contradicts`, `precedes`
- **Tag-to-type inference:** automatic type assignment based on tags (e.g., `[project]` -> project type)
- **Typed links:** `[[rel_type::Target]]` syntax for semantic relationships

## Project Structure

```
heartwood/
  app.py                  # Core API + desktop app (pywebview + Three.js)
  index.html              # Full UI — 3D graph, landing page, onboarding (zero dependencies)
  reason.py               # Reasoning engine — 6-pass pipeline + memory + beliefs + links
  memory.py               # Compounding memory — reflections, living summary, rules, strategies
  beliefs.py              # Claim extraction — atomic beliefs from every note
  revision.py             # Belief revision — 3-layer contradiction detection + AGM revision
  link_prediction.py      # Link prediction — rule mining + PPR + embeddings + LLM reranking
  graph_analysis.py       # Graph analysis — NetworkX centrality, communities, PPR, edge surprise
  mcp_server.py           # MCP server — 7 tools for agent integration
  cli.py                  # CLI client
  ontology.yaml           # Type system + relation definitions + validation rules
  sync.py                 # File sync for external markdown sources
  storage_fs.py           # Filesystem storage backend
  models.py               # Shared Pydantic models
  context.html            # Context packet generator
  lib/                    # Vendored JS (Three.js, Cytoscape, d3-force)
  notes/                  # Your knowledge graph (markdown + YAML frontmatter + wikilinks)
  .graph/                 # Generated data (embeddings, beliefs, predictions, summaries)
  reports/                # Reasoning engine output
  REASONING-ENGINE-V2.md  # Full architecture document
```

## Design Decisions

A few non-obvious choices and why they were made:

- **No GPU required.** The embedding model (all-MiniLM-L6-v2) runs on CPU. First call takes ~2 seconds, then it's cached. This keeps the install simple and the hardware bar low.

- **Haiku, not Opus.** The reasoning engine uses Claude Haiku for all LLM calls to minimize cost — ~$0.003 per contradiction pair, < $0.10 for a full pipeline run.

- **Each post-run step is independent.** If reflection generation fails, summary update still runs. If summary update fails, rule extraction still runs. A failure in one phase won't crash the next, though later phases may produce less complete results if earlier ones failed. The report is always written with metadata indicating which phases completed. This is architecturally load-bearing.

- **Pydantic everywhere.** Claims, contradictions, revisions, predictions, living summary — all Pydantic v2 models. Validation at the boundary means the LLM can return garbage and the system catches it before state is corrupted.

- **HTML is a single file.** `index.html` inlines all CSS and JS. No build step, no bundler, no node_modules. Open the file and it works. This is deliberate — complexity budget goes to the reasoning engine, not the build system.

## Cost

Heartwood uses Claude Haiku for all LLM calls. Approximate costs per full pipeline run:

| Phase | What it calls | Cost driver | Typical cost |
|-------|--------------|-------------|--------------|
| Reasoning (6 passes) | 4-5 Haiku calls | Fixed per run | ~$0.02 |
| Memory (4 steps) | 4 Haiku calls | Fixed per run | ~$0.01 |
| Claim extraction | 1 call per modified note | Scales with new/changed notes | ~$0.001/note |
| Belief revision | 1 call per candidate pair | Scales with contradiction candidates | ~$0.003/pair |
| Link prediction | 1 reranking call | Fixed (top-20 candidates) | ~$0.005 |

A full run on a 170-note graph with ~3,100 claims costs < $0.10. Incremental runs (few modified notes) cost < $0.05. The `--quiet` flag skips all API calls for structural-only analysis at zero cost.

## When Things Fail

Every phase is independent. Partial failures don't corrupt state or block downstream phases. Every report includes a metadata header listing which phases completed, the model used, and whether the run was full or structural-only — so you can always tell what you're looking at.

Heartwood assumes single-process access to the `notes/` and `.graph/` directories. Don't run the desktop app and MCP server simultaneously against the same data directory.

- **Anthropic API down or rate-limited:** Reasoning engine falls back to `--quiet` mode (structural passes only). Memory, beliefs, revision, and link prediction are skipped. The report is still written with structural results.
- **Embedding model fails to load:** Graph analysis and reasoning still work (they don't need embeddings). Belief revision Layer 2 and link prediction semantic scoring are skipped.
- **LLM returns malformed JSON:** Pydantic validation catches it. The specific step (e.g., living summary update) fails but the next step proceeds. Existing state is never overwritten with invalid data.
- **A note has no wikilinks:** It still gets embedded, typed by tags, and included in claim extraction. It will show up as "underlinked" in the Knowledge Gaps pass.

## What Data Leaves Your Machine

Heartwood is local-first, but the reasoning engine sends note content to the Anthropic API. Here's exactly what goes out:

- **Reasoning passes:** Note titles, tags, and content previews (~500 chars each) are sent to Claude Haiku for interpretation. Full note content is not sent.
- **Claim extraction:** Full note content is sent to extract atomic beliefs. One API call per new or modified note.
- **Belief revision:** Pairs of extracted claims (not full notes) are sent for contradiction classification.
- **Link prediction:** Note titles and relation context are sent for reranking.
- **Embeddings:** Computed locally via MiniLM-L6-v2. Never sent to any API.

The `--quiet` flag runs structural analysis only — zero API calls, nothing leaves your machine.

## Running Tests

```bash
pytest heartwood/tests/
```

## License

MIT. See [LICENSE](LICENSE).
