"""Heartwood MCP Server — Phase 2+ of Reasoning Engine v2.

Exposes the Heartwood knowledge graph as an MCP server with 7 tools:
  heartwood_search      — semantic + full-text search
  heartwood_read        — read a note's full content and connections
  heartwood_propose     — propose a new note for human review (inbox)
  heartwood_context     — structured context packet via PPR + community detection
  heartwood_reflect     — query past reasoning reflections, rules, strategies
  heartwood_beliefs     — query extracted claims with confidence and temporal scope
  heartwood_predictions — query predicted missing links with multi-signal scores

Run:
    python heartwood/mcp_server.py

Or configure in Claude Code's MCP settings:
    "heartwood": {"command": "python", "args": ["heartwood/mcp_server.py"]}
"""

import os
import sys
import json
import datetime
import uuid

# Ensure heartwood/ is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from app import Api, _cosine_similarity  # _cosine_similarity used in heartwood_reflect
from reason import load_graph
from graph_analysis import GraphAnalysis
from memory import load_living_summary, load_rules, load_strategies, _load_recent_reflections


# ── Globals ────────────────────────────────────────────────────────────

mcp = FastMCP("heartwood")
api = Api()  # filesystem backend (auto-detects)

# Graph + analysis — loaded lazily on first use, cached for the session
_graph = None
_ga = None


def _ensure_graph():
    """Load graph and run structural analysis (lazy, cached)."""
    global _graph, _ga
    if _graph is None:
        _graph = load_graph()
        _ga = GraphAnalysis(_graph)
        _ga.analyze()
    return _graph, _ga


def _reload_graph():
    """Force reload after mutations (e.g. inbox accept)."""
    global _graph, _ga
    _graph = None
    _ga = None


def _inbox_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.graph', 'inbox.json')


def _load_inbox() -> list[dict]:
    path = _inbox_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def _save_inbox(items: list[dict]):
    path = _inbox_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, default=str)


# ── Tool 1: heartwood_search ────────────────────────────────────────────

@mcp.tool()
def heartwood_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic + full-text search across the Heartwood knowledge graph.

    Combines embedding similarity with keyword matching. Returns the most
    relevant notes with titles, types, tags, relevance scores, and snippets.
    Use this to find notes related to a topic before reading them in full.

    Args:
        query: Search query (natural language or keywords)
        top_k: Maximum number of results to return (default 10)
    """
    results = []
    seen_ids = set()

    # Semantic search (embedding similarity)
    try:
        semantic_results = api.semantic_search(query, top_k=top_k)
        for r in semantic_results:
            seen_ids.add(r['id'])
            results.append({
                'id': r['id'],
                'title': r['title'],
                'type': r.get('type', ''),
                'tags': r.get('tags', []),
                'relevance': r.get('score', 0),
                'snippet': r.get('snippet', ''),
            })
    except Exception:
        pass  # embeddings not available — fall through to full-text

    # Full-text search (fills remaining slots)
    if len(results) < top_k:
        text_results = api.search_notes(query)
        for r in text_results:
            if r['id'] not in seen_ids and len(results) < top_k:
                seen_ids.add(r['id'])
                results.append({
                    'id': r['id'],
                    'title': r['title'],
                    'type': '',
                    'tags': r.get('tags', []),
                    'relevance': 0,
                    'snippet': r.get('snippet', ''),
                })

    return results


# ── Tool 2: heartwood_read ──────────────────────────────────────────────

@mcp.tool()
def heartwood_read(note_id: str) -> dict:
    """Read a specific note's full content, metadata, and connections.

    Returns the complete note including its body text, frontmatter metadata,
    and all edges (both wikilink and semantic connections) to other notes.
    Use after heartwood_search to get the full content of a relevant note.

    Args:
        note_id: The note's ID (filename without .md extension)
    """
    note = api.read_note(note_id)
    if 'error' in note:
        return {'error': f'Note "{note_id}" not found'}

    # Get edges for this note from the graph
    edges = []
    try:
        graph, _ = _ensure_graph()
        for edge in graph['edges']:
            if edge['source'] == note_id:
                target = graph['notes_by_id'].get(edge['target'])
                if target:
                    edges.append({
                        'target': edge['target'],
                        'target_title': target['title'],
                        'rel_type': edge.get('rel', 'related_to'),
                        'edge_type': edge.get('type', 'wikilink'),
                    })
            elif edge['target'] == note_id:
                source = graph['notes_by_id'].get(edge['source'])
                if source:
                    edges.append({
                        'target': edge['source'],
                        'target_title': source['title'],
                        'rel_type': edge.get('rel', 'related_to'),
                        'edge_type': edge.get('type', 'wikilink'),
                    })
    except Exception:
        pass  # graph not loadable — return note without edges

    return {
        'id': note['id'],
        'title': note['title'],
        'type': note.get('type', ''),
        'tags': note.get('tags', []),
        'created': note.get('created', ''),
        'body': note.get('body', ''),
        'edges': edges,
    }


# ── Tool 3: heartwood_propose ──────────────────────────────────────────

@mcp.tool()
def heartwood_propose(
    title: str,
    body: str,
    tags: list[str],
    suggested_links: list[str],
    source: str,
) -> dict:
    """Propose a new note for the knowledge graph inbox (queued for human review).

    Does NOT create a note directly — adds it to a review queue. The human
    owner reviews proposals and accepts or rejects them. Use this when you
    encounter information worth preserving in the knowledge graph.

    Args:
        title: Proposed note title
        body: Note body in markdown (use [[wikilinks]] to reference existing notes)
        tags: List of tags for categorization
        suggested_links: List of existing note IDs this note should link to
        source: Who/what is proposing this ('claude-code', 'computer-use', 'capture-bot', etc.)
    """
    inbox = _load_inbox()

    item = {
        'id': str(uuid.uuid4())[:8],
        'title': title,
        'body': body,
        'tags': tags,
        'suggested_links': suggested_links,
        'source': source,
        'proposed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'pending',
    }

    inbox.append(item)

    try:
        _save_inbox(inbox)
        return {'inbox_id': item['id'], 'status': 'pending'}
    except Exception as e:
        return {'error': str(e)}


# ── Tool 4: heartwood_context ──────────────────────────────────────────

@mcp.tool()
def heartwood_context(topic: str) -> dict:
    """Return a structured context packet for a topic — one call, full picture.

    Combines semantic search, Personalized PageRank subgraph extraction,
    community detection, and edge data to give you everything you need to
    understand the user's knowledge and active work around a topic.

    The context packet includes: relevant notes with relevance scores,
    connections between them, community membership, and graph position.

    Args:
        topic: The topic to get context for (natural language)
    """
    graph, ga = _ensure_graph()

    # Step 1: Semantic search to find seed notes
    seed_notes = []
    try:
        semantic_results = api.semantic_search(topic, top_k=5)
        seed_notes = [r['id'] for r in semantic_results if r.get('score', 0) > 0.25]
    except Exception:
        pass

    # Fallback to full-text if semantic search finds nothing
    if not seed_notes:
        text_results = api.search_notes(topic)
        seed_notes = [r['id'] for r in text_results[:5]]

    if not seed_notes:
        return {
            'topic': topic,
            'relevant_notes': [],
            'connections': [],
            'community': None,
            'graph_position': 'No notes found for this topic.',
        }

    # Step 2: PPR subgraph extraction from seeds
    subgraph_context = ga.extract_subgraph_context(seed_notes, top_k=15)

    relevant_notes = []
    relevant_ids = set()
    for item in subgraph_context:
        relevant_ids.add(item['id'])
        relevant_notes.append({
            'id': item['id'],
            'title': item['title'],
            'type': item['type'],
            'tags': item['tags'],
            'relevance': item['relevance'],
            'summary': item['preview'][:300],
            'is_seed': item['is_seed'],
        })

    # Step 3: Connections between relevant notes
    connections = []
    for edge in graph['edges']:
        if edge['source'] in relevant_ids and edge['target'] in relevant_ids:
            source_note = graph['notes_by_id'].get(edge['source'])
            target_note = graph['notes_by_id'].get(edge['target'])
            if source_note and target_note:
                connections.append({
                    'from': edge['source'],
                    'from_title': source_note['title'],
                    'to': edge['target'],
                    'to_title': target_note['title'],
                    'type': edge.get('rel', 'related_to'),
                })

    # Step 4: Community membership of the primary seed
    community_info = None
    primary_seed = seed_notes[0]
    community_map = ga.community_map.get(1.0, {})
    if primary_seed in community_map:
        comm_idx = community_map[primary_seed]
        communities = ga.get_communities(1.0)
        for comm in communities:
            if comm['index'] == comm_idx:
                community_info = {
                    'name': ', '.join(comm['top_tags'][:3]) if comm['top_tags'] else f'Cluster {comm_idx}',
                    'size': comm['size'],
                    'top_tags': comm['top_tags'],
                    'member_titles': comm['titles'][:10],
                }
                break

    # Step 5: Graph position summary
    position_parts = []
    centrality = ga.centrality.get('betweenness', {})
    pagerank = ga.centrality.get('pagerank', {})

    if primary_seed in centrality:
        bet = centrality[primary_seed]
        all_bet = sorted(centrality.values(), reverse=True)
        rank = all_bet.index(bet) + 1 if bet in all_bet else len(all_bet)
        total = len(all_bet)
        if rank <= total * 0.1:
            position_parts.append(f'Top 10% by betweenness centrality (rank {rank}/{total})')
        elif rank <= total * 0.25:
            position_parts.append(f'Top 25% by betweenness centrality (rank {rank}/{total})')

    if primary_seed in pagerank:
        pr = pagerank[primary_seed]
        all_pr = sorted(pagerank.values(), reverse=True)
        rank = all_pr.index(pr) + 1 if pr in all_pr else len(all_pr)
        total = len(all_pr)
        if rank <= total * 0.1:
            position_parts.append(f'Top 10% by PageRank (rank {rank}/{total})')

    if community_info:
        position_parts.append(f'Member of "{community_info["name"]}" community ({community_info["size"]} notes)')

    # Check if it bridges communities
    bridge_nodes = ga.get_bridge_nodes(top_k=20)
    bridge_ids = {nid for nid, _ in bridge_nodes}
    if primary_seed in bridge_ids:
        position_parts.append('Acts as a bridge between communities')

    graph_position = '. '.join(position_parts) if position_parts else 'Peripheral node with few connections.'

    return {
        'topic': topic,
        'relevant_notes': relevant_notes,
        'connections': connections,
        'community': community_info,
        'graph_position': graph_position,
    }


# ── Tool 5: heartwood_reflect ──────────────────────────────────────────

@mcp.tool()
def heartwood_reflect(query: str, top_k: int = 5) -> dict:
    """Query past reasoning reflections, the living summary, active rules, and strategies.

    Uses importance-weighted retrieval for reflections:
        score = 0.3 * recency + 0.3 * importance + 0.4 * relevance

    Returns the memory system's accumulated intelligence — what the reasoning
    engine has learned across all prior runs. Use this to understand meta-level
    patterns, blind spots, and analytical strategies before diving into notes.

    Args:
        query: Topic or question to match against reflections (natural language)
        top_k: Maximum number of reflections to return (default 5)
    """
    import math
    import datetime as dt

    graph, _ = _ensure_graph()
    emb_cache = graph['emb_cache']

    # --- Reflections with importance-weighted scoring ---
    all_reflections = _load_recent_reflections(limit=50)
    scored = []

    # Get query embedding for relevance scoring
    query_emb = emb_cache.embed_query(query) or None

    now = dt.datetime.now()
    for ref in all_reflections:
        # Recency: exponential decay, half-life ~30 days
        try:
            created = dt.datetime.fromisoformat(ref['created'])
            days_old = (now - created).days
        except Exception:
            days_old = 90
        recency = math.exp(-0.023 * days_old)  # ~0.5 at 30 days

        # Importance: normalize 1-10 to 0-1
        importance = ref.get('importance', 5) / 10.0

        # Relevance: cosine similarity if we have embeddings
        relevance = 0.5  # default mid-range
        if query_emb:
            try:
                body = ref['content']
                if '---' in body:
                    chunks = body.split('---', 2)
                    if len(chunks) >= 3:
                        body = chunks[2].strip()
                ref_emb = emb_cache.embed_query(body[:1500])
                if ref_emb:
                    relevance = max(0, _cosine_similarity(query_emb, ref_emb))
            except Exception:
                pass

        score = 0.3 * recency + 0.3 * importance + 0.4 * relevance
        scored.append({**ref, 'score': score, 'recency': recency, 'relevance': relevance})

    scored.sort(key=lambda x: x['score'], reverse=True)
    top_reflections = scored[:top_k]

    # Format reflections for output
    reflections_out = []
    for r in top_reflections:
        body = r['content']
        if '---' in body:
            chunks = body.split('---', 2)
            if len(chunks) >= 3:
                body = chunks[2].strip()
        reflections_out.append({
            'file': r['file'],
            'created': r['created'],
            'importance': r['importance'],
            'score': round(r['score'], 3),
            'body': body[:2000],
        })

    # --- Living summary ---
    summary = load_living_summary()
    summary_out = summary.model_dump(mode='json') if summary.version > 1 else None

    # --- Active rules ---
    rules = load_rules()
    active_rules = [
        {'id': r.id, 'rule': r.rule, 'confidence': r.confidence}
        for r in rules.rules if r.confidence > 0.5
    ]

    # --- Strategies ---
    strategies = load_strategies()
    strategies_out = [
        {'name': s.name, 'description': s.description,
         'useful_ratio': f'{s.times_useful}/{s.times_applied}'}
        for s in strategies.strategies
    ]

    return {
        'query': query,
        'reflections': reflections_out,
        'living_summary': summary_out,
        'active_rules': active_rules,
        'strategies': strategies_out,
    }


# ── Tool 6: heartwood_beliefs ───────────────────────────────────────────

@mcp.tool()
def heartwood_beliefs(query: str = "", note_id: str = "", min_confidence: float = 0.0,
                    top_k: int = 20) -> dict:
    """Query extracted claims/beliefs from the knowledge graph.

    Search by semantic similarity to a query, filter by note, or both.
    Returns atomic claims with confidence scores, source types, and temporal scope.

    Args:
        query: Semantic search across all claims (optional)
        note_id: Filter to claims from a specific note (optional)
        min_confidence: Minimum confidence threshold (0.0-1.0, default 0.0)
        top_k: Maximum claims to return (default 20)
    """
    from beliefs import load_store

    store = load_store()
    if not store.notes:
        return {'error': 'No claims extracted yet. Run: python heartwood/beliefs.py --all'}

    # Collect candidate claims
    candidates = []
    for nid, nb in store.notes.items():
        if note_id and nid != note_id:
            continue
        for claim in nb.claims:
            if claim.confidence >= min_confidence:
                candidates.append(claim)

    if not candidates:
        return {'query': query, 'note_id': note_id, 'claims': [], 'total': 0}

    # Semantic ranking if query provided
    if query:
        graph, _ = _ensure_graph()
        emb_cache = graph['emb_cache']
        query_emb = emb_cache.embed_query(query)

        if query_emb:
            scored = []
            for claim in candidates:
                claim_emb = emb_cache.embed_query(claim.text)
                if claim_emb:
                    sim = max(0, _cosine_similarity(query_emb, claim_emb))
                    scored.append((claim, sim))
            scored.sort(key=lambda x: -x[1])
            claims_out = [
                {
                    'text': c.text,
                    'confidence': c.confidence,
                    'source_note': c.source_note,
                    'source_type': c.source_type,
                    'valid_from': c.valid_from,
                    'valid_until': c.valid_until,
                    'relevance': round(sim, 3),
                }
                for c, sim in scored[:top_k]
            ]
        else:
            claims_out = [_claim_dict(c) for c in candidates[:top_k]]
    else:
        # No query — sort by confidence descending
        candidates.sort(key=lambda c: -c.confidence)
        claims_out = [_claim_dict(c) for c in candidates[:top_k]]

    return {
        'query': query,
        'note_id': note_id,
        'claims': claims_out,
        'total': len(candidates),
    }


def _claim_dict(claim):
    return {
        'text': claim.text,
        'confidence': claim.confidence,
        'source_note': claim.source_note,
        'source_type': claim.source_type,
        'valid_from': claim.valid_from,
        'valid_until': claim.valid_until,
    }


# ── Tool 7: heartwood_predictions ──────────────────────────────────────

@mcp.tool()
def heartwood_predictions(note_id: str = "", top_k: int = 10) -> dict:
    """Query predicted missing links in the knowledge graph.

    Returns link predictions with multi-signal scores (semantic, structural,
    rule-based) and LLM-predicted relation types. Optionally filter to
    predictions involving a specific note.

    Args:
        note_id: Filter to predictions involving this note (optional)
        top_k: Maximum predictions to return (default 10)
    """
    from link_prediction import load_store as load_pred_store

    store = load_pred_store()
    if not store.predictions:
        return {'error': 'No predictions yet. Run: python heartwood/link_prediction.py'}

    predictions = store.predictions
    if note_id:
        predictions = [p for p in predictions
                       if p.source == note_id or p.target == note_id]

    predictions_out = [
        {
            'source': p.source_title,
            'target': p.target_title,
            'score': p.score,
            'semantic': p.semantic_score,
            'structural': p.structural_score,
            'rule': p.rule_score,
            'predicted_relation': p.predicted_relation,
            'explanation': p.explanation,
            'matching_rules': p.matching_rules[:2],
        }
        for p in predictions[:top_k]
    ]

    rules_summary = [
        {'pattern': r.pattern, 'confidence': r.confidence}
        for r in store.rules[:5]
    ]

    return {
        'note_id': note_id,
        'predictions': predictions_out,
        'total': len(predictions),
        'rules_mined': len(store.rules),
        'top_rules': rules_summary,
        'last_run': store.last_run,
        'graph_stats': store.graph_stats,
    }


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
