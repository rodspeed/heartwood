"""Heartwood Link Prediction — Phase 5 of Reasoning Engine v2.

Predicts missing links in the knowledge graph using three signals:
  1. Rule-based — lightweight Horn rule mining from graph triples
  2. Structural — Personalized PageRank for candidate pre-filtering
  3. Semantic — embedding cosine similarity via MiniLM-L6-v2

Top candidates are reranked by Haiku with relation type prediction
constrained to the ontology's 7 relation types.

Usage:
    python heartwood/link_prediction.py                    # predict links (full pipeline)
    python heartwood/link_prediction.py --dry-run          # structural + embedding only, no LLM
    python heartwood/link_prediction.py --stats            # show prediction stats
    python heartwood/link_prediction.py --rules            # show mined rules only
    python heartwood/link_prediction.py --node firm-intel  # predictions for a specific node
"""

import os
import sys
import json
import logging
import datetime
import argparse
from collections import defaultdict, Counter
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import EmbeddingCache, _cosine_similarity

logger = logging.getLogger('cerebro.link_prediction')


# ── Constants ────────────────────────────────────────────────────────────

# Multi-signal scoring weights (sum to 1.0)
WEIGHT_SEMANTIC = 0.35    # embedding cosine similarity
WEIGHT_STRUCTURAL = 0.30  # PPR-based structural proximity
WEIGHT_RULE = 0.35        # rule-based prediction confidence

# Rule mining thresholds
MIN_RULE_SUPPORT = 2       # minimum instances for a rule to be valid
MIN_RULE_CONFIDENCE = 0.3  # minimum confidence (head_count / body_count)

# Candidate generation
PPR_TOP_K = 15             # PPR non-neighbors per node
MAX_CANDIDATES = 200       # cap before LLM reranking
LLM_RERANK_TOP_K = 20     # candidates sent to LLM

# Relation types from ontology
RELATION_TYPES = [
    'features_character', 'part_of', 'supports', 'coordinated_by',
    'related_to', 'contradicts', 'precedes',
]


# ── Pydantic Models ──────────────────────────────────────────────────────

class Triple(BaseModel):
    """A knowledge graph triple: (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str


class MinedRule(BaseModel):
    """A Horn rule mined from the graph."""
    head: str             # predicted relation: "A -rel-> C"
    body: list[str]       # conditions: ["A -rel1-> B", "B -rel2-> C"]
    confidence: float     # head_count / body_count
    support: int          # number of instances satisfying the full rule
    pattern: str          # human-readable rule description


class LinkCandidate(BaseModel):
    """A predicted missing link between two notes."""
    source: str                          # source note ID
    target: str                          # target note ID
    source_title: str = ''
    target_title: str = ''
    score: float = 0.0                   # combined multi-signal score
    semantic_score: float = 0.0          # embedding similarity
    structural_score: float = 0.0        # PPR score
    rule_score: float = 0.0             # best matching rule confidence
    matching_rules: list[str] = Field(default_factory=list)  # rule patterns
    predicted_relation: str = ''         # LLM-predicted relation type
    explanation: str = ''                # LLM explanation
    status: str = 'predicted'            # predicted|accepted|rejected


class PredictionStore(BaseModel):
    """Persistent store for link predictions."""
    predictions: list[LinkCandidate] = Field(default_factory=list)
    rules: list[MinedRule] = Field(default_factory=list)
    last_run: str = ''
    graph_stats: dict = Field(default_factory=dict)


# ── Triple Extraction ────────────────────────────────────────────────────

def extract_triples(graph: dict) -> list[Triple]:
    """Convert graph edges to (subject, predicate, object) triples.

    Extracts from:
    1. Typed wikilinks — [[rel_type::Target]] become typed triples
    2. Plain wikilinks — become 'related_to' triples
    3. Tag-inferred type relations — note types from ontology inference
    """
    triples = []

    for edge in graph['edges']:
        rel = edge.get('rel') or 'related_to'
        triples.append(Triple(
            subject=edge['source'],
            predicate=rel,
            object=edge['target'],
        ))

    # Add type-based triples: (note, has_type, type)
    for note in graph['notes']:
        if note.get('type'):
            triples.append(Triple(
                subject=note['id'],
                predicate='has_type',
                object=f'type:{note["type"]}',
            ))

        # Tag-based triples: (note, has_tag, tag)
        for tag in note.get('tags', []):
            triples.append(Triple(
                subject=note['id'],
                predicate='has_tag',
                object=f'tag:{tag}',
            ))

    return triples


# ── Rule Mining ──────────────────────────────────────────────────────────

def mine_rules(triples: list[Triple], graph: dict) -> list[MinedRule]:
    """Mine Horn rules from graph triples.

    Discovers patterns like:
    - Path rules: A -r1-> B and B -r2-> C => A -r3-> C
    - Type rules: A has_type X and B has_type Y and A -r-> B => common pattern
    - Shared-context rules: A and B both link to C => A related_to B
    """
    rules = []

    # Build adjacency index
    adj = defaultdict(list)       # node -> [(rel, target)]
    rev_adj = defaultdict(list)   # node -> [(rel, source)]
    for t in triples:
        if not t.subject.startswith(('type:', 'tag:')):
            adj[t.subject].append((t.predicate, t.object))
        if not t.object.startswith(('type:', 'tag:')):
            rev_adj[t.object].append((t.predicate, t.subject))

    # Build existing edge set for checking
    existing = set()
    for t in triples:
        if not t.subject.startswith(('type:', 'tag:')) and not t.object.startswith(('type:', 'tag:')):
            existing.add((t.subject, t.object))
            existing.add((t.object, t.subject))  # undirected check

    # ── Pattern 1: Two-hop path rules ──
    # If A -r1-> B and B -r2-> C, does A -r3-> C often hold?
    path_counts = Counter()     # (r1, r2, r3) -> count
    body_counts = Counter()     # (r1, r2) -> count

    for a in list(adj.keys()):
        for r1, b in adj[a]:
            if b.startswith(('type:', 'tag:')):
                continue
            for r2, c in adj.get(b, []):
                if c.startswith(('type:', 'tag:')) or c == a:
                    continue
                body_counts[(r1, r2)] += 1
                # Check if A->C exists with any relation
                for r3, target in adj[a]:
                    if target == c:
                        path_counts[(r1, r2, r3)] += 1

    for (r1, r2, r3), support in path_counts.items():
        body = body_counts[(r1, r2)]
        if body < MIN_RULE_SUPPORT:
            continue
        confidence = support / body
        if confidence >= MIN_RULE_CONFIDENCE:
            rules.append(MinedRule(
                head=r3,
                body=[r1, r2],
                confidence=round(confidence, 3),
                support=support,
                pattern=f'A -{r1}-> B, B -{r2}-> C => A -{r3}-> C (conf={confidence:.2f}, sup={support})',
            ))

    # ── Pattern 2: Shared neighbor rules ──
    # If A and B both link to C, they might be related
    # Count: for each C, pairs of nodes linking to C
    in_neighbors = defaultdict(set)  # node -> set of nodes linking TO it
    for t in triples:
        if not t.subject.startswith(('type:', 'tag:')) and not t.object.startswith(('type:', 'tag:')):
            in_neighbors[t.object].add(t.subject)

    shared_link_count = 0
    shared_link_connected = 0
    for c, sources in in_neighbors.items():
        sources_list = list(sources)
        for i in range(len(sources_list)):
            for j in range(i + 1, len(sources_list)):
                shared_link_count += 1
                if (sources_list[i], sources_list[j]) in existing:
                    shared_link_connected += 1

    if shared_link_count >= MIN_RULE_SUPPORT:
        confidence = shared_link_connected / shared_link_count if shared_link_count > 0 else 0
        if confidence >= MIN_RULE_CONFIDENCE:
            rules.append(MinedRule(
                head='related_to',
                body=['shared_target'],
                confidence=round(confidence, 3),
                support=shared_link_connected,
                pattern=f'A -> C, B -> C => A -related_to-> B (conf={confidence:.2f}, sup={shared_link_connected})',
            ))

    # ── Pattern 3: Type-affinity rules ──
    # Which type pairs tend to be connected?
    notes_by_id = graph.get('notes_by_id', {})
    type_pair_total = Counter()       # (type_a, type_b) -> total possible pairs
    type_pair_connected = Counter()   # (type_a, type_b) -> connected pairs

    # Group nodes by type
    nodes_by_type = defaultdict(list)
    for note in graph['notes']:
        if note.get('type'):
            nodes_by_type[note['type']].append(note['id'])

    type_list = list(nodes_by_type.keys())
    for i, type_a in enumerate(type_list):
        for type_b in type_list[i:]:
            nodes_a = nodes_by_type[type_a]
            nodes_b = nodes_by_type[type_b]
            for a in nodes_a:
                for b in nodes_b:
                    if a == b:
                        continue
                    type_pair_total[(type_a, type_b)] += 1
                    if (a, b) in existing:
                        type_pair_connected[(type_a, type_b)] += 1

    for (ta, tb), connected in type_pair_connected.items():
        total = type_pair_total[(ta, tb)]
        if total < MIN_RULE_SUPPORT:
            continue
        confidence = connected / total
        if confidence >= MIN_RULE_CONFIDENCE:
            rules.append(MinedRule(
                head='related_to',
                body=[f'has_type:{ta}', f'has_type:{tb}'],
                confidence=round(confidence, 3),
                support=connected,
                pattern=f'{ta} + {tb} tend to be connected (conf={confidence:.2f}, sup={connected}/{total})',
            ))

    # ── Pattern 4: Shared-tag co-occurrence rules ──
    # If notes sharing tag T tend to be connected, that tag predicts links
    tag_to_notes = defaultdict(set)
    for note in graph['notes']:
        for tag in note.get('tags', []):
            tag_to_notes[tag].add(note['id'])

    for tag, members in tag_to_notes.items():
        if len(members) < 2:
            continue
        members_list = list(members)
        pairs_total = 0
        pairs_connected = 0
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                pairs_total += 1
                if (members_list[i], members_list[j]) in existing:
                    pairs_connected += 1
        if pairs_total < MIN_RULE_SUPPORT:
            continue
        confidence = pairs_connected / pairs_total
        if confidence >= MIN_RULE_CONFIDENCE:
            rules.append(MinedRule(
                head='related_to',
                body=[f'shared_tag:{tag}'],
                confidence=round(confidence, 3),
                support=pairs_connected,
                pattern=f'shared tag [{tag}] predicts link (conf={confidence:.2f}, sup={pairs_connected}/{pairs_total})',
            ))

    # Sort by confidence descending
    rules.sort(key=lambda r: -r.confidence)
    return rules


# ── Candidate Generation ─────────────────────────────────────────────────

def generate_candidates(graph: dict, ga, emb_cache: EmbeddingCache,
                        rules: list[MinedRule],
                        node_filter: Optional[str] = None) -> list[LinkCandidate]:
    """Generate link candidates using PPR pre-filtering + multi-signal scoring.

    For each node, get PPR top-k non-neighbors, score with all 3 signals.
    """
    import networkx as nx

    G = ga.G
    notes_by_id = graph.get('notes_by_id', {})
    embeddings = graph.get('embeddings', {})

    # Build rule lookup: for a given (source_type, target_type), what rules apply?
    type_rules = defaultdict(list)
    tag_rules = {}  # tag -> rule
    for rule in rules:
        if len(rule.body) == 2 and rule.body[0].startswith('has_type:'):
            ta = rule.body[0].split(':')[1]
            tb = rule.body[1].split(':')[1]
            type_rules[(ta, tb)].append(rule)
            type_rules[(tb, ta)].append(rule)
        elif len(rule.body) == 1 and rule.body[0].startswith('shared_tag:'):
            tag = rule.body[0].split(':', 1)[1]
            tag_rules[tag] = rule

    # Build adjacency for path rule matching
    adj = defaultdict(list)
    for edge in graph['edges']:
        adj[edge['source']].append((edge.get('rel', 'related_to'), edge['target']))

    # Path rules indexed by (r1, r2)
    path_rules = defaultdict(list)
    for rule in rules:
        if len(rule.body) == 2 and not rule.body[0].startswith('has_type:'):
            path_rules[(rule.body[0], rule.body[1])].append(rule)

    nodes = list(G.nodes()) if not node_filter else [node_filter]
    candidates = []
    seen_pairs = set()

    for node in nodes:
        if node not in G:
            continue

        # PPR-based candidate pre-filtering
        try:
            ppr = nx.pagerank(G, personalization={node: 1.0}, alpha=0.85)
        except nx.NetworkXError:
            continue

        neighbors = set(G.neighbors(node)) | set(G.predecessors(node)) | {node}
        non_neighbors = [(v, s) for v, s in ppr.items()
                         if v != node and v not in neighbors]
        non_neighbors.sort(key=lambda x: -x[1])

        for target, struct_score in non_neighbors[:PPR_TOP_K]:
            pair = (min(node, target), max(node, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Signal 1: Semantic similarity
            emb_a = embeddings.get(node)
            emb_b = embeddings.get(target)
            if emb_a and emb_b:
                semantic = _cosine_similarity(emb_a, emb_b)
            else:
                semantic = 0.0

            # Signal 2: Structural (PPR score, normalized)
            structural = min(struct_score * 100, 1.0)  # PPR scores are small

            # Signal 3: Rule matching
            rule_score = 0.0
            matching = []

            # Check type-affinity rules
            note_a = notes_by_id.get(node, {})
            note_b = notes_by_id.get(target, {})
            type_a = note_a.get('type', '')
            type_b = note_b.get('type', '')
            for rule in type_rules.get((type_a, type_b), []):
                if rule.confidence > rule_score:
                    rule_score = rule.confidence
                    matching.append(rule.pattern)

            # Check shared-tag rules
            tags_a = set(note_a.get('tags', []))
            tags_b = set(note_b.get('tags', []))
            shared_tags = tags_a & tags_b
            for tag in shared_tags:
                if tag in tag_rules and tag_rules[tag].confidence > rule_score:
                    rule_score = tag_rules[tag].confidence
                    matching.append(tag_rules[tag].pattern)

            # Check path rules: does a 2-hop path from node to target exist?
            for r1, mid in adj.get(node, []):
                for r2, end in adj.get(mid, []):
                    if end == target:
                        for rule in path_rules.get((r1, r2), []):
                            if rule.confidence > rule_score:
                                rule_score = rule.confidence
                                matching.append(rule.pattern)

            # Combined score
            combined = (WEIGHT_SEMANTIC * semantic +
                        WEIGHT_STRUCTURAL * structural +
                        WEIGHT_RULE * rule_score)

            candidates.append(LinkCandidate(
                source=node,
                target=target,
                source_title=note_a.get('title', node),
                target_title=note_b.get('title', target),
                score=round(combined, 4),
                semantic_score=round(semantic, 4),
                structural_score=round(structural, 4),
                rule_score=round(rule_score, 4),
                matching_rules=matching[:3],  # top 3 rules
            ))

    # Sort by combined score, take top candidates
    candidates.sort(key=lambda c: -c.score)
    return candidates[:MAX_CANDIDATES]


# ── LLM Reranking ────────────────────────────────────────────────────────

RERANK_PROMPT = """You are evaluating predicted missing links in a personal knowledge graph.

For each candidate pair, decide:
1. Should these notes be connected? (YES / MAYBE / NO)
2. If YES or MAYBE, what relationship type fits best?
3. Brief explanation (one sentence).

Relationship types (pick one):
- features_character — an anecdote features a person
- part_of — note is part of a larger collection/project
- supports — research/evidence supports a claim or project
- coordinated_by — skill/process is coordinated by another
- related_to — general thematic connection
- contradicts — notes contain conflicting information
- precedes — temporal/causal ordering

Respond in this exact format, one line per candidate:
CANDIDATE_NUM | YES/MAYBE/NO | relation_type | explanation

Candidates:
{candidates}"""


def rerank_with_llm(candidates: list[LinkCandidate], graph: dict,
                    api_key: str, call_llm=None) -> list[LinkCandidate]:
    """Send top candidates to Haiku for semantic evaluation + relation type prediction."""
    if not candidates:
        return []

    top = candidates[:LLM_RERANK_TOP_K]
    notes_by_id = graph.get('notes_by_id', {})

    # Format candidates for the prompt
    lines = []
    for i, c in enumerate(top, 1):
        note_a = notes_by_id.get(c.source, {})
        note_b = notes_by_id.get(c.target, {})
        preview_a = note_a.get('preview', '')[:200]
        preview_b = note_b.get('preview', '')[:200]
        lines.append(
            f"{i}. [{c.source_title}] (type: {note_a.get('type', '?')}) <-> "
            f"[{c.target_title}] (type: {note_b.get('type', '?')})\n"
            f"   Score: {c.score:.3f} (semantic={c.semantic_score:.2f}, "
            f"structural={c.structural_score:.2f}, rule={c.rule_score:.2f})\n"
            f"   A: {preview_a}\n"
            f"   B: {preview_b}"
        )

    prompt = RERANK_PROMPT.format(candidates='\n'.join(lines))

    try:
        if call_llm:
            response = call_llm(prompt, api_key, max_tokens=2048)
        else:
            response = _call_haiku_standalone(prompt, api_key)
    except Exception as e:
        logger.warning(f'LLM reranking failed: {e}')
        # Return candidates with score-based ordering, no relation types
        return top

    # Parse LLM response
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            continue

        try:
            # Extract candidate number
            num_str = parts[0].strip().rstrip('.')
            # Handle "CANDIDATE_NUM" or just the number
            num = int(''.join(c for c in num_str if c.isdigit()))
            idx = num - 1
            if idx < 0 or idx >= len(top):
                continue

            verdict = parts[1].strip().upper()
            rel_type = parts[2].strip().lower()
            explanation = parts[3].strip()

            if verdict == 'NO':
                top[idx].status = 'rejected'
                top[idx].explanation = explanation
            elif verdict in ('YES', 'MAYBE'):
                top[idx].status = 'predicted'
                if rel_type in RELATION_TYPES:
                    top[idx].predicted_relation = rel_type
                else:
                    top[idx].predicted_relation = 'related_to'
                top[idx].explanation = explanation
                # Boost YES candidates
                if verdict == 'YES':
                    top[idx].score = round(top[idx].score * 1.2, 4)
        except (ValueError, IndexError):
            continue

    # Filter out rejected, re-sort
    accepted = [c for c in top if c.status != 'rejected']
    accepted.sort(key=lambda c: -c.score)
    return accepted


def _call_haiku_standalone(prompt: str, api_key: str) -> str:
    """Standalone Haiku call for CLI usage."""
    import httpx

    response = httpx.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        json={
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 2048,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()['content'][0]['text']


# ── Storage ──────────────────────────────────────────────────────────────

def _store_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '.graph', 'link-predictions.json')


def load_store() -> PredictionStore:
    path = _store_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return PredictionStore.model_validate(data)
    return PredictionStore()


def save_store(store: PredictionStore):
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(store.model_dump(), f, indent=2)


# ── Pipeline Entry Point ─────────────────────────────────────────────────

def run_link_prediction(graph: dict, ga, api_key: str = '',
                        call_llm=None, dry_run: bool = False,
                        node_filter: Optional[str] = None) -> dict:
    """Entry point for reason.py integration.

    Extracts triples, mines rules, generates candidates, optionally reranks with LLM.
    Returns results dict for inclusion in the reasoning pipeline.
    """
    # Step 1: Extract triples
    triples = extract_triples(graph)

    # Step 2: Mine rules
    rules = mine_rules(triples, graph)
    non_trivial = [r for r in rules if r.confidence >= 0.5]

    # Step 3: Generate candidates
    emb_cache = graph.get('emb_cache')
    if not emb_cache:
        default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.graph', 'embeddings.json')
        emb_cache = EmbeddingCache(default_cache)

    candidates = generate_candidates(graph, ga, emb_cache, rules,
                                     node_filter=node_filter)

    # Step 4: LLM reranking (unless dry run or no API key)
    predictions = []
    if not dry_run and api_key and candidates:
        predictions = rerank_with_llm(candidates, graph, api_key, call_llm)
    else:
        predictions = candidates[:LLM_RERANK_TOP_K]

    # Step 5: Save results
    store = PredictionStore(
        predictions=predictions,
        rules=rules,
        last_run=datetime.datetime.now().isoformat(),
        graph_stats={
            'nodes': len(graph['notes']),
            'edges': len(graph['edges']),
            'triples': len(triples),
            'rules_mined': len(rules),
            'rules_confident': len(non_trivial),
            'candidates_generated': len(candidates),
            'predictions_final': len(predictions),
        },
    )
    save_store(store)

    return {
        'triples': len(triples),
        'rules': len(rules),
        'rules_confident': len(non_trivial),
        'candidates': len(candidates),
        'predictions': len(predictions),
        'top_predictions': [
            {
                'source': p.source_title,
                'target': p.target_title,
                'score': p.score,
                'relation': p.predicted_relation,
                'explanation': p.explanation,
            }
            for p in predictions[:5]
        ],
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Heartwood Link Prediction')
    parser.add_argument('--dry-run', action='store_true',
                        help='Structural + embedding only, no LLM reranking')
    parser.add_argument('--stats', action='store_true',
                        help='Show prediction stats from last run')
    parser.add_argument('--rules', action='store_true',
                        help='Show mined rules only')
    parser.add_argument('--node', type=str, default=None,
                        help='Predictions for a specific node')
    args = parser.parse_args()

    if args.stats:
        store = load_store()
        if not store.predictions:
            print('[predict] No predictions yet. Run without --stats first.')
            return

        print(f'[predict] Last run: {store.last_run}')
        print(f'[predict] Graph: {store.graph_stats}')
        print(f'[predict] Rules: {len(store.rules)}')
        print(f'[predict] Predictions: {len(store.predictions)}')
        print('\nTop predictions:')
        for i, p in enumerate(store.predictions[:10], 1):
            rel = f' ({p.predicted_relation})' if p.predicted_relation else ''
            print(f'  {i}. {p.source_title} <-> {p.target_title} '
                  f'[score={p.score:.3f}]{rel}')
            if p.explanation:
                print(f'     {p.explanation}')
        return

    if args.rules:
        # Just mine and show rules, no predictions
        from reason import load_graph
        from graph_analysis import GraphAnalysis

        print('[predict] Loading graph...')
        graph = load_graph()
        triples = extract_triples(graph)
        print(f'[predict] {len(triples)} triples extracted')

        rules = mine_rules(triples, graph)
        print(f'[predict] {len(rules)} rules mined:\n')
        for r in rules:
            print(f'  {r.pattern}')
        return

    # Full pipeline
    from reason import load_graph, _get_api_key, _call_haiku
    from graph_analysis import GraphAnalysis

    print('[predict] Loading graph...')
    graph = load_graph()
    print(f'[predict] {len(graph["notes"])} notes, {len(graph["edges"])} edges')

    print('[predict] Running graph analysis...')
    ga = GraphAnalysis(graph)
    ga.analyze()

    api_key = _get_api_key()
    if not api_key and not args.dry_run:
        print('[predict] No API key — running in dry-run mode')
        args.dry_run = True

    call_llm = _call_haiku if api_key else None

    print('[predict] Mining rules...')
    triples = extract_triples(graph)
    print(f'[predict] {len(triples)} triples')

    rules = mine_rules(triples, graph)
    print(f'[predict] {len(rules)} rules mined')
    for r in rules[:5]:
        print(f'  {r.pattern}')

    print('[predict] Generating candidates...')
    results = run_link_prediction(
        graph, ga, api_key=api_key or '',
        call_llm=call_llm,
        dry_run=args.dry_run,
        node_filter=args.node,
    )

    print(f'\n[predict] Results:')
    print(f'  Triples: {results["triples"]}')
    print(f'  Rules: {results["rules"]} ({results["rules_confident"]} confident)')
    print(f'  Candidates: {results["candidates"]}')
    print(f'  Predictions: {results["predictions"]}')

    if results['top_predictions']:
        print('\nTop predictions:')
        for i, p in enumerate(results['top_predictions'], 1):
            rel = f' ({p["relation"]})' if p.get('relation') else ''
            print(f'  {i}. {p["source"]} <-> {p["target"]} '
                  f'[score={p["score"]:.3f}]{rel}')
            if p.get('explanation'):
                print(f'     {p["explanation"]}')


if __name__ == '__main__':
    main()
