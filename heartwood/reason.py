"""Heartwood Reasoning Engine — Layer 3 above the ontology.

Reads the full knowledge graph (notes, embeddings, ontology, edges) and produces
an intelligence report: latent connections, knowledge gaps, cluster narratives,
inter-cluster bridges, contradictions, drift alerts, and synthesis opportunities.

Usage:
    python heartwood/reason.py                    # full report (all 6 passes)
    python heartwood/reason.py --pass drift       # single pass
    python heartwood/reason.py --quiet            # structural passes only (no Haiku calls)
    python heartwood/reason.py --apply            # run report + write back links into notes
    python heartwood/reason.py --curate           # review captures (dry run)
    python heartwood/reason.py --curate --apply   # review + apply curation decisions
    python heartwood/reason.py --backup           # zip notes/ + .graph/ to backups/
"""

import os
import sys
import json
import math
import argparse
import datetime
import shutil
from collections import Counter, defaultdict

# Add parent dir so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    EmbeddingCache, load_ontology, parse_frontmatter, filename_to_id,
    extract_wikilinks, extract_typed_wikilinks, infer_type,
    _cosine_similarity, notes_dir,
)

from graph_analysis import GraphAnalysis
from memory import build_memory_context, post_run_memory
from beliefs import extract_all as extract_all_claims, load_store as load_beliefs_store
from revision import run_belief_revision
from link_prediction import run_link_prediction


# ── Helpers ──────────────────────────────────────────────────────────────

def _notes_path():
    return notes_dir()


def _graph_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.graph')


def _reports_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')


def _get_api_key():
    cfg_path = os.path.join(_graph_path(), 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            key = json.load(f).get('anthropic_api_key', '')
            if key:
                return key
    return os.environ.get('ANTHROPIC_API_KEY', '')


def _call_haiku(prompt, api_key, max_tokens=1024):
    """Call Claude Haiku and return the text response."""
    import http.client
    import ssl

    payload = json.dumps({
        'model': 'claude-haiku-4-5-20251001',
        'max_tokens': max_tokens,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode('utf-8')

    conn = http.client.HTTPSConnection(
        'api.anthropic.com', timeout=60,
        context=ssl.create_default_context()
    )
    conn.request('POST', '/v1/messages', body=payload, headers={
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
    })
    resp = conn.getresponse()
    raw = resp.read().decode('utf-8')
    conn.close()

    if resp.status != 200:
        raise RuntimeError(f'Haiku API returned HTTP {resp.status}: {raw[:200]}')

    result = json.loads(raw)
    if result.get('content') and len(result['content']) > 0:
        return result['content'][0]['text']
    raise RuntimeError('Empty Haiku response')


# ── Graph loader ─────────────────────────────────────────────────────────

def load_graph():
    """Load all notes, embeddings, ontology, and edges. Returns a dict with
    everything the passes need."""
    npath = _notes_path()
    gpath = _graph_path()
    ontology = load_ontology()

    notes = []       # list of note dicts
    notes_by_id = {} # id -> note dict
    title_to_id = {} # lowercase title -> id
    all_edges = []   # wikilink + manual edges
    edge_pairs = set()

    for f in os.listdir(npath):
        if not f.endswith('.md'):
            continue
        filepath = os.path.join(npath, f)
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()

        meta, body = parse_frontmatter(content)
        nid = filename_to_id(f)
        title = str(meta.get('title', nid.replace('-', ' ').title()))
        tags = meta.get('tags', [])
        created = str(meta.get('created', ''))
        mtime = os.path.getmtime(filepath)
        node_type = infer_type(meta, ontology)

        # Preview
        preview_lines = [l for l in body.split('\n') if l.strip() and not l.strip().startswith('#')]
        preview = ' '.join(preview_lines)[:1200].strip()

        typed_links = extract_typed_wikilinks(content)
        plain_links = extract_wikilinks(content)

        note = {
            'id': nid,
            'title': title,
            'tags': tags,
            'type': node_type,
            'created': created,
            'mtime': mtime,
            'body': body,
            'preview': preview,
            'typed_links': typed_links,
            'plain_links': plain_links,
            'file': f,
        }
        notes.append(note)
        notes_by_id[nid] = note
        title_to_id[title.lower()] = nid

    # Build wikilink edges
    for note in notes:
        for rel_type, link_title in note['typed_links']:
            target_id = title_to_id.get(link_title.lower())
            if target_id and target_id != note['id']:
                pair = (note['id'], target_id)
                if pair not in edge_pairs:
                    edge_pairs.add(pair)
                    all_edges.append({
                        'source': note['id'], 'target': target_id,
                        'type': 'wikilink', 'rel': rel_type,
                    })

    # Manual connections
    conn_path = os.path.join(gpath, 'connections.json')
    if os.path.exists(conn_path):
        with open(conn_path, 'r', encoding='utf-8') as fh:
            for edge in json.load(fh):
                pair = (edge['source'], edge['target'])
                if pair not in edge_pairs:
                    edge_pairs.add(pair)
                    all_edges.append({**edge, 'type': 'manual'})

    # Embedding cache
    emb_cache = EmbeddingCache(os.path.join(gpath, 'embeddings.json'))
    embeddings = emb_cache.get_embeddings(notes)

    # Build adjacency for shortest-path queries
    adjacency = defaultdict(set)
    for a, b in edge_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return {
        'notes': notes,
        'notes_by_id': notes_by_id,
        'title_to_id': title_to_id,
        'edges': all_edges,
        'edge_pairs': edge_pairs,
        'adjacency': adjacency,
        'embeddings': embeddings,
        'emb_cache': emb_cache,
        'ontology': ontology,
    }


def load_graph_from_data(notes_data, connections, ontology):
    """Build the same graph dict from pre-fetched data (for hosted/Supabase mode).

    notes_data: list of dicts with keys: id, title, tags, type, created, mtime,
                body, raw_content, preview
    connections: list of dicts with keys: source, target
    ontology: dict (loaded ontology)
    """
    notes = []
    notes_by_id = {}
    title_to_id = {}
    all_edges = []
    edge_pairs = set()

    for n in notes_data:
        nid = n['id']
        title = n.get('title', nid.replace('-', ' ').title())
        tags = n.get('tags', [])
        body = n.get('body', '')
        raw_content = n.get('raw_content', '')
        content = raw_content or body

        preview_lines = [l for l in body.split('\n') if l.strip() and not l.strip().startswith('#')]
        preview = n.get('preview') or ' '.join(preview_lines)[:1200].strip()

        typed_links = extract_typed_wikilinks(content)
        plain_links = extract_wikilinks(content)

        note = {
            'id': nid,
            'title': title,
            'tags': tags,
            'type': n.get('type') or infer_type({'type': n.get('type'), 'tags': tags}, ontology),
            'created': n.get('created', ''),
            'mtime': n.get('mtime', 0),
            'body': body,
            'preview': preview,
            'typed_links': typed_links,
            'plain_links': plain_links,
            'file': '',
        }
        notes.append(note)
        notes_by_id[nid] = note
        title_to_id[title.lower()] = nid

    # Build wikilink edges
    for note in notes:
        for rel_type, link_title in note['typed_links']:
            target_id = title_to_id.get(link_title.lower())
            if target_id and target_id != note['id']:
                pair = (note['id'], target_id)
                if pair not in edge_pairs:
                    edge_pairs.add(pair)
                    all_edges.append({
                        'source': note['id'], 'target': target_id,
                        'type': 'wikilink', 'rel': rel_type,
                    })

    # Manual connections
    for edge in connections:
        pair = (edge['source'], edge['target'])
        if pair not in edge_pairs:
            edge_pairs.add(pair)
            all_edges.append({
                'source': edge['source'], 'target': edge['target'],
                'type': 'manual',
            })

    # Embedding cache — use temp dir for hosted mode
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), 'cerebro_reason_cache')
    os.makedirs(cache_dir, exist_ok=True)
    emb_cache = EmbeddingCache(os.path.join(cache_dir, 'embeddings.json'))
    embeddings = emb_cache.get_embeddings(notes)

    # Build adjacency
    adjacency = defaultdict(set)
    for a, b in edge_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return {
        'notes': notes,
        'notes_by_id': notes_by_id,
        'title_to_id': title_to_id,
        'edges': all_edges,
        'edge_pairs': edge_pairs,
        'adjacency': adjacency,
        'embeddings': embeddings,
        'emb_cache': emb_cache,
        'ontology': ontology,
    }


# ── Graph-aware helpers ─────────────────────────────────────────────────

def _bfs_distance(adjacency, start, end, max_depth=5):
    """BFS shortest path between two nodes. Returns hop count or None if
    unreachable within max_depth."""
    if start == end:
        return 0
    visited = {start}
    frontier = [start]
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency.get(node, ()):
                if neighbor == end:
                    return depth
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return None


def _graph_aware_threshold(hops):
    """Return the similarity threshold for a given hop distance.

    Close pairs (≤2 hops) need very high similarity to justify a direct link —
    the path already exists. Distant or disconnected pairs get a lower bar
    because the link would bridge separate regions.

    Returns the minimum cosine similarity required to propose a link.
    """
    if hops is None:
        return 0.30    # disconnected — low bar, bridge link
    if hops <= 2:
        return 0.70    # already close — only link if very similar
    if hops == 3:
        return 0.50    # moderate distance
    return 0.35        # 4+ hops — standard threshold


# ── Prior-Run Context ────────────────────────────────────────────────────

def _build_prior_context(previous_report, changed_note_titles, section_name=None):
    """Build a concise context block from the previous reasoning report.

    Returns a string to prepend to Haiku prompts, or '' if no prior run.
    section_name: e.g. 'Latent Connections', 'Contradictions' — extracts
    just that section from the previous report to keep tokens tight.
    """
    if not previous_report:
        return ''

    parts = []

    # Extract the relevant section from the previous report
    if section_name:
        # Find section by heading (## N. Section Name or ## Section Name)
        import re
        # Match headings like "## 1. Latent Connections" or "## 3b. Inter-Cluster Bridges"
        pattern = rf'^## \d*b?\.?\s*{re.escape(section_name)}'
        lines = previous_report.split('\n')
        capture = False
        section_lines = []
        for line in lines:
            if re.match(pattern, line):
                capture = True
                continue
            elif capture and line.startswith('## '):
                break
            elif capture:
                section_lines.append(line)
        if section_lines:
            # Trim to ~800 chars to control token cost
            section_text = '\n'.join(section_lines).strip()
            if len(section_text) > 800:
                section_text = section_text[:800] + '...'
            parts.append(f'PREVIOUS RUN — {section_name}:\n{section_text}')

    if changed_note_titles:
        if len(changed_note_titles) <= 20:
            parts.append(f'NOTES CHANGED SINCE LAST RUN ({len(changed_note_titles)}): '
                        + ', '.join(changed_note_titles))
        else:
            parts.append(f'NOTES CHANGED SINCE LAST RUN: {len(changed_note_titles)} notes '
                        f'(including: {", ".join(changed_note_titles[:15])}...)')

    if not parts:
        return ''

    return ('\n\n--- CONTEXT FROM PREVIOUS ANALYSIS ---\n'
            + '\n\n'.join(parts)
            + '\n\nIMPORTANT: Focus on what is NEW or CHANGED since the previous analysis. '
            'Do not repeat prior findings unless they have meaningfully evolved.\n'
            '--- END CONTEXT ---\n\n')


# ── Pass 1: Latent Connections ───────────────────────────────────────────

def pass_latent(graph, api_key=None, quiet=False, prior_context='', ga=None):
    """Find high-similarity note pairs with no existing edge.

    Uses graph-aware filtering: pairs that are already close in the graph
    (≤2 hops) need much higher similarity to justify a direct link. Pairs
    that are distant or disconnected get a lower threshold, since those
    links bridge separate regions and add structural information.

    When ga (GraphAnalysis) is provided, uses PPR subgraph extraction to send
    focused context to the LLM instead of raw previews.
    """
    embeddings = graph['embeddings']
    edge_pairs = graph['edge_pairs']
    notes_by_id = graph['notes_by_id']
    adjacency = graph['adjacency']

    ids = list(embeddings.keys())
    candidates = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            # Skip if already connected (either direction)
            if (a, b) in edge_pairs or (b, a) in edge_pairs:
                continue
            # Skip duplicate-title pairs (audit's job, not ours)
            if a in notes_by_id and b in notes_by_id:
                if notes_by_id[a]['title'].lower() == notes_by_id[b]['title'].lower():
                    continue
            sim = _cosine_similarity(embeddings[a], embeddings[b])

            # Graph-aware threshold: adjust based on hop distance
            hops = _bfs_distance(adjacency, a, b)
            threshold = _graph_aware_threshold(hops)

            if sim >= threshold:
                candidates.append((a, b, sim, hops))

    candidates.sort(key=lambda x: -x[2])
    candidates = candidates[:20]

    results = []
    if not candidates:
        return results

    def _hop_label(hops):
        return f'{hops} hops' if hops is not None else 'disconnected'

    if quiet or not api_key:
        # Structural only — no LLM reasoning
        for a, b, sim, hops in candidates:
            na, nb = notes_by_id[a], notes_by_id[b]
            results.append({
                'note_a': na['title'],
                'note_b': nb['title'],
                'similarity': round(sim, 3),
                'hops': hops,
                'reasoning': f'_(structural match, {_hop_label(hops)} — run without --quiet for LLM reasoning)_',
                'action': 'link',
            })
        return results

    # Batch candidates into a single Haiku call for efficiency
    batch_text = []
    for idx, (a, b, sim, hops) in enumerate(candidates[:10]):
        na, nb = notes_by_id[a], notes_by_id[b]

        if ga:
            # PPR subgraph: send focused structural context instead of raw previews
            ctx = ga.extract_subgraph_context([a, b], top_k=8)
            ctx_lines = [f"    - {c['title']} [{c['type'] or 'untyped'}]"
                         + (' (seed)' if c['is_seed'] else '')
                         for c in ctx]
            batch_text.append(
                f"Pair {idx+1} (similarity: {sim:.2f}, {_hop_label(hops)}):\n"
                f"  A: \"{na['title']}\" [{', '.join(na['tags'])}] — {na['preview'][:400]}\n"
                f"  B: \"{nb['title']}\" [{', '.join(nb['tags'])}] — {nb['preview'][:400]}\n"
                f"  Structural neighborhood:\n" + '\n'.join(ctx_lines)
            )
        else:
            batch_text.append(
                f"Pair {idx+1} (similarity: {sim:.2f}, {_hop_label(hops)}):\n"
                f"  A: \"{na['title']}\" [{', '.join(na['tags'])}] — {na['preview'][:1000]}\n"
                f"  B: \"{nb['title']}\" [{', '.join(nb['tags'])}] — {nb['preview'][:1000]}"
            )

    prompt = f"""{prior_context}You are analyzing a knowledge graph. These note pairs are semantically similar but have no explicit link between them. The hop distance tells you how far apart they are in the graph — "disconnected" means no path exists, making bridges especially valuable.

For each pair, decide:
1. Should they be connected? (yes/no)
2. If yes, what relationship type fits? (related_to, supports, part_of, precedes, contradicts)
3. One sentence explaining why.

Respond as a numbered list matching the pair numbers. Format each as:
[number]. [yes/no] | [relationship] | [reason]

{chr(10).join(batch_text)}"""

    try:
        response = _call_haiku(prompt, api_key)
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

        for idx, (a, b, sim, hops) in enumerate(candidates[:10]):
            na, nb = notes_by_id[a], notes_by_id[b]
            reasoning = lines[idx] if idx < len(lines) else 'No response'
            should_link = 'yes' in reasoning.lower().split('|')[0] if '|' in reasoning else 'yes' in reasoning.lower()[:20]
            results.append({
                'note_a': na['title'],
                'note_b': nb['title'],
                'similarity': round(sim, 3),
                'hops': hops,
                'reasoning': reasoning,
                'action': 'link' if should_link else 'skip',
            })
    except Exception as e:
        print(f'[reason] Haiku error in pass_latent: {e}')
        for a, b, sim, hops in candidates[:10]:
            na, nb = notes_by_id[a], notes_by_id[b]
            results.append({
                'note_a': na['title'],
                'note_b': nb['title'],
                'similarity': round(sim, 3),
                'hops': hops,
                'reasoning': f'_(LLM error: {e})_',
                'action': 'link',
            })

    return results


# ── Pass 2: Knowledge Gaps ───────────────────────────────────────────────

def pass_gaps(graph, ga=None):
    """Analyze type/tag distribution and find structural gaps."""
    notes = graph['notes']
    ontology = graph['ontology']
    title_to_id = graph['title_to_id']

    results = {
        'type_distribution': {},
        'tag_clusters': {},
        'underlinked_notes': [],
        'hub_gaps': [],
        'community_gaps': [],
    }

    # Type distribution
    type_counts = Counter()
    untyped = []
    for note in notes:
        if note['type']:
            type_counts[note['type']] += 1
        else:
            untyped.append(note['title'])
    results['type_distribution'] = dict(type_counts)
    results['untyped'] = untyped

    # Tag cluster analysis
    tag_counts = Counter()
    tag_to_notes = defaultdict(list)
    for note in notes:
        for tag in note['tags']:
            tag_counts[tag] += 1
            tag_to_notes[tag].append(note['title'])
    results['tag_clusters'] = {
        tag: {'count': count, 'notes': tag_to_notes[tag]}
        for tag, count in tag_counts.most_common()
    }

    # Find hub notes and check for missing connections
    hubs = [n for n in notes if n['type'] == 'hub']
    for hub in hubs:
        hub_links = set(l.lower() for l in hub['plain_links'])
        # Find notes that share tags with this hub but aren't linked
        # Exclude meta-tags that don't indicate topical relevance
        meta_tags = {'hub', 'project', 'skills', 'feedback', 'daily', 'capture', 'user', 'people'}
        hub_tags = set(hub['tags']) - meta_tags
        missing = []
        for note in notes:
            if note['id'] == hub['id']:
                continue
            if note['title'].lower() in hub_links:
                continue
            shared_tags = hub_tags & (set(note['tags']) - meta_tags)
            if shared_tags:
                missing.append({
                    'title': note['title'],
                    'shared_tags': list(shared_tags),
                })
        if missing:
            results['hub_gaps'].append({
                'hub': hub['title'],
                'missing_connections': missing[:8],
            })

    # Underlinked notes: notes with connections below ontology minimums
    type_defs = ontology.get('types', {})
    for note in notes:
        if not note['type'] or note['type'] not in type_defs:
            continue
        td = type_defs[note['type']]
        min_links = td.get('expected_links', {}).get('min', 0)
        # Count actual outbound links
        actual_links = len([
            l for l in note['plain_links']
            if title_to_id.get(l.lower()) and title_to_id[l.lower()] != note['id']
        ])
        if actual_links < min_links:
            results['underlinked_notes'].append({
                'title': note['title'],
                'type': note['type'],
                'expected': min_links,
                'actual': actual_links,
            })

    # v2: Community-based gap detection
    if ga:
        results['community_gaps'] = ga.get_community_gaps(resolution=1.0)

    return results


# ── Pass 3: Cluster Narratives ───────────────────────────────────────────

def pass_clusters(graph, api_key=None, quiet=False, prior_context='', ga=None):
    """Cluster notes using Louvain community detection (if ga provided) or k-means fallback."""
    embeddings = graph['embeddings']
    notes_by_id = graph['notes_by_id']

    # v2: Use Louvain communities from graph analysis when available
    if ga:
        ga_communities = ga.get_communities(resolution=1.0)
        if not ga_communities:
            return []

        cluster_summaries = []
        for comm in ga_communities:
            # Find boundary notes: nodes with neighbors in other communities
            boundary = []
            cmap = ga.community_map.get(1.0, {})
            G_undirected = ga.G.to_undirected()
            for nid in comm['member_ids']:
                cn = cmap.get(nid)
                for neighbor in G_undirected.neighbors(nid):
                    if cmap.get(neighbor) != cn:
                        if nid in notes_by_id:
                            boundary.append(notes_by_id[nid]['title'])
                        break

            # Add hierarchy context from macro-level communities
            hierarchy_context = ''
            macro_comms = ga.community_map.get(0.5, {})
            if macro_comms and comm['member_ids']:
                macro_id = macro_comms.get(comm['member_ids'][0])
                if macro_id is not None:
                    macro_members = [nid for nid, mid in macro_comms.items()
                                     if mid == macro_id and nid in notes_by_id]
                    if len(macro_members) > len(comm['member_ids']):
                        hierarchy_context = f"Part of a {len(macro_members)}-note macro-cluster"

            cluster_summaries.append({
                'index': comm['index'],
                'size': comm['size'],
                'titles': comm['titles'],
                'top_tags': comm['top_tags'],
                'boundary_notes': boundary,
                'hierarchy_context': hierarchy_context,
            })
    else:
        # Fallback: hand-rolled k-means (v1 behavior)
        ids = [nid for nid in embeddings if nid in notes_by_id]
        if len(ids) < 4:
            return []

        vectors = [embeddings[nid] for nid in ids]
        k = min(8, max(2, int(math.sqrt(len(ids)))))

        centroids = [vectors[0]]
        for _ in range(1, k):
            dists = []
            for v in vectors:
                min_d = min(_cosine_similarity(v, c) for c in centroids)
                dists.append(1.0 - min_d)
            best_idx = dists.index(max(dists))
            centroids.append(vectors[best_idx])

        assignments = [0] * len(ids)
        for _ in range(15):
            changed = False
            for i, v in enumerate(vectors):
                sims = [_cosine_similarity(v, c) for c in centroids]
                best = sims.index(max(sims))
                if best != assignments[i]:
                    assignments[i] = best
                    changed = True
            if not changed:
                break
            dim = len(vectors[0])
            new_centroids = []
            for ci in range(k):
                members = [vectors[j] for j in range(len(ids)) if assignments[j] == ci]
                if members:
                    centroid = [sum(m[d] for m in members) / len(members) for d in range(dim)]
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(centroids[ci])
            centroids = new_centroids

        clusters = defaultdict(list)
        for i, nid in enumerate(ids):
            clusters[assignments[i]].append(nid)

        cluster_summaries = []
        for ci in sorted(clusters.keys()):
            members = clusters[ci]
            note_titles = [notes_by_id[nid]['title'] for nid in members if nid in notes_by_id]
            note_tags = []
            for nid in members:
                if nid in notes_by_id:
                    note_tags.extend(notes_by_id[nid]['tags'])
            top_tags = [t for t, _ in Counter(note_tags).most_common(5)]

            boundary = []
            for nid in members:
                if nid not in embeddings:
                    continue
                v = embeddings[nid]
                sims = sorted([_cosine_similarity(v, c) for c in centroids], reverse=True)
                if len(sims) >= 2 and (sims[0] - sims[1]) < 0.05:
                    boundary.append(notes_by_id[nid]['title'])

            cluster_summaries.append({
                'index': ci,
                'size': len(members),
                'titles': note_titles,
                'top_tags': top_tags,
                'boundary_notes': boundary,
            })

    if quiet or not api_key or not cluster_summaries:
        for cs in cluster_summaries:
            cs['name'] = f"Cluster {cs['index']+1}"
            cs['narrative'] = f"_{len(cs['titles'])} notes, top tags: {', '.join(cs['top_tags'])}_"
        return cluster_summaries

    # Batch name all clusters in one Haiku call
    batch = []
    for cs in cluster_summaries:
        sample_titles = cs['titles'][:8]
        batch.append(
            f"Cluster {cs['index']+1} ({cs['size']} notes):\n"
            f"  Notes: {', '.join(sample_titles)}\n"
            f"  Common tags: {', '.join(cs['top_tags'])}"
        )

    prompt = f"""{prior_context}You are naming clusters in a personal knowledge graph. Each cluster contains semantically similar notes.

For each cluster, provide:
1. A short name (2-4 words)
2. One sentence describing what binds these notes together

Format each as:
[number]. [Name] — [description]

{chr(10).join(batch)}"""

    try:
        response = _call_haiku(prompt, api_key)
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for idx, cs in enumerate(cluster_summaries):
            if idx < len(lines):
                line = lines[idx]
                # Parse "1. Name — description"
                if '—' in line or '-' in line:
                    sep = '—' if '—' in line else ' - '
                    parts = line.split(sep, 1)
                    # Strip leading number and period
                    name_part = parts[0].strip()
                    for prefix in [f'{idx+1}.', f'{idx+1})', f'Cluster {idx+1}:']:
                        name_part = name_part.replace(prefix, '').strip()
                    cs['name'] = name_part or f'Cluster {idx+1}'
                    cs['narrative'] = parts[1].strip() if len(parts) > 1 else ''
                else:
                    cs['name'] = f'Cluster {idx+1}'
                    cs['narrative'] = line
            else:
                cs['name'] = f'Cluster {idx+1}'
                cs['narrative'] = ''
    except Exception as e:
        print(f'[reason] Haiku error in pass_clusters: {e}')
        for cs in cluster_summaries:
            cs['name'] = f"Cluster {cs['index']+1}"
            cs['narrative'] = f"_{len(cs['titles'])} notes_"

    return cluster_summaries


# ── Pass 3b: Inter-Cluster Bridges ───────────────────────────────────────

def pass_bridges(graph, clusters, api_key=None, quiet=False, prior_context='', ga=None):
    """Find thematic bridges between clusters — what connects seemingly
    unrelated domains. Takes cluster results from pass_clusters.

    v2 (when ga provided): Uses edge surprise scores + path verbalization.
    v1 fallback: cross-cluster cosine similarity.
    """
    if not clusters or len(clusters) < 2:
        return []

    embeddings = graph['embeddings']
    notes_by_id = graph['notes_by_id']
    title_to_id = graph['title_to_id']

    bridges = []

    if ga:
        # v2: Use edge surprise + path verbalization from graph analysis
        cross_pairs = ga.get_cross_community_pairs(resolution=1.0, top_k=30)
        bridge_paths = ga.find_bridge_paths(resolution=1.0, max_paths=15)

        # Group cross-community pairs by community pair
        pair_groups = defaultdict(list)
        for cp in cross_pairs:
            key = (min(cp['community_a'], cp['community_b']),
                   max(cp['community_a'], cp['community_b']))
            pair_groups[key].append(cp)

        # Build bridge entries with path context
        path_by_pair = {}
        for bp in bridge_paths:
            key = (min(bp['community_a'], bp['community_b']),
                   max(bp['community_a'], bp['community_b']))
            if key not in path_by_pair:
                path_by_pair[key] = bp

        # Map cluster index to cluster name
        cluster_name = {}
        for cs in clusters:
            cluster_name[cs['index']] = cs.get('name', f"Cluster {cs['index']+1}")

        for (ci, cj), pairs in pair_groups.items():
            if not pairs:
                continue
            top_pairs = pairs[:3]
            best = top_pairs[0]

            bridge_entry = {
                'cluster_a': cluster_name.get(ci, f'Cluster {ci+1}'),
                'cluster_b': cluster_name.get(cj, f'Cluster {cj+1}'),
                'cluster_a_index': ci,
                'cluster_b_index': cj,
                'top_pairs': [
                    {
                        'note_a': p['title_a'],
                        'note_b': p['title_b'],
                        'similarity': p['similarity'],
                        'preview_a': notes_by_id[p['note_a']]['preview'][:600] if p['note_a'] in notes_by_id else '',
                        'preview_b': notes_by_id[p['note_b']]['preview'][:600] if p['note_b'] in notes_by_id else '',
                    }
                    for p in top_pairs
                ],
                'best_similarity': best['similarity'],
                'bridge_theme': '',
            }

            # Add verbalized path if available
            bp = path_by_pair.get((ci, cj))
            if bp:
                bridge_entry['bridge_path'] = bp['chain_text']
                bridge_entry['path_surprise'] = bp['avg_surprise']

            bridges.append(bridge_entry)

        bridges.sort(key=lambda x: -x['best_similarity'])
        bridges = bridges[:10]

    else:
        # v1 fallback: O(N² per cluster pair) cosine similarity
        cluster_ids = {}
        for cs in clusters:
            ids = []
            for title in cs['titles']:
                nid = title_to_id.get(title.lower())
                if nid and nid in embeddings:
                    ids.append(nid)
            cluster_ids[cs['index']] = ids

        cluster_indices = sorted(cluster_ids.keys())

        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                ci, cj = cluster_indices[i], cluster_indices[j]
                ids_a = cluster_ids[ci]
                ids_b = cluster_ids[cj]
                if not ids_a or not ids_b:
                    continue

                top_pairs = []
                for a in ids_a:
                    for b in ids_b:
                        if a == b:
                            continue
                        sim = _cosine_similarity(embeddings[a], embeddings[b])
                        top_pairs.append((sim, a, b))
                top_pairs.sort(reverse=True)
                top_pairs = top_pairs[:3]

                if not top_pairs:
                    continue

                best_sim, best_a, best_b = top_pairs[0]
                if best_sim < 0.25:
                    continue

                cs_a = clusters[i]
                cs_b = clusters[j]

                bridges.append({
                    'cluster_a': cs_a.get('name', f'Cluster {ci+1}'),
                    'cluster_b': cs_b.get('name', f'Cluster {cj+1}'),
                    'cluster_a_index': ci,
                    'cluster_b_index': cj,
                    'top_pairs': [
                        {
                            'note_a': notes_by_id[a]['title'],
                            'note_b': notes_by_id[b]['title'],
                            'similarity': round(s, 3),
                            'preview_a': notes_by_id[a]['preview'][:600],
                            'preview_b': notes_by_id[b]['preview'][:600],
                        }
                        for s, a, b in top_pairs
                    ],
                    'best_similarity': round(best_sim, 3),
                    'bridge_theme': '',
                })

        bridges.sort(key=lambda x: -x['best_similarity'])
        bridges = bridges[:10]

    if quiet or not api_key or not bridges:
        for b in bridges:
            pairs_str = '; '.join(
                f"{p['note_a']} ↔ {p['note_b']} ({p['similarity']})"
                for p in b['top_pairs']
            )
            b['bridge_theme'] = f"_{pairs_str}_"
        return bridges

    # Batch Haiku call — send previews of the best pair per bridge
    batch = []
    for idx, b in enumerate(bridges):
        best = b['top_pairs'][0]
        entry = (
            f"Bridge {idx+1}: {b['cluster_a']} ↔ {b['cluster_b']}\n"
            f"  Closest pair: \"{best['note_a']}\" ↔ \"{best['note_b']}\" (sim: {best['similarity']})\n"
            f"  Note A preview: {best['preview_a'][:400]}\n"
            f"  Note B preview: {best['preview_b'][:400]}"
        )
        # v2: include verbalized path chain if available
        if b.get('bridge_path'):
            entry += f"\n  Structural path: {b['bridge_path']} (avg surprise: {b.get('path_surprise', '?')})"
        batch.append(entry)

    prompt = f"""{prior_context}You are analyzing a personal knowledge graph to find thematic bridges between clusters of notes that appear unrelated on the surface.

For each bridge below, two clusters are connected by semantically similar notes. Your job: name the deeper theme that connects these two domains. Go beyond surface similarity — look for shared patterns, tensions, or drives.

For each bridge, respond with:
[number]. **[Theme name]** (2-5 words) — [1-2 sentences explaining what connects these domains at a deeper level. Be specific and surprising, not generic.]

{chr(10).join(batch)}"""

    try:
        response = _call_haiku(prompt, api_key, max_tokens=2000)
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for idx, b in enumerate(bridges):
            if idx < len(lines):
                b['bridge_theme'] = lines[idx]
            else:
                best = b['top_pairs'][0]
                b['bridge_theme'] = f"{best['note_a']} ↔ {best['note_b']}"
    except Exception as e:
        print(f'[reason] Haiku error in pass_bridges: {e}')
        for b in bridges:
            best = b['top_pairs'][0]
            b['bridge_theme'] = f"_{best['note_a']} ↔ {best['note_b']}_"

    return bridges


# ── Pass 4: Contradiction Detection ──────────────────────────────────────

def pass_contradictions(graph, api_key=None, quiet=False, prior_context='', ga=None):
    """Find note pairs that might make conflicting claims."""
    if quiet or not api_key:
        return []  # This pass is LLM-only

    embeddings = graph['embeddings']
    notes_by_id = graph['notes_by_id']
    ids = list(embeddings.keys())

    # Pre-filter: high similarity pairs (>0.5) — contradictions hide in similar notes
    candidates = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a not in notes_by_id or b not in notes_by_id:
                continue
            sim = _cosine_similarity(embeddings[a], embeddings[b])
            if sim >= 0.50:
                candidates.append((a, b, sim))

    candidates.sort(key=lambda x: -x[2])
    candidates = candidates[:15]

    if not candidates:
        return []

    # Load belief store for claim-level comparison when available
    beliefs = None
    try:
        beliefs = load_beliefs_store()
        if not beliefs.notes:
            beliefs = None
    except Exception:
        pass

    # Batch check for contradictions
    batch = []
    for idx, (a, b, sim) in enumerate(candidates):
        na, nb = notes_by_id[a], notes_by_id[b]

        # Use extracted claims when available (Phase 4a) — more precise than raw text
        claims_a = claims_b = None
        if beliefs:
            nb_a = beliefs.notes.get(a)
            nb_b = beliefs.notes.get(b)
            if nb_a and nb_a.claims:
                claims_a = '\n'.join(f"    - [{c.confidence:.1f}] {c.text}" for c in nb_a.claims)
            if nb_b and nb_b.claims:
                claims_b = '\n'.join(f"    - [{c.confidence:.1f}] {c.text}" for c in nb_b.claims)

        # Format: prefer claims, fall back to raw preview
        text_a = f"  Claims:\n{claims_a}" if claims_a else f"  Text: {na['preview'][:600]}"
        text_b = f"  Claims:\n{claims_b}" if claims_b else f"  Text: {nb['preview'][:600]}"

        if ga:
            ctx = ga.extract_subgraph_context([a, b], top_k=10)
            ctx_lines = [f"    - {c['title']} [{c['type'] or 'untyped'}]"
                         for c in ctx if not c['is_seed']]
            batch.append(
                f"Pair {idx+1}:\n"
                f"  A: \"{na['title']}\"\n{text_a}\n"
                f"  B: \"{nb['title']}\"\n{text_b}\n"
                f"  Related context:\n" + '\n'.join(ctx_lines[:6])
            )
        else:
            batch.append(
                f"Pair {idx+1}:\n"
                f"  A: \"{na['title']}\"\n{text_a}\n"
                f"  B: \"{nb['title']}\"\n{text_b}"
            )

    prompt = f"""{prior_context}You are checking a knowledge graph for contradictions. These note pairs are semantically similar — check if any make CONFLICTING claims (factual disagreements, contradictory assertions, incompatible statements).

Most pairs will NOT contradict. Only flag genuine contradictions, not mere differences in focus.

For each pair, respond:
[number]. [NONE/CONTRADICTION] | [brief explanation if contradiction found]

{chr(10).join(batch)}"""

    results = []
    try:
        response = _call_haiku(prompt, api_key)
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for idx, (a, b, sim) in enumerate(candidates):
            na, nb = notes_by_id[a], notes_by_id[b]
            if idx < len(lines) and 'CONTRADICTION' in lines[idx].upper():
                explanation = lines[idx].split('|', 1)[1].strip() if '|' in lines[idx] else lines[idx]
                results.append({
                    'note_a': na['title'],
                    'note_b': nb['title'],
                    'explanation': explanation,
                    'action': 'investigate',
                })
    except Exception as e:
        print(f'[reason] Haiku error in pass_contradictions: {e}')

    return results


# ── Pass 5: Drift Alerts ────────────────────────────────────────────────

def pass_drift(graph, ga=None):
    """Find notes that have drifted significantly from their original scope."""
    emb_cache = graph['emb_cache']
    notes_by_id = graph['notes_by_id']

    # Build centrality role lookup from graph analysis
    centrality_roles = {}
    if ga:
        for entry in ga.get_centrality_summary():
            centrality_roles[entry['id']] = entry

    drift_scores = emb_cache.get_drift_scores()
    threshold = 0.3

    results = []
    for nid, drift in sorted(drift_scores.items(), key=lambda x: -x[1]):
        if drift < threshold:
            break
        if nid not in notes_by_id:
            continue
        note = notes_by_id[nid]

        # v2: enrich with structural role context
        role_info = centrality_roles.get(nid)
        role_str = ''
        if role_info:
            role_str = f" (structural role: {role_info['role']}, betweenness: {role_info['betweenness']})"

        if drift > 0.5:
            suggestion = 'This note has evolved far from its original scope — consider splitting into focused notes'
        else:
            suggestion = 'This note has drifted moderately — review title and tags to ensure they still reflect the content'

        if role_info and role_info['role'] == 'bridge':
            suggestion += '. NOTE: this is a bridge node connecting communities — splitting may disconnect graph regions'

        results.append({
            'title': note['title'],
            'type': note['type'],
            'drift': round(drift, 3),
            'structural_role': role_info['role'] if role_info else None,
            'action': 'split' if drift > 0.5 else 'review',
            'suggestion': suggestion + role_str,
        })

    return results


# ── Pass 6: Synthesis Opportunities ──────────────────────────────────────

def pass_synthesis(graph, api_key=None, quiet=False, prior_context=''):
    """Find tightly-connected note clusters that lack a hub/index note."""
    notes = graph['notes']
    notes_by_id = graph['notes_by_id']
    edge_pairs = graph['edge_pairs']

    # Build adjacency from all edges
    adjacency = defaultdict(set)
    for a, b in edge_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Find existing hubs
    hub_ids = set(n['id'] for n in notes if n['type'] == 'hub')

    # Find dense subgraphs: groups of 4+ notes all connected to each other
    # that don't already contain a hub
    visited = set()
    clusters = []

    for note in notes:
        nid = note['id']
        if nid in visited or nid in hub_ids:
            continue
        # BFS to find connected component (excluding hubs)
        component = set()
        queue = [nid]
        while queue:
            current = queue.pop(0)
            if current in component or current in hub_ids:
                continue
            component.add(current)
            for neighbor in adjacency.get(current, []):
                if neighbor not in component and neighbor not in hub_ids:
                    queue.append(neighbor)

        visited |= component

        if len(component) >= 4:
            # Check density: what fraction of possible edges exist?
            n = len(component)
            max_edges = n * (n - 1) / 2
            actual_edges = 0
            comp_list = list(component)
            for i in range(len(comp_list)):
                for j in range(i + 1, len(comp_list)):
                    if (comp_list[i], comp_list[j]) in edge_pairs or \
                       (comp_list[j], comp_list[i]) in edge_pairs:
                        actual_edges += 1
            density = actual_edges / max_edges if max_edges > 0 else 0

            if density >= 0.3:  # At least 30% connected
                clusters.append({
                    'members': [notes_by_id[nid]['title'] for nid in component if nid in notes_by_id],
                    'size': len(component),
                    'density': round(density, 2),
                    'member_ids': list(component),
                })

    clusters.sort(key=lambda x: -x['density'])
    clusters = clusters[:5]

    if not clusters:
        return []

    if quiet or not api_key:
        for cl in clusters:
            cl['suggested_title'] = f"Hub: {cl['members'][0]} and related"
            cl['outline'] = '_(run without --quiet for LLM-generated outline)_'
            cl['action'] = 'create'
        return clusters

    # Ask Haiku to propose synthesis notes
    batch = []
    for idx, cl in enumerate(clusters):
        batch.append(
            f"Group {idx+1} ({cl['size']} notes, {cl['density']:.0%} connected):\n"
            f"  Notes: {', '.join(cl['members'][:8])}"
        )

    prompt = f"""{prior_context}You are proposing synthesis/hub notes for a personal knowledge graph. Each group below is a cluster of tightly-connected notes that lacks a unifying index note.

For each group, propose:
1. A title for the synthesis note (concise, 2-5 words)
2. A 3-bullet outline of what this note should contain

Format:
[number]. **[Title]**
- [bullet 1]
- [bullet 2]
- [bullet 3]

{chr(10).join(batch)}"""

    try:
        response = _call_haiku(prompt, api_key, max_tokens=1500)
        # Parse response — split by group numbers
        sections = response.strip().split('\n')
        current_idx = -1
        current_lines = []

        parsed = {}
        for line in sections:
            line = line.strip()
            if not line:
                continue
            # Check if this is a new group header
            for idx in range(len(clusters)):
                if line.startswith(f'{idx+1}.') or line.startswith(f'{idx+1})'):
                    if current_idx >= 0:
                        parsed[current_idx] = '\n'.join(current_lines)
                    current_idx = idx
                    current_lines = [line]
                    break
            else:
                current_lines.append(line)
        if current_idx >= 0:
            parsed[current_idx] = '\n'.join(current_lines)

        for idx, cl in enumerate(clusters):
            if idx in parsed:
                text = parsed[idx]
                # Extract title from **bold** markers
                import re
                title_match = re.search(r'\*\*(.+?)\*\*', text)
                cl['suggested_title'] = title_match.group(1) if title_match else f'Hub: {cl["members"][0]}'
                cl['outline'] = text
            else:
                cl['suggested_title'] = f'Hub: {cl["members"][0]}'
                cl['outline'] = '_(no LLM response for this group)_'
            cl['action'] = 'create'

    except Exception as e:
        print(f'[reason] Haiku error in pass_synthesis: {e}')
        for cl in clusters:
            cl['suggested_title'] = f'Hub: {cl["members"][0]}'
            cl['outline'] = f'_(LLM error: {e})_'
            cl['action'] = 'create'

    return clusters


# ── Report generator ─────────────────────────────────────────────────────

def generate_report(latent, gaps, clusters, bridges, contradictions, drift, synthesis,
                     is_follow_up=False, changed_note_titles=None):
    """Assemble all pass results into a markdown report."""
    today = datetime.date.today().isoformat()
    lines = [
        '---',
        f'title: "Reasoning Report"',
        f'tags: [cerebro, reasoning]',
        f'type: daily',
        f'created: {today}',
        '---',
        '',
        '# Reasoning Report',
        f'_Generated {today}_',
        '',
    ]

    # Delta summary for follow-up runs
    if is_follow_up:
        lines.append('## Delta Summary')
        lines.append('_This report builds on a previous analysis._')
        lines.append('')
        if changed_note_titles:
            lines.append(f'**{len(changed_note_titles)} notes changed** since last run:')
            for title in changed_note_titles[:20]:
                lines.append(f'- [[{title}]]')
            if len(changed_note_titles) > 20:
                lines.append(f'- _...and {len(changed_note_titles) - 20} more_')
        else:
            lines.append('_No notes changed since last run — re-analyzing with fresh eyes._')
        lines.append('')

    # Pass 1: Latent Connections
    lines.append('## 1. Latent Connections')
    lines.append('_∞ = no existing path between notes — these are your blind spots and the highest-value links._')
    lines.append('')
    if latent:
        # Sort: ∞ hops first, then by similarity descending
        def sort_key(item):
            hops = item.get('hops')
            return (0 if hops is None else 1, -(item.get('similarity', 0)))
        latent_sorted = sorted(latent, key=sort_key)
        lines.append('| Note A | Note B | Similarity | Hops | Reasoning | Action |')
        lines.append('|--------|--------|------------|------|-----------|--------|')
        for item in latent_sorted:
            reasoning = item['reasoning'].replace('|', '/').replace('\n', ' ')[:80]
            hops = item.get('hops')
            hop_str = str(hops) if hops is not None else '∞'
            lines.append(
                f"| [[{item['note_a']}]] | [[{item['note_b']}]] | "
                f"{item['similarity']} | {hop_str} | {reasoning} | {item['action']} |"
            )
    else:
        lines.append('No latent connections found above threshold.')
    lines.append('')

    # Pass 2: Knowledge Gaps
    lines.append('## 2. Knowledge Gaps')
    lines.append('')

    lines.append('### Type Distribution')
    if gaps['type_distribution']:
        lines.append('| Type | Count |')
        lines.append('|------|-------|')
        for t, c in sorted(gaps['type_distribution'].items(), key=lambda x: -x[1]):
            lines.append(f'| {t} | {c} |')
    lines.append('')

    if gaps.get('untyped'):
        lines.append(f"**Untyped notes ({len(gaps['untyped'])}):** {', '.join(gaps['untyped'][:10])}")
        lines.append('')

    if gaps.get('hub_gaps'):
        lines.append('### Hub Gaps')
        lines.append('_Notes sharing tags with a hub but not linked to it._')
        lines.append('')
        for hg in gaps['hub_gaps']:
            lines.append(f"**[[{hg['hub']}]]** — missing connections:")
            for m in hg['missing_connections']:
                lines.append(f"  - [[{m['title']}]] (shared: {', '.join(m['shared_tags'])})")
        lines.append('')

    if gaps.get('underlinked_notes'):
        lines.append('### Underlinked Notes')
        lines.append('| Note | Type | Expected Links | Actual |')
        lines.append('|------|------|---------------|--------|')
        for item in gaps['underlinked_notes']:
            lines.append(f"| [[{item['title']}]] | {item['type']} | {item['expected']} | {item['actual']} |")
    lines.append('')

    if gaps.get('community_gaps'):
        lines.append('### Community Gaps')
        lines.append('_Communities missing expected note types — structural blind spots._')
        lines.append('')
        for cg in gaps['community_gaps']:
            sample = ', '.join(f'[[{t}]]' for t in cg['sample_titles'][:4])
            lines.append(f"- **{cg['size']}-note cluster** (tags: {', '.join(cg['community_tags'][:3])}): "
                         f"missing **{', '.join(cg['missing_types'])}** types. Sample: {sample}")
        lines.append('')

    # Pass 3: Cluster Narratives
    lines.append('## 3. Cluster Narratives')
    lines.append('_Semantic groupings across the knowledge graph._')
    lines.append('')
    if clusters:
        for cs in clusters:
            lines.append(f"### {cs['name']} ({cs['size']} notes)")
            lines.append(f"_{cs.get('narrative', '')}_")
            if cs.get('hierarchy_context'):
                lines.append(f"_{cs['hierarchy_context']}_")
            lines.append('')
            lines.append(f"Top tags: {', '.join(cs.get('top_tags', []))}")
            lines.append(f"Notes: {', '.join(f'[[{t}]]' for t in cs['titles'][:8])}")
            if cs.get('boundary_notes'):
                lines.append(f"**Bridge notes:** {', '.join(f'[[{t}]]' for t in cs['boundary_notes'])}")
            lines.append('')
    else:
        lines.append('Not enough notes for clustering.')
    lines.append('')

    # Pass 3b: Inter-Cluster Bridges
    lines.append('## 3b. Inter-Cluster Bridges')
    lines.append('_Deeper themes connecting seemingly unrelated clusters._')
    lines.append('')
    if bridges:
        for b in bridges:
            lines.append(f"### {b['cluster_a']} ↔ {b['cluster_b']} (sim: {b['best_similarity']})")
            lines.append(f"{b['bridge_theme']}")
            lines.append('')
            if b.get('bridge_path'):
                lines.append(f"**Path:** {b['bridge_path']}")
                lines.append('')
            for p in b['top_pairs']:
                lines.append(f"- [[{p['note_a']}]] ↔ [[{p['note_b']}]] ({p['similarity']})")
            lines.append('')
    else:
        lines.append('No inter-cluster bridges found.')
    lines.append('')

    # Pass 4: Contradictions
    lines.append('## 4. Contradictions')
    lines.append('_Notes that may make conflicting claims._')
    lines.append('')
    if contradictions:
        lines.append('| Note A | Note B | Issue | Action |')
        lines.append('|--------|--------|-------|--------|')
        for item in contradictions:
            explanation = item['explanation'].replace('|', '/').replace('\n', ' ')[:100]
            lines.append(
                f"| [[{item['note_a']}]] | [[{item['note_b']}]] | "
                f"{explanation} | {item['action']} |"
            )
    else:
        lines.append('No contradictions detected.')
    lines.append('')

    # Pass 5: Drift Alerts
    lines.append('## 5. Drift Alerts')
    lines.append('_Notes that have evolved significantly from their original scope._')
    lines.append('')
    if drift:
        lines.append('| Note | Type | Drift | Action | Suggestion |')
        lines.append('|------|------|-------|--------|------------|')
        for item in drift:
            lines.append(
                f"| [[{item['title']}]] | {item['type']} | {item['drift']} | "
                f"{item['action']} | {item['suggestion'][:60]} |"
            )
    else:
        lines.append('No significant drift detected.')
    lines.append('')

    # Pass 6: Synthesis Opportunities
    lines.append('## 6. Synthesis Opportunities')
    lines.append('_Tightly-connected note clusters that could use a unifying hub note._')
    lines.append('')
    if synthesis:
        for item in synthesis:
            lines.append(f"### Proposed: {item['suggested_title']}")
            lines.append(f"_{item['size']} notes, {item['density']:.0%} connected_")
            lines.append(f"Members: {', '.join(f'[[{t}]]' for t in item['members'][:8])}")
            lines.append('')
            if item.get('outline'):
                lines.append(item['outline'])
            lines.append('')
    else:
        lines.append('No synthesis opportunities found.')
    lines.append('')

    return '\n'.join(lines)


# ── Prune ───────────────────────────────────────────────────────────────

def prune_writeback_links(dry_run=True):
    """Remove links from ## Related sections that don't pass graph-aware filtering.

    Loads the graph, rebuilds adjacency *without* Related-section links, then
    checks each Related link against the graph-aware threshold. Links that
    wouldn't pass are removed.

    When dry_run=True, prints what would be removed without modifying files.
    Returns count of links removed.
    """
    import re

    npath = _notes_path()

    # First, collect all Related-section links per file
    related_links = {}  # filename -> list of link titles
    for f in os.listdir(npath):
        if not f.endswith('.md'):
            continue
        filepath = os.path.join(npath, f)
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()
        if '\n## Related' not in content:
            continue
        idx = content.index('\n## Related')
        rest = content[idx:]
        # Find end of Related section
        next_section = rest[1:].find('\n## ')
        section = rest if next_section < 0 else rest[:next_section + 1]
        links = re.findall(r'- \[\[([^\]]+)\]\]', section)
        if links:
            related_links[f] = links

    if not related_links:
        print('[prune] No ## Related sections found.')
        return 0

    # Load graph and build adjacency WITHOUT Related-section links
    # to get the "natural" graph topology
    graph = load_graph()
    notes_by_id = graph['notes_by_id']
    title_to_id = graph['title_to_id']
    embeddings = graph['embeddings']
    edge_pairs = graph['edge_pairs']

    # Build a clean adjacency excluding Related-section edges
    related_pairs = set()
    for f, links in related_links.items():
        nid = filename_to_id(f)
        for link_title in links:
            target_id = title_to_id.get(link_title.lower())
            if target_id:
                related_pairs.add((nid, target_id))
                related_pairs.add((target_id, nid))

    clean_adjacency = defaultdict(set)
    for a, b in edge_pairs:
        if (a, b) not in related_pairs:
            clean_adjacency[a].add(b)
            clean_adjacency[b].add(a)

    # Check each Related link against graph-aware threshold
    to_remove = {}  # filename -> list of link titles to remove
    total_checked = 0
    total_removed = 0

    for f, links in related_links.items():
        nid = filename_to_id(f)
        if nid not in embeddings:
            continue
        removals = []
        for link_title in links:
            target_id = title_to_id.get(link_title.lower())
            if not target_id or target_id not in embeddings:
                continue
            total_checked += 1

            sim = _cosine_similarity(embeddings[nid], embeddings[target_id])
            hops = _bfs_distance(clean_adjacency, nid, target_id)
            threshold = _graph_aware_threshold(hops)

            if sim < threshold:
                removals.append(link_title)
                hop_str = str(hops) if hops is not None else 'disconnected'
                print(f'  [prune] {notes_by_id[nid]["title"]} -> [[{link_title}]]: '
                      f'sim={sim:.3f} < threshold={threshold:.2f} ({hop_str})')

        if removals:
            to_remove[f] = removals
            total_removed += len(removals)

    print(f'\n[prune] Checked {total_checked} links, {total_removed} would be removed')

    if dry_run or not to_remove:
        if to_remove:
            print('[prune] Dry run — use --prune --apply to execute.')
        return total_removed

    # Apply removals
    for f, removals in to_remove.items():
        filepath = os.path.join(npath, f)
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()

        for link_title in removals:
            content = re.sub(rf'\n- \[\[{re.escape(link_title)}\]\]', '', content)

        # Clean up empty Related sections
        content = re.sub(r'\n## Related\n+(?=\n## |\Z)', '', content)

        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(content)

    print(f'[prune] Removed {total_removed} links from {len(to_remove)} notes')
    return total_removed


# ── Write-back ──────────────────────────────────────────────────────────

def apply_writeback(graph, latent, gaps):
    """Insert wikilinks into notes based on latent connections and hub gaps.

    For latent connections with action='link': adds [[Note B]] to Note A's body
    and [[Note A]] to Note B's body (bidirectional).

    For hub gaps: adds missing [[Note]] links to hub notes.

    Returns count of notes modified.
    """
    import re

    notes_by_id = graph['notes_by_id']
    title_to_id = graph['title_to_id']
    npath = _notes_path()
    modified = set()

    def _add_wikilink_to_note(note_id, link_title):
        """Append a wikilink to a note if it doesn't already exist."""
        if note_id not in notes_by_id:
            return False
        note = notes_by_id[note_id]
        filepath = os.path.join(npath, note['file'])

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if link already exists (any form)
        link_pattern = re.escape(link_title)
        if re.search(rf'\[\[([a-z_]*::)?{link_pattern}\]\]', content, re.IGNORECASE):
            return False

        # Find or create a "## Related" section at the end
        if '\n## Related' in content:
            # Append to existing Related section
            idx = content.index('\n## Related')
            # Find the end of the Related section (next ## or EOF)
            rest = content[idx + len('\n## Related'):]
            next_section = rest.find('\n## ')
            if next_section >= 0:
                insert_pos = idx + len('\n## Related') + next_section
                content = content[:insert_pos] + f'\n- [[{link_title}]]' + content[insert_pos:]
            else:
                content = content.rstrip() + f'\n- [[{link_title}]]\n'
        else:
            content = content.rstrip() + f'\n\n## Related\n\n- [[{link_title}]]\n'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    # Apply latent connections (bidirectional)
    link_count = 0
    for item in latent:
        if item.get('action') != 'link':
            continue
        title_a = item['note_a']
        title_b = item['note_b']
        id_a = title_to_id.get(title_a.lower())
        id_b = title_to_id.get(title_b.lower())
        if id_a and id_b:
            if _add_wikilink_to_note(id_a, title_b):
                modified.add(id_a)
                link_count += 1
            if _add_wikilink_to_note(id_b, title_a):
                modified.add(id_b)
                link_count += 1

    # Apply hub gaps
    hub_link_count = 0
    for hg in gaps.get('hub_gaps', []):
        hub_title = hg['hub']
        hub_id = title_to_id.get(hub_title.lower())
        if not hub_id:
            continue
        for missing in hg['missing_connections']:
            if _add_wikilink_to_note(hub_id, missing['title']):
                modified.add(hub_id)
                hub_link_count += 1

    print(f'[apply] Inserted {link_count} latent links across {len(modified)} notes')
    print(f'[apply] Inserted {hub_link_count} hub-gap links')
    return len(modified)


# ── Capture Curation ───────────────────────────────────────────────────

def curate_captures(graph, api_key):
    """Review capture-type notes and decide: promote (add links, retype) or discard.

    Returns list of actions taken.
    """
    notes = graph['notes']

    captures = [n for n in notes if n['type'] == 'capture']
    if not captures:
        print('[curate] No captures to curate.')
        return []

    print(f'[curate] Found {len(captures)} captures to review.')

    if not api_key:
        print('[curate] No API key — cannot curate without Haiku.')
        return []

    # Build context: list of all non-capture note titles for linking suggestions
    all_titles = [n['title'] for n in notes if n['type'] != 'capture']

    results = []

    # Process captures in batches of 5
    for batch_start in range(0, len(captures), 5):
        batch = captures[batch_start:batch_start + 5]

        batch_text = []
        for idx, cap in enumerate(batch):
            batch_text.append(
                f"Capture {idx+1}: \"{cap['title']}\"\n"
                f"  Tags: {', '.join(cap['tags'])}\n"
                f"  Content: {cap['preview'][:1000]}"
            )

        prompt = f"""You are curating a personal knowledge graph. These are uncurated "capture" notes — quick mobile captures that need review.

For each capture, decide:
1. **PROMOTE** — worth keeping. Suggest: a better type (concept, research, project, feedback, profile, anecdote), 2-3 wikilinks to existing notes, and any tag additions.
2. **DISCARD** — too vague, duplicate, or low-value to keep.

Available notes to link to (sample): {', '.join(all_titles[:40])}

Format each as:
[number]. [PROMOTE/DISCARD] | type: [new_type] | links: [[Note1]], [[Note2]] | reason: [why]

{chr(10).join(batch_text)}"""

        try:
            response = _call_haiku(prompt, api_key, max_tokens=1500)
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

            for idx, cap in enumerate(batch):
                if idx >= len(lines):
                    continue
                line = lines[idx]
                action = 'promote' if 'PROMOTE' in line.upper() else 'discard'

                result = {
                    'title': cap['title'],
                    'id': cap['id'],
                    'action': action,
                    'reasoning': line,
                }

                if action == 'promote':
                    # Extract suggested type
                    import re
                    type_match = re.search(r'type:\s*(\w+)', line, re.IGNORECASE)
                    new_type = type_match.group(1).lower() if type_match else None

                    # Extract suggested links
                    link_matches = re.findall(r'\[\[([^\]]+)\]\]', line)

                    result['new_type'] = new_type
                    result['suggested_links'] = link_matches

                results.append(result)

        except Exception as e:
            print(f'[curate] Haiku error: {e}')

    return results


def apply_curation(graph, curation_results):
    """Apply curation decisions to capture notes."""
    import re

    notes_by_id = graph['notes_by_id']
    npath = _notes_path()
    trash_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trash')

    promoted = 0
    discarded = 0

    for result in curation_results:
        nid = result['id']
        if nid not in notes_by_id:
            continue
        note = notes_by_id[nid]
        filepath = os.path.join(npath, note['file'])

        if result['action'] == 'discard':
            # Move to trash
            os.makedirs(trash_dir, exist_ok=True)
            dest = os.path.join(trash_dir, note['file'])
            shutil.move(filepath, dest)
            print(f'  [discard] {note["title"]} -> trash/')
            discarded += 1

        elif result['action'] == 'promote':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Update type tag if suggested
            new_type = result.get('new_type')
            if new_type and new_type != 'capture':
                # Replace 'capture' tag with new type in frontmatter
                content = re.sub(
                    r'(tags:\s*\[)([^\]]*)\]',
                    lambda m: m.group(1) + m.group(2).replace('capture', new_type) + ']',
                    content
                )

            # Add suggested links
            links = result.get('suggested_links', [])
            if links:
                # Check which links don't already exist
                new_links = []
                for link in links:
                    if f'[[{link}]]' not in content:
                        new_links.append(link)
                if new_links:
                    link_text = '\n'.join(f'- [[{l}]]' for l in new_links)
                    if '\n## Related' in content:
                        idx = content.index('\n## Related')
                        rest = content[idx + len('\n## Related'):]
                        next_section = rest.find('\n## ')
                        if next_section >= 0:
                            insert_pos = idx + len('\n## Related') + next_section
                            content = content[:insert_pos] + '\n' + link_text + content[insert_pos:]
                        else:
                            content = content.rstrip() + '\n' + link_text + '\n'
                    else:
                        content = content.rstrip() + f'\n\n## Related\n\n{link_text}\n'

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'  [promote] {note["title"]} -> {new_type or "kept type"}')
            promoted += 1

    print(f'[curate] Promoted {promoted}, discarded {discarded}')
    return {'promoted': promoted, 'discarded': discarded}


# ── Backup ─────────────────────────────────────────────────────────────

def backup_cerebro():
    """Create a timestamped zip of heartwood/notes/ and heartwood/.graph/."""
    import zipfile

    cerebro_dir = os.path.dirname(os.path.abspath(__file__))
    notes_path = _notes_path()
    graph_path = _graph_path()
    ontology_path = os.path.join(cerebro_dir, 'ontology.yaml')

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_dir = os.path.join(cerebro_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    zip_path = os.path.join(backup_dir, f'cerebro-backup-{timestamp}.zip')

    file_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # notes/
        if os.path.exists(notes_path):
            for f in os.listdir(notes_path):
                fpath = os.path.join(notes_path, f)
                if os.path.isfile(fpath):
                    zf.write(fpath, os.path.join('notes', f))
                    file_count += 1

        # .graph/
        if os.path.exists(graph_path):
            for f in os.listdir(graph_path):
                fpath = os.path.join(graph_path, f)
                if os.path.isfile(fpath):
                    zf.write(fpath, os.path.join('.graph', f))
                    file_count += 1

        # ontology.yaml
        if os.path.exists(ontology_path):
            zf.write(ontology_path, 'ontology.yaml')
            file_count += 1

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f'[backup] Created {zip_path}')
    print(f'[backup] {file_count} files, {size_mb:.1f} MB')

    # Prune old backups — keep last 10
    backups = sorted([
        f for f in os.listdir(backup_dir) if f.startswith('cerebro-backup-') and f.endswith('.zip')
    ])
    while len(backups) > 10:
        old = backups.pop(0)
        os.remove(os.path.join(backup_dir, old))
        print(f'[backup] Pruned old backup: {old}')

    return zip_path


# ── Narration ────────────────────────────────────────────────────────────

def _build_narration_prompt(latent, gaps, clusters, bridges, contradictions, drift, synthesis):
    """Build a prompt asking Haiku to narrate the reasoning results."""
    sections = []

    if latent:
        items = []
        for item in latent[:5]:
            items.append(f"  {item['note_a']} ↔ {item['note_b']} (sim: {item['similarity']}, "
                         f"hops: {item.get('hops', '∞')}): {item['reasoning'][:80]}")
        sections.append(f"LATENT CONNECTIONS ({len(latent)} found):\n" + '\n'.join(items))

    if clusters:
        items = []
        for cs in clusters:
            name = cs.get('name', 'Cluster %d' % (cs['index'] + 1))
            narrative = cs.get('narrative', ', '.join(cs['titles'][:5]))
            items.append(f"  {name} ({cs['size']} notes): {narrative}")
        sections.append(f"CLUSTERS ({len(clusters)}):\n" + '\n'.join(items))

    if bridges:
        items = []
        for b in bridges:
            pairs = ', '.join(f"{p['note_a']} ↔ {p['note_b']}" for p in b['top_pairs'][:2])
            items.append(f"  {b['cluster_a']} ↔ {b['cluster_b']} (sim: {b['best_similarity']}): "
                         f"{b['bridge_theme'][:120]}\n    Key pairs: {pairs}")
        sections.append(f"INTER-CLUSTER BRIDGES ({len(bridges)}):\n" + '\n'.join(items))

    if contradictions:
        items = [f"  {c['note_a']} vs {c['note_b']}: {c['explanation'][:80]}" for c in contradictions[:3]]
        sections.append(f"CONTRADICTIONS ({len(contradictions)}):\n" + '\n'.join(items))

    if drift:
        items = [f"  {d['title']} (drift: {d['drift']}): {d['suggestion'][:60]}" for d in drift[:3]]
        sections.append(f"DRIFT ({len(drift)}):\n" + '\n'.join(items))

    hub_gaps = gaps.get('hub_gaps', [])
    underlinked = gaps.get('underlinked_notes', [])
    if hub_gaps or underlinked:
        gap_items = []
        for hg in hub_gaps[:3]:
            missing = [m['title'] for m in hg['missing_connections'][:3]]
            gap_items.append(f"  Hub '{hg['hub']}' missing: {', '.join(missing)}")
        if underlinked:
            gap_items.append(f"  {len(underlinked)} underlinked notes")
        sections.append(f"KNOWLEDGE GAPS:\n" + '\n'.join(gap_items))

    if synthesis:
        items = [f"  {s['suggested_title']} ({s['size']} notes, {s['density']:.0%} connected)"
                 for s in synthesis[:3]]
        sections.append(f"SYNTHESIS OPPORTUNITIES:\n" + '\n'.join(items))

    data = '\n\n'.join(sections)

    return f"""You are narrating the results of a reasoning engine that analyzes a personal knowledge graph (118 notes about projects, identity, family, career, creativity, and style).

Write a 150-250 word narrative summary addressed directly to the graph's owner. Be specific — name actual notes and connections. Lead with the most interesting finding. Skip anything empty or boring. Write like a sharp colleague briefing someone, not a report generator.

Do NOT use headers, bullets, or markdown. Just flowing prose, 2-3 short paragraphs.

Here are the results:

{data}"""


def generate_narration(latent, gaps, clusters, bridges, contradictions, drift, synthesis, api_key):
    """Generate a readable narrative of the reasoning results. Returns the text, or None."""
    if not api_key:
        return None

    has_content = (latent or bridges or contradictions or drift
                   or gaps.get('hub_gaps') or gaps.get('underlinked_notes') or synthesis)
    if not has_content:
        return None

    prompt = _build_narration_prompt(latent, gaps, clusters, bridges, contradictions, drift, synthesis)

    try:
        return _call_haiku(prompt, api_key, max_tokens=1500).strip()
    except Exception as e:
        print(f'\n[narrate] Haiku error: {e}')
        return None


# ── Hosted entry point ───────────────────────────────────────────────────

def run_reasoning_from_data(notes_data, connections, ontology, api_key,
                            quiet=False, progress_cb=None,
                            previous_report=None, changed_note_titles=None):
    """Run the full reasoning pipeline from pre-fetched data.

    Returns dict: {success: bool, report: str, passes: dict}
    progress_cb: optional callable(pass_name, status_msg) for progress updates.
    previous_report: markdown text of the last reasoning report (or None for first run).
    changed_note_titles: list of note titles created/modified since last run (or None).
    """
    def _progress(pass_name, msg):
        if progress_cb:
            progress_cb(pass_name, msg)

    _progress('loading', 'Building graph from data...')
    graph = load_graph_from_data(notes_data, connections, ontology)
    _progress('loading', f'Loaded {len(graph["notes"])} notes, {len(graph["edges"])} edges')

    # v2: Build graph analysis layer
    _progress('analysis', 'Running graph analysis (centrality, communities, edge surprise)...')
    ga = GraphAnalysis(graph)
    ga.analyze()
    _progress('analysis', f'Graph analysis complete: {len(ga.communities.get(1.0, []))} communities')

    if not api_key and not quiet:
        quiet = True

    # Build per-section memory context (replaces _build_prior_context)
    def _prior(section_name):
        return build_memory_context(previous_report, changed_note_titles, section_name)

    is_follow_up = previous_report is not None

    latent = []
    gaps = {'type_distribution': {}, 'untyped': [], 'hub_gaps': [], 'underlinked_notes': []}
    clusters = []
    bridges = []
    contradictions = []
    drift = []
    synthesis = []

    _progress('latent', 'Pass 1: Latent connections...')
    latent = pass_latent(graph, api_key, quiet, prior_context=_prior('Latent Connections'), ga=ga)
    _progress('latent', f'{len(latent)} candidates')

    _progress('gaps', 'Pass 2: Knowledge gaps...')
    gaps = pass_gaps(graph, ga=ga)
    _progress('gaps', f'{len(gaps.get("hub_gaps", []))} hub gaps')

    _progress('clusters', 'Pass 3: Cluster narratives...')
    clusters = pass_clusters(graph, api_key, quiet, prior_context=_prior('Cluster Narratives'), ga=ga)
    _progress('clusters', f'{len(clusters)} clusters')

    _progress('bridges', 'Pass 3b: Inter-cluster bridges...')
    bridges = pass_bridges(graph, clusters, api_key, quiet, prior_context=_prior('Inter-Cluster Bridges'), ga=ga)
    _progress('bridges', f'{len(bridges)} bridges')

    _progress('contradictions', 'Pass 4: Contradiction detection...')
    contradictions = pass_contradictions(graph, api_key, quiet, prior_context=_prior('Contradictions'), ga=ga)
    _progress('contradictions', f'{len(contradictions)} contradictions')

    _progress('drift', 'Pass 5: Drift alerts...')
    drift = pass_drift(graph, ga=ga)
    _progress('drift', f'{len(drift)} drifted notes')

    _progress('synthesis', 'Pass 6: Synthesis opportunities...')
    synthesis = pass_synthesis(graph, api_key, quiet, prior_context=_prior('Synthesis Opportunities'))
    _progress('synthesis', f'{len(synthesis)} proposals')

    _progress('report', 'Generating report...')
    report = generate_report(latent, gaps, clusters, bridges, contradictions, drift, synthesis,
                             is_follow_up=is_follow_up,
                             changed_note_titles=changed_note_titles)

    # Phase 3: Post-run memory writing (each step independent)
    memory_results = {}
    if api_key and not quiet:
        _progress('memory', 'Running post-run memory system...')
        memory_results = post_run_memory(
            report, api_key, _call_haiku,
            progress_cb=lambda step, msg: _progress(step, msg),
        )

    # Phase 4a: Incremental claim extraction
    claim_stats = {}
    if api_key and not quiet:
        _progress('claims', 'Running incremental claim extraction...')
        try:
            claim_stats = extract_all_claims(model='claude-haiku-4-5-20251001')
            _progress('claims', f'{claim_stats["extracted"]} extracted, '
                      f'{claim_stats["cached"]} cached')
        except Exception as e:
            _progress('claims', f'Failed: {e}')

    # Phase 4b: Belief revision
    rev_results = {}
    if api_key and not quiet:
        _progress('revision', 'Running belief revision...')
        try:
            rev_results = run_belief_revision(graph, api_key=api_key, call_llm=_call_haiku)
            if not rev_results.get('skipped'):
                _progress('revision', f'{rev_results["confirmed"]} contradictions, '
                          f'{rev_results["revisions"]} revisions')
        except Exception as e:
            _progress('revision', f'Failed: {e}')

    # Phase 5: Link prediction
    link_results = {}
    if api_key and not quiet:
        _progress('links', 'Running link prediction...')
        try:
            link_results = run_link_prediction(graph, ga, api_key=api_key, call_llm=_call_haiku)
            _progress('links', f'{link_results["predictions"]} predictions from '
                      f'{link_results["candidates"]} candidates, '
                      f'{link_results["rules"]} rules mined')
        except Exception as e:
            _progress('links', f'Failed: {e}')

    return {
        'success': True,
        'report': report,
        'passes': {
            'latent': len(latent),
            'gaps': len(gaps.get('hub_gaps', [])),
            'clusters': len(clusters),
            'bridges': len(bridges),
            'contradictions': len(contradictions),
            'drift': len(drift),
            'synthesis': len(synthesis),
        },
        'memory': memory_results,
        'claims': claim_stats,
        'revision': rev_results,
        'links': link_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Heartwood Reasoning Engine')
    parser.add_argument('--pass', dest='single_pass', choices=[
        'latent', 'gaps', 'clusters', 'bridges', 'contradictions', 'drift', 'synthesis'
    ], help='Run a single pass only')
    parser.add_argument('--quiet', action='store_true',
                        help='Structural passes only — no Haiku API calls')
    parser.add_argument('--apply', action='store_true',
                        help='Write back: insert wikilinks from latent connections + hub gaps into notes')
    parser.add_argument('--curate', action='store_true',
                        help='Curate captures: promote or discard uncurated capture notes')
    parser.add_argument('--backup', action='store_true',
                        help='Backup notes/ and .graph/ to a timestamped zip')
    parser.add_argument('--narrate', action='store_true',
                        help='Print a readable narrative summary to the terminal')
    parser.add_argument('--prune', action='store_true',
                        help='Remove ## Related links that fail graph-aware filtering')
    args = parser.parse_args()

    # Handle standalone commands first
    if args.backup:
        backup_cerebro()
        return

    if args.prune:
        prune_writeback_links(dry_run=not args.apply)
        return

    print('[reason] Loading graph...')
    graph = load_graph()
    print(f'[reason] Loaded {len(graph["notes"])} notes, {len(graph["edges"])} edges, '
          f'{len(graph["embeddings"])} embeddings')

    api_key = _get_api_key()
    if not api_key and not args.quiet:
        print('[reason] Warning: no API key found — running in quiet mode')
        args.quiet = True

    # Handle --curate
    if args.curate:
        results = curate_captures(graph, api_key)
        if results:
            print(f'\n[curate] Curation plan:')
            for r in results:
                action_label = 'PROMOTE' if r['action'] == 'promote' else 'DISCARD'
                print(f'  {action_label}: {r["title"]}')
                if r['action'] == 'promote':
                    print(f'         -> type: {r.get("new_type", "?")}  links: {r.get("suggested_links", [])}')

            if args.apply:
                print('\n[curate] Applying curation...')
                apply_curation(graph, results)
            else:
                print('\n[curate] Dry run — use --curate --apply to execute.')
        return

    # v2: Build graph analysis layer
    print('[reason] Running graph analysis (centrality, communities, edge surprise)...')
    ga = GraphAnalysis(graph)
    ga.analyze()
    print(f'[reason] Graph analysis: {ga.G.number_of_nodes()} nodes, '
          f'{len(ga.communities.get(1.0, []))} communities, '
          f'{len(ga.edge_surprise)} edge surprise scores')

    run_all = args.single_pass is None

    # Phase 3: Build memory context for prompt injection
    def _prior(section_name):
        return build_memory_context(section_name=section_name)

    # Run passes
    latent = []
    gaps = {'type_distribution': {}, 'untyped': [], 'hub_gaps': [], 'underlinked_notes': []}
    clusters = []
    bridges = []
    contradictions = []
    drift = []
    synthesis = []

    if run_all or args.single_pass == 'latent':
        print('[reason] Pass 1: Latent connections...')
        latent = pass_latent(graph, api_key, args.quiet, prior_context=_prior('Latent Connections'), ga=ga)
        print(f'  -> {len(latent)} candidates')

    if run_all or args.single_pass == 'gaps':
        print('[reason] Pass 2: Knowledge gaps...')
        gaps = pass_gaps(graph, ga=ga)
        print(f'  -> {len(gaps.get("hub_gaps", []))} hub gaps, '
              f'{len(gaps.get("underlinked_notes", []))} underlinked')

    if run_all or args.single_pass in ('clusters', 'bridges'):
        print('[reason] Pass 3: Cluster narratives...')
        clusters = pass_clusters(graph, api_key, args.quiet, prior_context=_prior('Cluster Narratives'), ga=ga)
        print(f'  -> {len(clusters)} clusters')

    if run_all or args.single_pass == 'bridges':
        print('[reason] Pass 3b: Inter-cluster bridges...')
        bridges = pass_bridges(graph, clusters, api_key, args.quiet, prior_context=_prior('Inter-Cluster Bridges'), ga=ga)
        print(f'  -> {len(bridges)} bridges')

    if run_all or args.single_pass == 'contradictions':
        print('[reason] Pass 4: Contradiction detection...')
        contradictions = pass_contradictions(graph, api_key, args.quiet, prior_context=_prior('Contradictions'), ga=ga)
        print(f'  -> {len(contradictions)} contradictions')

    if run_all or args.single_pass == 'drift':
        print('[reason] Pass 5: Drift alerts...')
        drift = pass_drift(graph, ga=ga)
        print(f'  -> {len(drift)} drifted notes')

    if run_all or args.single_pass == 'synthesis':
        print('[reason] Pass 6: Synthesis opportunities...')
        synthesis = pass_synthesis(graph, api_key, args.quiet, prior_context=_prior('Synthesis Opportunities'))
        print(f'  -> {len(synthesis)} synthesis proposals')

    # Generate narration if requested (before report, so we can include it)
    narration = None
    if args.narrate:
        print('[reason] Generating narration...')
        narration = generate_narration(latent, gaps, clusters, bridges, contradictions, drift, synthesis, api_key)

    # Generate report
    print('[reason] Generating report...')
    report = generate_report(latent, gaps, clusters, bridges, contradictions, drift, synthesis)

    # Append narration to report if we have one
    if narration:
        report += '\n## Briefing\n\n' + narration + '\n'

    # Write to heartwood/reports/ only (not notes/ — report is an artifact, not a knowledge node)
    reports_dir = _reports_path()
    os.makedirs(reports_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    report_path = os.path.join(reports_dir, f'reasoning-report-{today}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    # Also write a stable path for the app to read
    latest_path = os.path.join(reports_dir, 'reasoning-report-latest.md')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[reason] Report written to {report_path}')

    # Print narration to terminal
    if narration:
        print('\n' + '=' * 60)
        print('  CEREBRO — What your knowledge graph is telling you')
        print('=' * 60 + '\n')
        print(narration)
        print('\n' + '=' * 60 + '\n')
    elif args.narrate:
        print('\n[narrate] Nothing notable to report.')

    # Phase 3: Post-run memory system
    if api_key and not args.quiet and run_all:
        print('[reason] Running post-run memory system...')
        mem_results = post_run_memory(
            report, api_key, _call_haiku,
            progress_cb=lambda step, msg: print(f'  [{step}] {msg}'),
        )
        successes = sum([
            mem_results.get('reflection') is not None,
            mem_results.get('summary', False),
            mem_results.get('rules', False),
            mem_results.get('strategies', False),
        ])
        print(f'[reason] Memory system: {successes}/4 steps succeeded')

    # Phase 4a: Incremental claim extraction (only modified notes)
    if api_key and not args.quiet and run_all:
        print('[reason] Running incremental claim extraction...')
        try:
            claim_stats = extract_all_claims(model='claude-haiku-4-5-20251001')
            print(f'[reason] Claims: {claim_stats["extracted"]} extracted, '
                  f'{claim_stats["cached"]} cached, '
                  f'{claim_stats["total_claims"]} total')
        except Exception as e:
            print(f'[reason] Claim extraction failed: {e}')

    # Phase 4b: Belief revision (3-layer contradiction detection)
    if api_key and not args.quiet and run_all:
        print('[reason] Running belief revision...')
        try:
            rev_results = run_belief_revision(graph, api_key=api_key, call_llm=_call_haiku)
            if not rev_results.get('skipped'):
                print(f'[reason] Belief revision: {rev_results["total_candidates"]} candidates, '
                      f'{rev_results["confirmed"]} confirmed, '
                      f'{rev_results["revisions"]} revisions')
        except Exception as e:
            print(f'[reason] Belief revision failed: {e}')

    # Phase 5: Link prediction
    if api_key and not args.quiet and run_all:
        print('[reason] Running link prediction...')
        try:
            link_results = run_link_prediction(graph, ga, api_key=api_key, call_llm=_call_haiku)
            print(f'[reason] Link prediction: {link_results["rules"]} rules mined, '
                  f'{link_results["candidates"]} candidates, '
                  f'{link_results["predictions"]} predictions')
            for p in link_results.get('top_predictions', [])[:3]:
                rel = f' ({p["relation"]})' if p.get('relation') else ''
                print(f'  -> {p["source"]} <-> {p["target"]} [score={p["score"]:.3f}]{rel}')
        except Exception as e:
            print(f'[reason] Link prediction failed: {e}')

    # Apply write-back if requested
    if args.apply:
        print('\n[reason] Applying write-back...')
        apply_writeback(graph, latent, gaps)

    print('[reason] Done.')


if __name__ == '__main__':
    main()
