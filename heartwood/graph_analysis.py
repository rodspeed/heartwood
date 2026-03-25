"""Heartwood Graph Analysis Layer — Phase 1 of Reasoning Engine v2.

NetworkX-based structural analysis: centrality metrics, edge surprise scores,
multi-resolution community detection, Personalized PageRank subgraph extraction,
and path verbalization. Graph algorithms do the structural reasoning; the LLM
does the semantic reasoning.

Usage:
    from graph_analysis import GraphAnalysis
    ga = GraphAnalysis(graph)   # graph dict from reason.load_graph()
    ga.analyze()                # compute all metrics
    # Then use ga.centrality, ga.communities, ga.edge_surprise, etc.
"""

import networkx as nx
from collections import defaultdict


class GraphAnalysis:
    """Structural analysis of the Heartwood knowledge graph."""

    def __init__(self, graph):
        """Initialize from a Heartwood graph dict (from reason.load_graph()).

        graph must have: notes, edges, embeddings, notes_by_id, title_to_id
        """
        self.graph = graph
        self.G = self._build_nx_graph()
        self.centrality = {}      # {metric_name: {node_id: score}}
        self.communities = {}     # {resolution: list of frozensets}
        self.community_map = {}   # {resolution: {node_id: community_index}}
        self.edge_surprise = {}   # {(u, v): surprise_score}

    def _build_nx_graph(self):
        """Convert Heartwood's graph dict to a NetworkX DiGraph with typed edges."""
        G = nx.DiGraph()
        for note in self.graph['notes']:
            G.add_node(note['id'], title=note['title'], type=note['type'],
                       tags=note['tags'], created=note['created'])
        for edge in self.graph['edges']:
            G.add_edge(edge['source'], edge['target'],
                       type=edge.get('type', 'wikilink'),
                       rel=edge.get('rel', 'related_to'))
        return G

    def analyze(self):
        """Run all structural analyses. Call once after construction."""
        self._compute_centrality()
        self._compute_edge_surprise()
        self._detect_communities()

    # ── Centrality ──────────────────────────────────────────────────────

    def _compute_centrality(self):
        """Compute betweenness, PageRank, and Burt's constraint."""
        G_undirected = self.G.to_undirected()

        self.centrality['betweenness'] = nx.betweenness_centrality(self.G)
        self.centrality['pagerank'] = nx.pagerank(self.G, alpha=0.85)

        # Burt's constraint requires undirected graph and nodes with edges
        try:
            constraint = nx.constraint(G_undirected)
            self.centrality['constraint'] = constraint
        except Exception:
            self.centrality['constraint'] = {}

    def get_bridge_nodes(self, top_k=15):
        """Return top structural hole spanners: high betweenness + low constraint.

        These nodes bridge otherwise disconnected communities — the most
        insightful nodes for cross-domain reasoning.
        """
        betweenness = self.centrality.get('betweenness', {})
        constraint = self.centrality.get('constraint', {})

        scores = {}
        for node in self.G.nodes():
            b = betweenness.get(node, 0)
            # Low constraint = high brokerage. Invert: brokerage = 1 - constraint.
            # Nodes not in constraint dict (isolates) get score 0.
            c = constraint.get(node)
            if c is None:
                brokerage = 0
            else:
                brokerage = max(0, 1.0 - c)
            # Combined score: geometric mean of betweenness and brokerage
            scores[node] = (b * brokerage) ** 0.5

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(nid, score) for nid, score in ranked[:top_k] if score > 0]

    def get_hub_nodes(self, top_k=10):
        """Return top hub nodes by PageRank."""
        pr = self.centrality.get('pagerank', {})
        ranked = sorted(pr.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    # ── Edge Surprise ───────────────────────────────────────────────────

    def _compute_edge_surprise(self):
        """Score every edge by how unexpected the connection is.

        High surprise = bridging edge (dissimilar nodes connected).
        Low surprise = reinforcing edge (similar nodes connected).
        """
        embeddings = self.graph['embeddings']

        for u, v in self.G.edges():
            if u in embeddings and v in embeddings:
                sim = _cosine_similarity(embeddings[u], embeddings[v])
                surprise = 1.0 - sim
            else:
                surprise = 0.5  # default for missing embeddings
            self.edge_surprise[(u, v)] = surprise
            self.G[u][v]['surprise'] = surprise

    def get_surprising_edges(self, top_k=20):
        """Return edges ranked by surprise score (most unexpected first)."""
        ranked = sorted(self.edge_surprise.items(), key=lambda x: -x[1])
        notes_by_id = self.graph['notes_by_id']
        results = []
        for (u, v), surprise in ranked[:top_k]:
            if u in notes_by_id and v in notes_by_id:
                results.append({
                    'source': u,
                    'target': v,
                    'source_title': notes_by_id[u]['title'],
                    'target_title': notes_by_id[v]['title'],
                    'surprise': round(surprise, 3),
                    'edge_type': self.G[u][v].get('rel', 'related_to'),
                })
        return results

    # ── Community Detection ─────────────────────────────────────────────

    def _detect_communities(self, resolutions=(0.5, 1.0, 2.0)):
        """Multi-resolution Louvain community detection.

        Low resolution (0.5): 3-4 macro-themes
        Medium resolution (1.0): 6-8 clusters
        High resolution (2.0): 12-15 micro-clusters
        """
        G_undirected = self.G.to_undirected()

        # Skip if graph is too small or disconnected with no edges
        if G_undirected.number_of_edges() == 0:
            return

        for res in resolutions:
            communities = nx.community.louvain_communities(
                G_undirected, resolution=res, seed=42)
            self.communities[res] = communities

            # Build node -> community index map
            cmap = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    cmap[node] = idx
            self.community_map[res] = cmap

    def get_communities(self, resolution=1.0):
        """Return communities at a given resolution with metadata."""
        communities = self.communities.get(resolution, [])
        notes_by_id = self.graph['notes_by_id']
        results = []

        for idx, comm in enumerate(communities):
            members = [nid for nid in comm if nid in notes_by_id]
            if not members:
                continue

            titles = [notes_by_id[nid]['title'] for nid in members]
            tags = defaultdict(int)
            types = defaultdict(int)
            for nid in members:
                note = notes_by_id[nid]
                for tag in note['tags']:
                    tags[tag] += 1
                if note['type']:
                    types[note['type']] += 1

            top_tags = [t for t, _ in sorted(tags.items(), key=lambda x: -x[1])[:5]]

            results.append({
                'index': idx,
                'member_ids': members,
                'titles': titles,
                'size': len(members),
                'top_tags': top_tags,
                'type_distribution': dict(types),
            })

        return results

    def get_community_hierarchy(self):
        """Return the multi-resolution community hierarchy.

        Maps each node to its community at each resolution level, enabling
        narratives like "these 5 notes form a micro-cluster about X, which
        is part of a broader cluster about Y."
        """
        hierarchy = {}
        for res in sorted(self.communities.keys()):
            hierarchy[res] = {
                'resolution': res,
                'num_communities': len(self.communities[res]),
                'communities': self.get_communities(res),
            }
        return hierarchy

    def get_community_gaps(self, resolution=1.0):
        """Find communities missing expected note types.

        If a community about 'research' topics has no 'project' type notes,
        that's a structural gap worth surfacing.
        """
        communities = self.get_communities(resolution)
        gaps = []

        # Expected types that should appear in non-trivial communities
        expected_types = {'concept', 'project', 'research'}

        for comm in communities:
            if comm['size'] < 4:
                continue
            present_types = set(comm['type_distribution'].keys())
            missing = expected_types - present_types
            if missing:
                gaps.append({
                    'community_index': comm['index'],
                    'community_tags': comm['top_tags'],
                    'size': comm['size'],
                    'missing_types': list(missing),
                    'present_types': list(present_types),
                    'sample_titles': comm['titles'][:5],
                })

        return gaps

    # ── Personalized PageRank Subgraph ──────────────────────────────────

    def extract_subgraph(self, seed_nodes, top_k=20):
        """PPR-based subgraph extraction.

        Returns the top-k most structurally relevant nodes given seed nodes.
        Use case: for contradiction detection, seed from both candidate notes
        and send only the local subgraph to the LLM.
        """
        # Filter to nodes actually in the graph
        valid_seeds = [n for n in seed_nodes if n in self.G]
        if not valid_seeds:
            return []

        personalization = {n: 1.0 for n in valid_seeds}
        try:
            scores = nx.pagerank(self.G, personalization=personalization, alpha=0.85)
        except nx.NetworkXError:
            return valid_seeds

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [node for node, score in ranked[:top_k]]

    def extract_subgraph_context(self, seed_nodes, top_k=20):
        """Like extract_subgraph but returns full context for LLM consumption.

        Returns a list of dicts with node info + relevance score.
        """
        valid_seeds = [n for n in seed_nodes if n in self.G]
        if not valid_seeds:
            return []

        personalization = {n: 1.0 for n in valid_seeds}
        try:
            scores = nx.pagerank(self.G, personalization=personalization, alpha=0.85)
        except nx.NetworkXError:
            scores = {n: 1.0 for n in valid_seeds}

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        notes_by_id = self.graph['notes_by_id']

        results = []
        for node, score in ranked[:top_k]:
            if node not in notes_by_id:
                continue
            note = notes_by_id[node]
            results.append({
                'id': node,
                'title': note['title'],
                'type': note['type'],
                'tags': note['tags'],
                'preview': note['preview'][:600],
                'relevance': round(score, 4),
                'is_seed': node in valid_seeds,
            })

        return results

    # ── Path Verbalization ──────────────────────────────────────────────

    def find_bridge_paths(self, resolution=1.0, max_paths=20):
        """Find shortest paths between communities, prioritizing high-surprise paths.

        Returns paths verbalized with node titles and types for LLM interpretation.
        """
        communities = self.communities.get(resolution, [])
        if len(communities) < 2:
            return []

        notes_by_id = self.graph['notes_by_id']
        G_undirected = self.G.to_undirected()

        # For efficiency, sample representative nodes from each community
        # (highest betweenness within community)
        betweenness = self.centrality.get('betweenness', {})

        def _top_nodes(comm, k=3):
            """Get top-k nodes by betweenness from a community."""
            scored = [(n, betweenness.get(n, 0)) for n in comm if n in notes_by_id]
            scored.sort(key=lambda x: -x[1])
            return [n for n, _ in scored[:k]]

        paths = []
        comm_list = list(communities)

        for i in range(len(comm_list)):
            for j in range(i + 1, len(comm_list)):
                comm_a, comm_b = comm_list[i], comm_list[j]
                sources = _top_nodes(comm_a)
                targets = _top_nodes(comm_b)

                for source in sources:
                    for target in targets:
                        try:
                            path = nx.shortest_path(G_undirected, source, target)
                        except nx.NetworkXNoPath:
                            continue

                        if len(path) < 2:
                            continue

                        # Compute average surprise along path
                        total_surprise = 0
                        edge_count = 0
                        for k in range(len(path) - 1):
                            s = self.edge_surprise.get((path[k], path[k+1]),
                                self.edge_surprise.get((path[k+1], path[k]), 0.5))
                            total_surprise += s
                            edge_count += 1

                        avg_surprise = total_surprise / edge_count if edge_count else 0

                        paths.append({
                            'path': path,
                            'length': len(path),
                            'avg_surprise': round(avg_surprise, 3),
                            'community_a': i,
                            'community_b': j,
                        })

        # Deduplicate: keep only the best path per community pair
        best_per_pair = {}
        for p in paths:
            key = (p['community_a'], p['community_b'])
            if key not in best_per_pair or p['avg_surprise'] > best_per_pair[key]['avg_surprise']:
                best_per_pair[key] = p

        paths = sorted(best_per_pair.values(), key=lambda x: -x['avg_surprise'])

        # Verbalize top paths
        verbalized = []
        for p in paths[:max_paths]:
            chain = []
            for nid in p['path']:
                if nid in notes_by_id:
                    note = notes_by_id[nid]
                    chain.append(f"{note['title']} [{note['type'] or 'untyped'}]")
                else:
                    chain.append(f"{nid} [unknown]")

            verbalized.append({
                'chain': chain,
                'chain_text': ' → '.join(chain),
                'path_ids': p['path'],
                'length': p['length'],
                'avg_surprise': p['avg_surprise'],
                'community_a': p['community_a'],
                'community_b': p['community_b'],
            })

        return verbalized

    # ── Centrality Change Tracking ──────────────────────────────────────

    def get_centrality_summary(self):
        """Return a compact summary of centrality metrics for the report.

        Useful for drift detection: was this node a bridge? Is it now isolated?
        """
        betweenness = self.centrality.get('betweenness', {})
        pagerank = self.centrality.get('pagerank', {})
        constraint = self.centrality.get('constraint', {})
        notes_by_id = self.graph['notes_by_id']

        summary = []
        for nid in self.G.nodes():
            if nid not in notes_by_id:
                continue
            note = notes_by_id[nid]
            b = betweenness.get(nid, 0)
            pr = pagerank.get(nid, 0)
            c = constraint.get(nid)

            # Flag structurally interesting nodes
            role = None
            if b > 0.05:
                role = 'bridge'
            elif pr > 2.0 / len(self.G.nodes()) if len(self.G.nodes()) > 0 else False:
                role = 'hub'
            elif c is not None and c < 0.5:
                role = 'broker'

            if role:
                summary.append({
                    'id': nid,
                    'title': note['title'],
                    'role': role,
                    'betweenness': round(b, 4),
                    'pagerank': round(pr, 4),
                    'constraint': round(c, 3) if c is not None else None,
                })

        return summary

    # ── Cross-Community Similarity ──────────────────────────────────────

    def get_cross_community_pairs(self, resolution=1.0, top_k=10):
        """Find most similar note pairs that span different communities.

        More efficient than the old O(N² per cluster pair) approach:
        only computes similarity for edges + near-edges.
        """
        cmap = self.community_map.get(resolution, {})
        embeddings = self.graph['embeddings']
        notes_by_id = self.graph['notes_by_id']

        if not cmap:
            return []

        # Use edge surprise to find cross-community bridges efficiently
        cross_pairs = []
        for (u, v), surprise in self.edge_surprise.items():
            cu = cmap.get(u)
            cv = cmap.get(v)
            if cu is not None and cv is not None and cu != cv:
                if u in embeddings and v in embeddings:
                    sim = 1.0 - surprise  # surprise = 1 - similarity
                    cross_pairs.append({
                        'note_a': u,
                        'note_b': v,
                        'title_a': notes_by_id[u]['title'] if u in notes_by_id else u,
                        'title_b': notes_by_id[v]['title'] if v in notes_by_id else v,
                        'similarity': round(sim, 3),
                        'surprise': round(surprise, 3),
                        'community_a': cu,
                        'community_b': cv,
                    })

        # Also check high-similarity unconnected cross-community pairs
        # but limit search to nodes near community boundaries
        boundary_nodes = set()
        G_undirected = self.G.to_undirected()
        for node in self.G.nodes():
            cn = cmap.get(node)
            if cn is None:
                continue
            for neighbor in G_undirected.neighbors(node):
                if cmap.get(neighbor) != cn:
                    boundary_nodes.add(node)
                    break

        boundary_list = [n for n in boundary_nodes if n in embeddings]
        for i in range(len(boundary_list)):
            for j in range(i + 1, len(boundary_list)):
                a, b = boundary_list[i], boundary_list[j]
                ca, cb = cmap.get(a), cmap.get(b)
                if ca == cb:
                    continue
                # Skip if already an edge (covered above)
                if self.G.has_edge(a, b) or self.G.has_edge(b, a):
                    continue
                sim = _cosine_similarity(embeddings[a], embeddings[b])
                if sim > 0.35:
                    cross_pairs.append({
                        'note_a': a,
                        'note_b': b,
                        'title_a': notes_by_id[a]['title'] if a in notes_by_id else a,
                        'title_b': notes_by_id[b]['title'] if b in notes_by_id else b,
                        'similarity': round(sim, 3),
                        'surprise': round(1.0 - sim, 3),
                        'community_a': ca,
                        'community_b': cb,
                    })

        cross_pairs.sort(key=lambda x: -x['similarity'])
        return cross_pairs[:top_k]


# ── Utility ──────────────────────────────────────────────────────────────

def _cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
