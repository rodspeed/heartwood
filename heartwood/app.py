import sys
import os
import re
import json
import math
try:
    import webview
except ImportError:
    webview = None  # server-only mode (no desktop GUI)
import yaml
import time
import datetime
import threading
from pathlib import Path
from collections import Counter
from storage_fs import FileSystemBackend


def _app_dir():
    """Root directory for user-editable data (notes, .graph, ontology, reports).

    In PyInstaller --onedir mode, __file__ resolves inside _internal/, but
    user data lives next to the .exe.  In dev mode, everything lives next to
    the script.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource(relative_path):
    """Resolve path for bundled assets (index.html, context.html).

    PyInstaller extracts these into _MEIPASS; in dev they sit next to the script.
    """
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def notes_dir():
    """Always resolve notes/ relative to the app directory."""
    return os.path.join(_app_dir(), 'notes')


WIKILINK_RE = re.compile(r'\[\[([^\]]+)\]\]')


def make_preview(body: str, max_len: int = 120) -> str:
    """Generate a preview from note body, truncating at word boundary."""
    lines = [l for l in body.split('\n') if l.strip() and not l.strip().startswith('#')]
    text = ' '.join(lines)
    if len(text) <= max_len:
        return text.strip()
    # Truncate at last space before max_len
    truncated = text[:max_len].rsplit(' ', 1)[0]
    return truncated.strip() + '...'


# ===== Sentence Embedding Semantic Similarity =====

_embedding_model = None


def _get_embedding_model():
    """Lazy-load the sentence-transformer model on first use."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            return None
    return _embedding_model


class EmbeddingCache:
    """Cache note embeddings to disk, re-embedding only when notes change.
    Also stores chunk-level embeddings for RAG and creation-time embeddings for drift detection."""

    def __init__(self, cache_path):
        self._cache_path = cache_path
        self._cache = {}  # {note_id: {'mtime': float, 'vector': [...], 'chunks': [...], 'origin_vector': [...]}}
        self._load()

    def _load(self):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def save(self):
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f)

    def get_embeddings(self, notes):
        """Get note-level embeddings, using cache where possible.
        Each note must have: id, title, tags, body, mtime.
        Returns {note_id: list_of_floats}."""
        result = {}
        to_embed = []

        for note in notes:
            nid = note['id']
            cached = self._cache.get(nid)
            if cached and cached.get('mtime') == note.get('mtime'):
                result[nid] = cached['vector']
            else:
                to_embed.append(note)

        if to_embed:
            model = _get_embedding_model()
            if model is None:
                return result
            texts = [_note_to_text(n) for n in to_embed]
            vectors = model.encode(texts, show_progress_bar=False).tolist()
            for note, vec in zip(to_embed, vectors):
                nid = note['id']
                entry = self._cache.get(nid, {})
                # Preserve origin_vector if it exists; set it on first embedding
                if 'origin_vector' not in entry:
                    entry['origin_vector'] = vec
                entry['mtime'] = note.get('mtime')
                entry['vector'] = vec
                # Recompute chunks
                entry['chunks'] = self._embed_chunks(note)
                self._cache[nid] = entry
                result[nid] = vec
            self.save()

        return result

    def _embed_chunks(self, note):
        """Split a note into semantic chunks and embed each one.
        Returns list of {'text': str, 'vector': [...]}."""
        chunks = _chunk_note(note)
        if not chunks:
            return []
        model = _get_embedding_model()
        if model is None:
            return []
        vectors = model.encode(chunks, show_progress_bar=False).tolist()
        return [{'text': t, 'vector': v} for t, v in zip(chunks, vectors)]

    def get_chunks(self, note_id):
        """Get cached chunks for a note. Returns list of {'text': str, 'vector': [...]}."""
        cached = self._cache.get(note_id, {})
        return cached.get('chunks', [])

    def get_all_chunks(self, note_ids=None):
        """Get all chunks across all (or specified) notes.
        Returns list of {'note_id': str, 'text': str, 'vector': [...]}."""
        result = []
        for nid, entry in self._cache.items():
            if note_ids and nid not in note_ids:
                continue
            for chunk in entry.get('chunks', []):
                result.append({'note_id': nid, 'text': chunk['text'], 'vector': chunk['vector']})
        return result

    def get_drift_scores(self):
        """Compute drift between each note's origin embedding and its current embedding.
        Returns {note_id: float} where higher = more drift (0.0 = unchanged)."""
        result = {}
        for nid, entry in self._cache.items():
            origin = entry.get('origin_vector')
            current = entry.get('vector')
            if origin and current:
                sim = _cosine_similarity(origin, current)
                result[nid] = round(1.0 - sim, 4)
        return result

    def embed_query(self, text):
        """Embed a query string. Returns a list of floats."""
        model = _get_embedding_model()
        if model is None:
            return []
        return model.encode(text, show_progress_bar=False).tolist()

    def prune(self, valid_ids):
        """Remove cached entries for notes that no longer exist."""
        stale = [k for k in self._cache if k not in valid_ids]
        for k in stale:
            del self._cache[k]
        if stale:
            self.save()


def _chunk_note(note):
    """Split a note into paragraph-level chunks for RAG retrieval.
    Each chunk includes the note title as prefix for context."""
    body = note.get('body', '')
    if not body.strip():
        return []

    title = note.get('title', '')
    prefix = f"{title}: "

    # Split on double newlines (paragraph boundaries) or markdown headings
    raw_chunks = re.split(r'\n\s*\n|(?=^#{1,3}\s)', body, flags=re.MULTILINE)

    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        # Skip very short chunks (headers alone, blank lines)
        if len(chunk) < 30:
            continue
        # Truncate very long chunks
        chunks.append(prefix + chunk[:500])

    # If note is too short to chunk, use the whole thing
    if not chunks and len(body.strip()) >= 10:
        chunks = [prefix + body[:500]]

    return chunks


def _note_to_text(note):
    """Build embedding input text from a note: title + tags + body (truncated)."""
    parts = [note['title']]
    if note.get('tags'):
        parts.append(', '.join(note['tags']))
    body = note.get('body', '')
    # MiniLM has ~256 token limit; front-load important content
    parts.append(body[:1500])
    return '. '.join(parts)


def _cosine_similarity(v1, v2):
    """Cosine similarity between two dense vectors (lists)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def find_semantic_edges(notes, threshold=0.30, max_per_node=5, embedding_cache=None):
    """Find semantically similar note pairs using sentence embeddings.
    Uses numpy vectorized cosine similarity for O(n²) speed with minimal constant factor.
    Returns list of {source, target, type:'semantic', score}."""
    if len(notes) < 2 or embedding_cache is None:
        return []

    embeddings = embedding_cache.get_embeddings(notes)
    ids = [n['id'] for n in notes]

    # Filter to ids that have embeddings
    valid = [(i, ids[i]) for i in range(len(ids)) if ids[i] in embeddings]
    if len(valid) < 2:
        return []

    try:
        import numpy as np
        return _find_semantic_edges_numpy(valid, embeddings, threshold, max_per_node)
    except ImportError:
        return _find_semantic_edges_fallback(valid, embeddings, threshold, max_per_node)


def _find_semantic_edges_numpy(valid, embeddings, threshold, max_per_node):
    """Vectorized semantic edge computation using numpy."""
    import numpy as np

    indices, id_list = zip(*valid)
    # Build matrix: rows = notes, cols = embedding dimensions
    matrix = np.array([embeddings[nid] for nid in id_list], dtype=np.float32)

    # Normalize rows to unit vectors
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    # Cosine similarity = dot product of normalized vectors (full matrix at once)
    sim_matrix = matrix @ matrix.T

    # Zero out diagonal (self-similarity)
    np.fill_diagonal(sim_matrix, 0.0)

    edges = []
    seen = set()
    n = len(id_list)

    for i in range(n):
        row = sim_matrix[i]
        # Find indices above threshold
        above = np.where(row >= threshold)[0]
        if len(above) == 0:
            continue
        # Sort by similarity descending, take top N
        top = above[np.argsort(-row[above])[:max_per_node]]
        for j in top:
            src, tgt = (id_list[i], id_list[j]) if i < j else (id_list[j], id_list[i])
            key = (src, tgt)
            if key not in seen:
                seen.add(key)
                edges.append({
                    'source': src,
                    'target': tgt,
                    'type': 'semantic',
                    'score': round(float(row[j]), 3),
                })

    return edges


def _find_semantic_edges_fallback(valid, embeddings, threshold, max_per_node):
    """Pure-Python fallback when numpy is not available."""
    indices, id_list = zip(*valid)
    edges = []
    seen = set()

    for i in range(len(id_list)):
        scores = []
        for j in range(len(id_list)):
            if i == j:
                continue
            sim = _cosine_similarity(embeddings[id_list[i]], embeddings[id_list[j]])
            if sim >= threshold:
                scores.append((j, sim))

        scores.sort(key=lambda x: -x[1])
        for j, sim in scores[:max_per_node]:
            src, tgt = (id_list[i], id_list[j]) if i < j else (id_list[j], id_list[i])
            key = (src, tgt)
            if key not in seen:
                seen.add(key)
                edges.append({
                    'source': src,
                    'target': tgt,
                    'type': 'semantic',
                    'score': round(sim, 3),
                })

    return edges


# Keep tokenize for keyword boosting in chat
TOKEN_RE = re.compile(r'[a-z][a-z0-9]{2,}')
STOP_WORDS = frozenset(
    'a an the is are was were be been being have has had do does did will would '
    'shall should may might can could of in to for on with at by from as into '
    'through during before after above below between out off over under again '
    'further then once here there when where why how all both each few more most '
    'other some such no nor not only own same so than too very and but if or '
    'because until while about this that these those it its he she they them '
    'their what which who whom i me my we our you your just also like get got '
    'one two make use used using note notes native source'.split()
)


def tokenize(text):
    """Extract lowercase tokens, removing stop words and short tokens."""
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS]


def title_to_filename(title):
    """Convert a note title to a filename slug."""
    slug = re.sub(r'[^\w\s-]', '', title.lower().strip())
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug + '.md'


def filename_to_id(filename):
    """Convert a filename to a node ID (strip .md)."""
    return Path(filename).stem


def title_from_capture_text(text, max_words=5):
    """Extract a short title from raw capture text.
    Takes the first N meaningful words, stripping URLs and punctuation."""
    # Remove URLs
    cleaned = re.sub(r'https?://\S+', '', text)
    # Remove markdown artifacts
    cleaned = re.sub(r'[#*_`>\[\]]', '', cleaned)
    # Split into words, keep only meaningful ones
    words = [w for w in cleaned.split() if len(w) > 1]
    if not words:
        return ''
    title_words = words[:max_words]
    title = ' '.join(title_words)
    # Capitalize first letter, strip trailing punctuation
    title = title[0].upper() + title[1:] if title else ''
    title = title.rstrip('.,;:!?-')
    return title


def parse_frontmatter(content):
    """Extract YAML frontmatter and body from markdown content."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return meta, body
            except yaml.YAMLError:
                pass
    return {}, content


def extract_wikilinks(content):
    """Extract all [[wikilink]] targets from markdown content.
    Strips relationship prefix from typed links (e.g. [[rel::Target]] -> Target)."""
    raw = WIKILINK_RE.findall(content)
    return [link.split('::', 1)[1] if '::' in link else link for link in raw]


def extract_typed_wikilinks(content):
    """Extract wikilinks with relationship type info.
    Returns list of (rel_type_or_None, target_title) tuples."""
    raw = WIKILINK_RE.findall(content)
    result = []
    for link in raw:
        if '::' in link:
            parts = link.split('::', 1)
            result.append((parts[0].strip(), parts[1].strip()))
        else:
            result.append((None, link))
    return result


def load_ontology():
    """Load ontology.yaml from the heartwood directory. Returns empty dict if missing."""
    path = os.path.join(_app_dir(), 'ontology.yaml')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def infer_type(meta, ontology=None):
    """Infer note type from frontmatter. Explicit 'type' field wins;
    falls back to tag-based inference from ontology rules."""
    explicit = meta.get('type')
    if explicit:
        return explicit
    tags = set(meta.get('tags', []))
    if not tags:
        return None
    # Use inference rules from ontology if available
    if ontology and 'inference' in ontology:
        for rule in ontology['inference']:
            if rule['tag'] in tags:
                return rule['type']
    return None


class Api:
    """Python API exposed to the JavaScript frontend via pywebview."""

    def __init__(self, storage=None):
        if storage is not None:
            # External backend (e.g. Supabase) — skip filesystem setup
            self._storage = storage
            self._notes_path = None
            self._layout_path = None
            # Use a per-user temp dir for embedding cache (pgvector migration later)
            import tempfile
            user_id = getattr(storage, '_user_id', 'default')
            cache_dir = os.path.join(tempfile.gettempdir(), 'heartwood_cache', user_id)
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, 'embeddings.json')
        else:
            # Desktop mode — filesystem backend
            self._notes_path = notes_dir()
            self._layout_path = os.path.join(self._notes_path, '..', '.graph')
            os.makedirs(self._notes_path, exist_ok=True)
            os.makedirs(self._layout_path, exist_ok=True)
            self._storage = FileSystemBackend(
                self._notes_path, self._layout_path,
                os.path.join(_app_dir(), 'ontology.yaml')
            )
            cache_path = os.path.join(self._layout_path, 'embeddings.json')

        self._semantic_threshold = 0.30
        self._embedding_cache = EmbeddingCache(cache_path)
        self._semantic_edges_ready = False
        self._semantic_edges = []
        self._embedding_thread = None

    def scan_notes(self):
        """Scan all markdown files and build the graph data structure."""
        notes_list = self._storage.list_notes()
        ontology = self._storage.load_ontology()

        nodes = []
        edges = []
        title_to_id = {}

        # First pass: collect all notes and build title->id mapping
        # Also map by note ID (filename without extension) for Obsidian-style [[links]]
        for n in notes_list:
            node_type = infer_type({'type': n.type, 'tags': n.tags}, ontology)
            title_to_id[n.title.lower()] = n.id
            title_to_id[n.id.lower()] = n.id
            nodes.append({
                'id': n.id, 'title': n.title, 'tags': n.tags,
                'type': node_type, 'created': n.created,
                'file': n.file,
                'preview': n.preview,
            })

        # Second pass: extract edges from wikilinks
        for n in notes_list:
            typed_links = extract_typed_wikilinks(n.raw_content)
            seen_edges = set()
            for rel_type, link_title in typed_links:
                target_id = title_to_id.get(link_title.lower())
                if target_id and target_id != n.id:
                    edge_key = (n.id, target_id)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            'source': n.id, 'target': target_id,
                            'type': 'wikilink', 'rel': rel_type,
                        })

        # Load manual connections
        for edge in self._storage.load_connections():
            edge['type'] = 'manual'
            edges.append(edge)

        # Soul node gravity fallback — if the soul node has no structural edges
        # (e.g., pre-anchor users), add invisible pull edges to keep it centered
        soul_id = self._storage.get_config('soul_node_id')
        if soul_id and soul_id in {n['id'] for n in nodes}:
            has_structural = any(
                e['type'] not in ('semantic', 'soul') and (e['source'] == soul_id or e['target'] == soul_id)
                for e in edges
            )
            if not has_structural:
                for n in nodes:
                    if n['id'] != soul_id:
                        edges.append({
                            'source': soul_id, 'target': n['id'],
                            'type': 'soul', 'rel': None,
                        })

        # Compute semantic edges — use cached embeddings if available, compute in background if not
        existing_pairs = set()
        for e in edges:
            existing_pairs.add((e['source'], e['target']))
            existing_pairs.add((e['target'], e['source']))

        note_data = [{'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime}
                     for n in notes_list]

        valid_ids = set(n['id'] for n in nodes)

        # Check if all notes have cached embeddings (no model load needed)
        all_cached = len(note_data) > 0 and all(
            self._embedding_cache._cache.get(n['id'], {}).get('mtime') == n.get('mtime')
            for n in note_data
        )

        if all_cached:
            # Fast path: compute synchronously from cache (no model load)
            try:
                semantic_edges = find_semantic_edges(
                    note_data, threshold=self._semantic_threshold,
                    embedding_cache=self._embedding_cache
                )
                self._embedding_cache.prune(valid_ids)
                self._semantic_edges = semantic_edges
                self._semantic_edges_ready = True
                for se in semantic_edges:
                    pair = (se['source'], se['target'])
                    if pair not in existing_pairs:
                        edges.append(se)
                        existing_pairs.add(pair)
                        existing_pairs.add((pair[1], pair[0]))
            except Exception as e:
                print(f'[embeddings] Cache-path error: {e}')
        else:
            # Slow path: model needs to load — compute in background
            def _compute_embeddings():
                try:
                    semantic_edges = find_semantic_edges(
                        note_data, threshold=self._semantic_threshold,
                        embedding_cache=self._embedding_cache
                    )
                    self._embedding_cache.prune(valid_ids)
                    self._semantic_edges = semantic_edges
                    self._semantic_edges_ready = True
                    print(f'[embeddings] Ready: {len(semantic_edges)} semantic edges')
                except Exception as e:
                    print(f'[embeddings] Error: {e}')
                    self._semantic_edges_ready = True
                    self._semantic_edges = []

            if self._embedding_thread is None or not self._embedding_thread.is_alive():
                self._semantic_edges_ready = False
                self._embedding_thread = threading.Thread(target=_compute_embeddings, daemon=True)
                self._embedding_thread.start()

        return {'nodes': nodes, 'edges': edges}

    def check_embeddings_available(self):
        """Check if sentence-transformers is installed."""
        try:
            import sentence_transformers  # noqa: F401
            return {'available': True}
        except ImportError:
            return {'available': False}

    def get_semantic_edges(self):
        """Poll endpoint: returns semantic edges once embedding computation is done."""
        if not self._semantic_edges_ready:
            return {'ready': False}
        return {'ready': True, 'edges': self._semantic_edges}

    def read_note(self, note_id):
        """Read a specific note's full content."""
        note = self._storage.read_note(note_id)
        if note is None:
            return {'error': 'Note not found'}
        ontology = self._storage.load_ontology()
        return {
            'id': note.id,
            'title': note.title,
            'tags': note.tags,
            'type': infer_type({'type': note.type, 'tags': note.tags}, ontology),
            'created': note.created,
            'content': note.raw_content,
            'body': note.body,
            'synced_from': note.synced_from,
        }

    def save_note(self, note_id, content):
        """Save note content (full markdown including frontmatter).
        If the note was synced from an external source, write changes back."""
        self._storage.save_note(note_id, content)

        # Reverse sync: write body back to original source if synced_from exists
        meta, body = parse_frontmatter(content)
        synced_from = meta.get('synced_from', '')
        if synced_from:
            self._reverse_sync(synced_from, body)

        # Compute auto-link suggestions for the saved note
        auto_links = self._get_auto_link_suggestions(note_id, content)
        return {'success': True, 'auto_links': auto_links}

    def _get_auto_link_suggestions(self, note_id, content):
        """After saving, find top notes this note should link to but doesn't."""
        meta, body = parse_frontmatter(content)
        title = meta.get('title', note_id)
        tags = meta.get('tags', [])
        existing_links = set(link.lower() for link in extract_wikilinks(content))

        # Get embedding for this note
        note = self._storage.read_note(note_id)
        mtime = note.mtime if note else 0
        this_note = [{'id': note_id, 'title': title, 'tags': tags, 'body': body, 'mtime': mtime}]

        # Gather all other notes
        all_notes = list(this_note)
        other_info = {}
        for n in self._storage.list_notes():
            if n.id == note_id:
                continue
            if n.title.lower() in existing_links:
                continue
            all_notes.append({'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime})
            other_info[n.id] = {'title': n.title, 'tags': n.tags}

        embeddings = self._embedding_cache.get_embeddings(all_notes)
        if note_id not in embeddings:
            return []

        current_vec = embeddings[note_id]
        scored = []
        for fid, info in other_info.items():
            if fid not in embeddings:
                continue
            sim = _cosine_similarity(current_vec, embeddings[fid])
            if sim >= 0.35:
                scored.append({
                    'id': fid,
                    'title': info['title'],
                    'score': round(sim, 3),
                })

        scored.sort(key=lambda x: -x['score'])
        return scored[:5]

    def _reverse_sync(self, synced_from, new_body):
        """Write edited body back to the original source file, preserving its frontmatter."""
        try:
            # Resolve the synced_from path relative to the Claude Code directory
            claude_code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Handle paths that start with ../../ (relative to heartwood dir)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            source_path = os.path.normpath(os.path.join(script_dir, synced_from))

            if not os.path.exists(source_path):
                # Try resolving relative to Claude Code dir instead
                source_path = os.path.normpath(os.path.join(claude_code_dir, synced_from))

            if not os.path.exists(source_path):
                return  # Source file doesn't exist, skip

            # Read the original file to preserve its frontmatter
            with open(source_path, 'r', encoding='utf-8') as fh:
                original = fh.read()

            orig_meta, _ = parse_frontmatter(original)

            if orig_meta:
                # Rebuild with original frontmatter + new body
                # Strip the "Native source:" footer line that sync.py adds
                clean_body = new_body
                if '\n---\n*Native source:' in clean_body:
                    clean_body = clean_body[:clean_body.rindex('\n---\n*Native source:')].rstrip()

                fm_lines = ['---']
                for key, val in orig_meta.items():
                    fm_lines.append(f'{key}: {val}')
                fm_lines.append('---')
                new_content = '\n'.join(fm_lines) + '\n\n' + clean_body + '\n'
            else:
                new_content = new_body + '\n'

            with open(source_path, 'w', encoding='utf-8') as fh:
                fh.write(new_content)
        except Exception as e:
            print(f"[reverse-sync] Error writing back to {synced_from}: {e}")

    def create_note(self, title):
        """Create a new empty note with the given title."""
        filename = title_to_filename(title)
        node_id = filename_to_id(filename)

        today = time.strftime('%Y-%m-%d')
        content = f"""---
title: {title}
tags: []
created: {today}
---

# {title}

"""
        return self._storage.create_note(node_id, filename, content)

    def delete_note(self, note_id):
        """Delete a note by ID."""
        return self._storage.delete_note(note_id)

    def rename_note(self, old_id, new_title):
        """Rename a note — updates filename and frontmatter title."""
        note = self._storage.read_note(old_id)
        if note is None:
            return {'error': 'Note not found'}

        meta, body = parse_frontmatter(note.raw_content)
        meta['title'] = new_title

        new_filename = title_to_filename(new_title)
        new_id = filename_to_id(new_filename)

        # Rebuild frontmatter
        new_content = f"---\ntitle: {new_title}\ntags: {json.dumps(meta.get('tags', []))}\ncreated: {meta.get('created', '')}\n---\n\n{body}"
        return self._storage.rename_note(old_id, new_id, new_filename, new_content)

    def add_connection(self, source_id, target_id):
        """Add a manual connection between two notes."""
        connections = self._storage.load_connections()

        # Check for duplicate
        for c in connections:
            if c['source'] == source_id and c['target'] == target_id:
                return {'error': 'Connection already exists'}

        connections.append({'source': source_id, 'target': target_id})
        return self._storage.save_connections(connections)

    def remove_connection(self, source_id, target_id):
        """Remove a manual connection and/or wikilink between two notes."""
        import re

        # Remove manual connection
        connections = self._storage.load_connections()
        connections = [c for c in connections
                       if not (c['source'] == source_id and c['target'] == target_id)
                       and not (c['source'] == target_id and c['target'] == source_id)]
        self._storage.save_connections(connections)

        # Also remove wikilinks from source note → target note
        target_note = self._storage.read_note(target_id) if self._storage.note_exists(target_id) else None
        source_note = self._storage.read_note(source_id) if self._storage.note_exists(source_id) else None

        if target_note and source_note:
            # Remove [[target title]] from source content
            target_title = target_note.get('title', '')
            if target_title:
                source_content = source_note.get('content', '')
                # Match [[target_title]] and [[rel_type::target_title]]
                pattern = r'\[\[(?:[a-z_]+::)?' + re.escape(target_title) + r'\]\]'
                new_content = re.sub(pattern, '', source_content, flags=re.IGNORECASE)
                # Clean up leftover empty list items and blank lines
                new_content = re.sub(r'\n- \s*\n', '\n', new_content)
                new_content = re.sub(r'\n{3,}', '\n\n', new_content)
                if new_content != source_content:
                    self._storage.save_note(source_id, new_content)

        return {'success': True}

    def get_note_edges(self, note_id):
        """Return all edges involving a note, with type and neighbor info."""
        graph = self.scan_notes()
        nodes_by_id = {n['id']: n for n in graph['nodes']}
        edges = []
        seen = set()
        for e in graph['edges']:
            if e.get('type') == 'soul':
                continue  # skip invisible soul gravity edges
            if e['source'] == note_id or e['target'] == note_id:
                neighbor_id = e['target'] if e['source'] == note_id else e['source']
                edge_key = (min(note_id, neighbor_id), max(note_id, neighbor_id), e.get('type'))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                neighbor = nodes_by_id.get(neighbor_id, {})
                edges.append({
                    'neighbor_id': neighbor_id,
                    'neighbor_title': neighbor.get('title', neighbor_id),
                    'type': e.get('type', 'unknown'),
                    'score': e.get('score'),
                    'rel': e.get('rel'),
                    'removable': e.get('type') in ('manual', 'wikilink'),
                })
        # Sort: manual first, then wikilink, then semantic
        type_order = {'manual': 0, 'wikilink': 1, 'semantic': 2}
        edges.sort(key=lambda x: (type_order.get(x['type'], 9), x['neighbor_title']))
        return edges

    def save_layout(self, positions):
        """Save node positions from the graph layout."""
        return self._storage.save_layout(positions)

    def load_layout(self):
        """Load saved node positions."""
        return self._storage.load_layout()

    def get_all_tags(self):
        """Get all unique tags across all notes."""
        tags = set()
        for n in self._storage.list_notes():
            for tag in n.tags:
                tags.add(tag)
        return sorted(list(tags))

    def search_notes(self, query):
        """Search notes by title and content. Returns snippets for content matches."""
        query_lower = query.lower()
        results = []
        for n in self._storage.list_notes():
            title_match = query_lower in n.title.lower()
            tag_match = any(query_lower in t.lower() for t in n.tags)
            body_lower = n.body.lower()
            body_match = query_lower in body_lower

            if title_match or tag_match or body_match:
                snippet = ''
                if body_match:
                    idx = body_lower.index(query_lower)
                    start = max(0, idx - 40)
                    end = min(len(n.body), idx + len(query) + 60)
                    raw = n.body[start:end].strip()
                    raw = ' '.join(raw.split())
                    if start > 0:
                        raw = '...' + raw
                    if end < len(n.body):
                        raw = raw + '...'
                    snippet = raw

                results.append({
                    'id': n.id,
                    'title': n.title,
                    'tags': n.tags,
                    'snippet': snippet,
                    'match_type': 'title' if title_match else ('tag' if tag_match else 'content'),
                })
        return results

    def update_semantic_threshold(self, threshold):
        """Update the similarity threshold for semantic edges."""
        self._semantic_threshold = max(0.10, min(0.80, float(threshold)))
        return {'success': True, 'threshold': self._semantic_threshold}

    def semantic_search(self, query, top_k=10):
        """Search notes by embedding similarity to a query string."""
        q_vec = self._embedding_cache.embed_query(query)

        notes_list = self._storage.list_notes()
        all_notes = [{'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime}
                     for n in notes_list]

        embeddings = self._embedding_cache.get_embeddings(all_notes)

        scored = []
        for note in all_notes:
            nid = note['id']
            if nid not in embeddings:
                continue
            sim = _cosine_similarity(q_vec, embeddings[nid])
            preview = make_preview(note['body'])
            scored.append({
                'id': nid,
                'title': note['title'],
                'tags': note['tags'],
                'snippet': preview,
                'score': round(sim, 3),
                'match_type': 'semantic',
            })

        scored.sort(key=lambda x: -x['score'])
        return scored[:top_k]

    def save_constellations(self, data):
        """Save constellation names."""
        return self._storage.save_constellations(data)

    def load_constellations(self):
        """Load saved constellation names."""
        return self._storage.load_constellations()

    # ===== Graph Chat =====

    def chat_with_graph(self, question, context_note_ids=None):
        """Answer a question using notes as context. Sends relevant notes to Claude."""
        api_key = self._get_api_key()
        if not api_key:
            return {'error': 'No Anthropic API key configured. Click the key icon to set one.'}

        # Gather context: if specific notes provided, use those; otherwise find relevant ones
        notes_context = []
        if context_note_ids:
            for nid in context_note_ids[:10]:
                note = self._storage.read_note(nid)
                if note:
                    notes_context.append({
                        'title': note.title,
                        'tags': note.tags,
                        'body': note.body,
                    })
        else:
            # RAG: retrieve the most relevant chunks across all notes
            q_vec = self._embedding_cache.embed_query(question)
            all_chunks = self._embedding_cache.get_all_chunks()

            # Score each chunk by similarity to the question
            scored_chunks = []
            for chunk in all_chunks:
                sim = _cosine_similarity(q_vec, chunk['vector'])
                scored_chunks.append((sim, chunk))

            # Also do substring search for exact-match boosting
            results = self.search_notes(question)
            search_ids = set(r['id'] for r in results)

            scored_chunks.sort(key=lambda x: -x[0])

            # Take top chunks, ensuring we pull from diverse notes
            seen_notes = {}  # note_id -> count of chunks included
            selected_chunks = []
            for sim, chunk in scored_chunks:
                nid = chunk['note_id']
                note_count = seen_notes.get(nid, 0)
                if note_count >= 3:  # max 3 chunks per note
                    continue
                # Boost chunks from substring-matched notes
                effective_sim = sim + (0.10 if nid in search_ids else 0)
                selected_chunks.append((effective_sim, chunk))
                seen_notes[nid] = note_count + 1
                if len(selected_chunks) >= 30:
                    break

            # Build note context from selected chunks, grouped by note
            note_chunks = {}  # note_id -> [chunk_texts]
            note_titles = {}
            note_tags = {}
            for sim, chunk in selected_chunks:
                nid = chunk['note_id']
                if nid not in note_chunks:
                    note_chunks[nid] = []
                    note = self._storage.read_note(nid)
                    if note:
                        note_titles[nid] = note.title
                        note_tags[nid] = note.tags
                note_chunks[nid].append(chunk['text'])

            # Convert to notes_context format for the prompt
            for nid, chunks in note_chunks.items():
                notes_context.append({
                    'title': note_titles.get(nid, nid),
                    'tags': note_tags.get(nid, []),
                    'body': '\n\n'.join(chunks),
                })

        if not notes_context:
            return {'answer': "I couldn't find any notes relevant to that question. Try rephrasing or asking about a topic that exists in your graph."}

        # Build context string
        context_parts = []
        total_chars = 0
        for note in notes_context:
            part = f"## {note['title']}\nTags: {', '.join(note['tags'])}\n\n{note['body']}"
            if total_chars + len(part) > 40000:
                break
            context_parts.append(part)
            total_chars += len(part)

        context_str = '\n\n---\n\n'.join(context_parts)

        prompt = f"""You are Heartwood, an AI assistant that answers questions using a personal knowledge graph. You have access to the user's notes below. Answer the question using ONLY information from these notes. If the notes don't contain enough information, say so honestly.

When referencing specific notes, mention them by title so the user can navigate to them.

Be concise and direct. Use markdown formatting.

## Knowledge Graph Notes

{context_str}

## Question

{question}"""

        import http.client
        import ssl

        payload = json.dumps({
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 2048,
            'messages': [{'role': 'user', 'content': prompt}]
        }).encode('utf-8')

        try:
            conn = http.client.HTTPSConnection('api.anthropic.com', timeout=60,
                                                context=ssl.create_default_context())
            conn.request('POST', '/v1/messages', body=payload, headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            })
            resp = conn.getresponse()
            raw = resp.read().decode('utf-8')
            conn.close()

            if resp.status != 200:
                return {'error': f'Claude API returned HTTP {resp.status}'}

            result = json.loads(raw)
            if result.get('content') and len(result['content']) > 0:
                answer = result['content'][0]['text']
                # Return answer + list of note titles used as context
                source_titles = [n['title'] for n in notes_context]
                return {'answer': answer, 'sources': source_titles}
            return {'error': 'Empty response from Claude'}
        except Exception as e:
            return {'error': f'API error: {type(e).__name__}: {e}'}

    # ===== Context Packet =====

    def generate_context_packet(self, query, depth='standard'):
        """Generate a context packet for pasting into an AI conversation.

        Retrieves relevant notes via semantic search, expands wikilinks one hop,
        and synthesizes a compact briefing via Haiku.

        depth: 'standard' (semantic + wikilinks + synthesis)
               'deep' (also pulls latest reasoning report)
        """
        api_key = self._get_api_key()
        if not api_key:
            return {'error': 'No Anthropic API key configured.'}

        # 1. Semantic search — top 10 most relevant notes
        semantic_hits = self.semantic_search(query, top_k=10)
        if not semantic_hits:
            return {'error': 'No notes found relevant to that query.'}

        # 2. Wikilink expansion — for each hit, pull directly linked notes (1 hop)
        hit_ids = set(h['id'] for h in semantic_hits)
        notes_list = self._storage.list_notes()
        notes_by_id = {n.id: n for n in notes_list}
        title_to_id = {}
        for n in notes_list:
            title_to_id[n.title.lower()] = n.id
            title_to_id[n.id.lower()] = n.id

        expanded_ids = set()
        for nid in hit_ids:
            note = notes_by_id.get(nid)
            if not note:
                continue
            links = extract_wikilinks(note.raw_content)
            for link in links:
                target_id = title_to_id.get(link.lower())
                if target_id and target_id not in hit_ids:
                    expanded_ids.add(target_id)

        # Also check backlinks — notes that link TO the hits
        for n in notes_list:
            if n.id in hit_ids or n.id in expanded_ids:
                continue
            links = extract_wikilinks(n.raw_content)
            for link in links:
                target_id = title_to_id.get(link.lower())
                if target_id and target_id in hit_ids:
                    expanded_ids.add(n.id)
                    break  # one match is enough

        # 3. Deduplicate and rank — semantic hits first, then expanded (capped)
        # Limit expansion to avoid bloating the packet
        expanded_ids = set(list(expanded_ids)[:8])
        all_ids = list(hit_ids) + list(expanded_ids)

        # Build note content for synthesis
        note_contents = []
        sources = []
        for nid in all_ids:
            note = notes_by_id.get(nid)
            if not note:
                continue
            is_primary = nid in hit_ids
            score = next((h['score'] for h in semantic_hits if h['id'] == nid), 0)
            note_contents.append({
                'id': nid,
                'title': note.title,
                'tags': note.tags,
                'body': note.body[:3000],  # cap per-note length
                'primary': is_primary,
                'score': score,
            })
            sources.append({
                'id': nid,
                'title': note.title,
                'primary': is_primary,
            })

        if not note_contents:
            return {'error': 'Could not load any relevant notes.'}

        # 4. Build synthesis prompt
        notes_text = []
        for nc in note_contents:
            label = f" (similarity: {nc['score']:.2f})" if nc['primary'] else " (linked)"
            tags_str = ', '.join(nc['tags']) if nc['tags'] else 'none'
            notes_text.append(
                f"### {nc['title']}{label}\n"
                f"Tags: {tags_str}\n\n"
                f"{nc['body']}"
            )
        notes_block = '\n\n---\n\n'.join(notes_text)

        # If deep mode, include reasoning report
        reasoning_section = ''
        if depth == 'deep':
            try:
                report_text = None
                # Try Supabase first
                if hasattr(self._storage, '_client'):
                    resp = self._storage._client.table('reasoning_reports') \
                        .select('report').eq('user_id', self._storage._user_id) \
                        .limit(1).execute()
                    if resp.data and resp.data[0].get('report'):
                        report_text = resp.data[0]['report']
                else:
                    # Filesystem mode — read latest report
                    reports_dir = os.path.join(_app_dir(), 'reports')
                    latest = os.path.join(reports_dir, 'reasoning-report-latest.md')
                    if os.path.exists(latest):
                        with open(latest, 'r', encoding='utf-8') as f:
                            report_text = f.read()
                if report_text:
                    reasoning_section = (
                        "\n\n## Latest Reasoning Report (excerpts)\n\n"
                        + report_text[:4000]
                    )
            except Exception:
                pass  # reasoning report is optional enrichment

        synthesis_prompt = f"""You are compiling a context briefing from a personal knowledge graph.
The user is about to have an AI conversation about: {query}

Here are their relevant notes (ranked by relevance):

{notes_block}
{reasoning_section}

Compile a briefing that:
- Leads with the most important facts, decisions, and frameworks the user has captured
- Preserves specific details (names, numbers, dates, sources)
- Flags any contradictions between notes
- Ends with 2-3 knowledge gaps worth exploring
- Uses first-person "I" voice (this will be pasted as the user's own context)
- Stays under 800 tokens — density over completeness

Do NOT summarize what a topic is in general terms. The AI already knows that.
Focus on what THIS USER specifically knows, believes, has decided, or is uncertain about."""

        # 5. Call Haiku for synthesis
        import http.client
        import ssl

        payload = json.dumps({
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 1200,
            'messages': [{'role': 'user', 'content': synthesis_prompt}]
        }).encode('utf-8')

        try:
            conn = http.client.HTTPSConnection('api.anthropic.com', timeout=60,
                                                context=ssl.create_default_context())
            conn.request('POST', '/v1/messages', body=payload, headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            })
            resp = conn.getresponse()
            raw = resp.read().decode('utf-8')
            conn.close()

            if resp.status != 200:
                return {'error': f'Synthesis API returned HTTP {resp.status}'}

            result = json.loads(raw)
            if result.get('content') and len(result['content']) > 0:
                packet_text = result['content'][0]['text']
                # Estimate tokens (~4 chars per token)
                token_estimate = len(packet_text) // 4

                # Extract gaps — find the gap/questions section and collect items
                gaps = []
                lines = packet_text.split('\n')
                in_gaps_section = False
                for line in lines:
                    stripped = line.strip()
                    low = stripped.lower()
                    # Detect gap section headers
                    if any(kw in low for kw in ['knowledge gap', 'gaps worth', 'questions worth',
                                                 'worth exploring', 'worth asking']):
                        in_gaps_section = True
                        continue
                    # Another header ends the section
                    if in_gaps_section and stripped.startswith('#'):
                        break
                    # Collect bullets or numbered items in gap section
                    if in_gaps_section and stripped:
                        # Strip bullet/number prefix
                        item = stripped
                        if item.startswith('- '):
                            item = item[2:]
                        elif len(item) > 2 and item[0].isdigit() and item[1] in '.):':
                            item = item[2:].lstrip()
                        elif len(item) > 3 and item[:2].isdigit() and item[2] in '.):':
                            item = item[3:].lstrip()
                        # Clean bold markers
                        if item.startswith('**') and '**' in item[2:]:
                            item = item.replace('**', '')
                        if item:
                            gaps.append(item)
                # Fallback: if no section found, grab question-mark items
                if not gaps:
                    for line in lines:
                        stripped = line.strip()
                        if '?' in stripped and (stripped.startswith('- ') or
                            (len(stripped) > 2 and stripped[0].isdigit())):
                            item = stripped.lstrip('- 0123456789.):')
                            if item:
                                gaps.append(item)

                return {
                    'packet': packet_text,
                    'sources': sources,
                    'gaps': gaps,
                    'token_estimate': token_estimate,
                    'depth': depth,
                    'notes_used': len(note_contents),
                }
            return {'error': 'Empty response from synthesis'}
        except Exception as e:
            return {'error': f'Synthesis error: {type(e).__name__}: {e}'}

    # ===== Backlinks =====

    def get_backlinks(self, note_id):
        """Find all notes that link TO this note via wikilinks."""
        target = self._storage.read_note(note_id)
        if target is None:
            return []
        target_title = target.title

        notes_list = self._storage.list_notes()
        notes_by_id = {n.id: n for n in notes_list}

        backlinks = []
        for n in notes_list:
            if n.id == note_id:
                continue
            links = extract_wikilinks(n.raw_content)
            if any(link.lower() == target_title.lower() for link in links):
                snippet = ''
                link_pattern = f'[[{target_title}]]'
                idx = n.raw_content.lower().find(link_pattern.lower())
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(n.raw_content), idx + len(link_pattern) + 60)
                    snippet = n.raw_content[start:end].strip()
                    snippet = ' '.join(snippet.split())
                    if start > 0:
                        snippet = '...' + snippet
                    if end < len(n.raw_content):
                        snippet += '...'

                backlinks.append({
                    'id': n.id,
                    'title': n.title,
                    'tags': n.tags,
                    'snippet': snippet,
                })

        # Also check manual connections
        for c in self._storage.load_connections():
            if c['target'] == note_id and c['source'] != note_id:
                src_id = c['source']
                if not any(b['id'] == src_id for b in backlinks):
                    src = notes_by_id.get(src_id)
                    if src:
                        backlinks.append({
                            'id': src_id,
                            'title': src.title,
                            'tags': src.tags,
                            'snippet': '(manual connection)',
                        })

        return backlinks

    # ===== Smart Suggestions =====

    def get_link_suggestions(self, note_id):
        """Suggest notes that should be linked from this note based on embedding similarity,
        excluding notes already linked."""
        note = self._storage.read_note(note_id)
        if note is None:
            return []

        existing_links = set(link.lower() for link in extract_wikilinks(note.raw_content))

        notes_list = self._storage.list_notes()
        all_notes = [{'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime}
                     for n in notes_list]

        other_notes = {}
        for n in notes_list:
            if n.id != note_id and n.title.lower() not in existing_links:
                other_notes[n.id] = {
                    'title': n.title,
                    'tags': n.tags,
                    'preview': ' '.join(l for l in n.body.split('\n') if l.strip() and not l.startswith('#'))[:100],
                }

        embeddings = self._embedding_cache.get_embeddings(all_notes)
        if note_id not in embeddings:
            return []

        current_vec = embeddings[note_id]

        scored = []
        for fid, info in other_notes.items():
            if fid not in embeddings:
                continue
            sim = _cosine_similarity(current_vec, embeddings[fid])
            if sim >= 0.20:
                scored.append({
                    'id': fid,
                    'title': info['title'],
                    'tags': info['tags'],
                    'preview': info['preview'],
                    'score': round(sim, 3),
                })

        scored.sort(key=lambda x: -x['score'])
        return scored[:8]

    # ===== Note Templates =====

    def create_note_from_template(self, template_type, title):
        """Create a new note from a template type."""
        filename = title_to_filename(title)
        node_id = filename_to_id(filename)

        if self._storage.note_exists(node_id):
            return {'error': 'Note already exists', 'id': node_id}

        today = time.strftime('%Y-%m-%d')

        templates = {
            'blank': {
                'tags': [],
                'body': f'# {title}\n\n',
            },
            'anecdote': {
                'tags': ['anecdote'],
                'body': f"""# {title}

## Context
_When and where does this take place?_

## The Story
_What happened?_

## Emotional Core
_What does this reveal about the characters?_

## Screenplay Potential
_How could this be dramatized? What's the visual?_

## Related
""",
            },
            'research': {
                'tags': ['research'],
                'body': f"""# {title}

## Key Findings

## Sources

## Connections to Other Notes
""",
            },
            'hunt-skill': {
                'tags': ['skills', 'style'],
                'body': f"""# {title}

## What It Searches For

## Search Strategy

## Filters & Criteria

## Scoring & Ranking

## Related
- [[Hunt Skills]]
- [[Hunt Common Methodology]]
- [[Wardrober]]
""",
            },
            'project': {
                'tags': ['project'],
                'body': f"""# {title}

## Overview

## Goals

## Architecture

## Current Status

## Next Steps
""",
            },
            'quant-skill': {
                'tags': ['quant-skills'],
                'body': f"""# {title}

## Purpose

## How It Works

## Key Concepts

## Related
- [[Training Resources]]
""",
            },
            'feedback': {
                'tags': ['feedback'],
                'body': f"""# {title}

## The Feedback

## Why It Matters

## When To Apply
""",
            },
            'daily': {
                'tags': ['daily'],
                'body': f"""# {time.strftime('%A')}, {title}

## Captures

## Notes

""",
            },
        }

        tmpl = templates.get(template_type, templates['blank'])
        tags_str = json.dumps(tmpl['tags'])

        content = f"""---
title: {title}
tags: {tags_str}
created: {today}
---

{tmpl['body']}"""

        return self._storage.create_note(node_id, filename, content)

    def get_templates(self):
        """Return available note template types."""
        return [
            {'id': 'blank', 'label': 'Blank Note', 'description': 'Empty note with title'},
            {'id': 'anecdote', 'label': 'Anecdote', 'description': 'Story or memoir template'},
            {'id': 'research', 'label': 'Research Note', 'description': 'Key findings + sources'},
            {'id': 'hunt-skill', 'label': 'Hunt Skill', 'description': 'eBay search skill template'},
            {'id': 'project', 'label': 'Project', 'description': 'Project overview + status'},
            {'id': 'quant-skill', 'label': 'Quant Skill', 'description': 'RM training skill'},
            {'id': 'feedback', 'label': 'Feedback', 'description': 'Guidance or correction'},
            {'id': 'daily', 'label': 'Daily Note', 'description': 'Today\'s capture log + journal'},
        ]

    # ===== Daily Digest =====

    def get_daily_digest(self):
        """Generate a summary of graph state for the daily digest."""
        notes_list = self._storage.list_notes()
        notes = []
        tag_counts = Counter()
        total_wikilinks = 0
        all_links = {}  # note_id -> list of wikilink targets
        stale_threshold = time.time() - (30 * 86400)  # 30 days ago

        for n in notes_list:
            links = extract_wikilinks(n.raw_content)
            all_links[n.id] = links
            total_wikilinks += len(links)
            for tag in n.tags:
                tag_counts[tag] += 1
            notes.append({
                'id': n.id, 'title': n.title, 'tags': n.tags,
                'created': n.created, 'mtime': n.mtime,
                'link_count': len(links),
                'word_count': len(n.body.split()),
            })

        # Find orphans (no edges at all)
        edge_nodes = set()
        manual_edges = self._storage.load_connections()

        for note in notes:
            links = all_links.get(note['id'], [])
            if links:
                edge_nodes.add(note['id'])
            # Check if this note is a wikilink target
            for other in notes:
                if other['id'] == note['id']:
                    continue
                olinks = all_links.get(other['id'], [])
                if any(l.lower() == note['title'].lower() for l in olinks):
                    edge_nodes.add(note['id'])
                    break

        for e in manual_edges:
            edge_nodes.add(e['source'])
            edge_nodes.add(e['target'])

        orphans = [n for n in notes if n['id'] not in edge_nodes]

        # Sort by mtime for stale notes (oldest first)
        stale_notes = [n for n in notes if n['mtime'] < stale_threshold]
        stale_notes.sort(key=lambda x: x['mtime'])

        # Recent notes (last 7 days)
        week_ago = time.time() - (7 * 86400)
        recent = [n for n in notes if n['mtime'] > week_ago]
        recent.sort(key=lambda x: -x['mtime'])

        # Drift detection: notes whose meaning has shifted significantly from creation
        drift_scores = self._embedding_cache.get_drift_scores()
        title_map = {n['id']: n['title'] for n in notes}
        drifted = []
        for nid, drift in sorted(drift_scores.items(), key=lambda x: -x[1]):
            if drift >= 0.08 and nid in title_map:  # meaningful drift threshold
                drifted.append({
                    'id': nid,
                    'title': title_map[nid],
                    'drift': round(drift, 3),
                    'drift_pct': round(drift * 100, 1),
                })
        drifted = drifted[:10]

        return {
            'total_notes': len(notes),
            'total_wikilinks': total_wikilinks,
            'total_words': sum(n['word_count'] for n in notes),
            'orphan_count': len(orphans),
            'orphans': [{'id': n['id'], 'title': n['title'], 'tags': n['tags']} for n in orphans[:10]],
            'stale_count': len(stale_notes),
            'stale_notes': [{'id': n['id'], 'title': n['title'], 'days_ago': int((time.time() - n['mtime']) / 86400)} for n in stale_notes[:10]],
            'recent_notes': [{'id': n['id'], 'title': n['title'], 'tags': n['tags']} for n in recent[:10]],
            'top_tags': tag_counts.most_common(10),
            'tag_count': len(tag_counts),
            'drifted_notes': drifted,
            'drifted_count': len(drifted),
        }

    # ===== Context Dashboard =====

    def get_context_dashboard(self):
        """Return rich analytics for the context dashboard panel."""
        notes_list = self._storage.list_notes()
        notes = []
        tag_counts = Counter()
        total_wikilinks = 0
        all_links = {}  # note_id -> list of wikilink targets

        for n in notes_list:
            links = extract_wikilinks(n.raw_content)
            all_links[n.id] = links
            total_wikilinks += len(links)
            for tag in n.tags:
                tag_counts[tag] += 1
            notes.append({
                'id': n.id, 'title': n.title, 'tags': n.tags,
                'created': n.created, 'mtime': n.mtime,
                'link_count': len(links),
                'word_count': len(n.body.split()),
            })

        # --- Connection counts per node (in-degree + out-degree via wikilinks) ---
        title_to_id = {}
        for n in notes:
            title_to_id[n['title'].lower()] = n['id']

        degree = Counter()
        for n in notes:
            for link_title in all_links.get(n['id'], []):
                target_id = title_to_id.get(link_title.lower())
                if target_id and target_id != n['id']:
                    degree[n['id']] += 1
                    degree[target_id] += 1

        # Manual connections
        manual_edges = self._storage.load_connections()
        manual_count = len(manual_edges)
        manual_nodes = set()
        for e in manual_edges:
            degree[e['source']] += 1
            degree[e['target']] += 1
            manual_nodes.add(e['source'])
            manual_nodes.add(e['target'])

        # --- Orphans ---
        connected = set(k for k, v in degree.items() if v > 0)
        orphans = [n for n in notes if n['id'] not in connected]

        # --- Hub nodes (most connected) ---
        hub_nodes = []
        for nid, deg in degree.most_common(8):
            n = next((x for x in notes if x['id'] == nid), None)
            if n:
                hub_nodes.append({'id': nid, 'title': n['title'], 'degree': deg})

        # --- Graph density ---
        num_nodes = len(notes)
        num_edges = total_wikilinks + manual_count
        max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
        density = round(num_edges / max_edges, 4) if max_edges > 0 else 0

        avg_degree = round(sum(degree.values()) / num_nodes, 1) if num_nodes > 0 else 0

        now = time.time()

        # --- Stale notes ---
        stale_threshold = now - (30 * 86400)
        stale_notes = [n for n in notes if n['mtime'] < stale_threshold]
        stale_notes.sort(key=lambda x: x['mtime'])

        # --- Recent notes (7 days) ---
        week_ago = now - (7 * 86400)
        recent = [n for n in notes if n['mtime'] > week_ago]
        recent.sort(key=lambda x: -x['mtime'])

        # --- Tag distribution (all tags with counts) ---
        tag_distribution = tag_counts.most_common(20)

        # --- Word count distribution ---
        total_words = sum(n['word_count'] for n in notes)
        avg_words = round(total_words / num_nodes) if num_nodes > 0 else 0
        longest_notes = sorted(notes, key=lambda x: -x['word_count'])[:5]

        # --- Ontology validation ---
        onto_result = self.validate_ontology()

        return {
            # Core stats
            'total_notes': num_nodes,
            'total_wikilinks': total_wikilinks,
            'total_manual': manual_count,
            'total_edges': num_edges,
            'total_words': total_words,
            'avg_words': avg_words,
            'density': density,
            'avg_degree': avg_degree,
            # Health
            'orphan_count': len(orphans),
            'orphans': [{'id': n['id'], 'title': n['title']} for n in orphans[:10]],
            'stale_count': len(stale_notes),
            'stale_notes': [{'id': n['id'], 'title': n['title'], 'days_ago': int((now - n['mtime']) / 86400)} for n in stale_notes[:8]],
            # Activity
            'recent_count': len(recent),
            'recent_notes': [{'id': n['id'], 'title': n['title']} for n in recent[:8]],
            # Structure
            'hub_nodes': hub_nodes,
            'tag_distribution': tag_distribution,
            'tag_count': len(tag_counts),
            'longest_notes': [{'id': n['id'], 'title': n['title'], 'words': n['word_count']} for n in longest_notes],
            # Ontology
            'ontology_violations': onto_result['violations'][:15],
            'ontology_stats': onto_result['stats'],
        }

    # ===== Ontology =====

    def get_ontology(self):
        """Return the ontology definition for frontend use."""
        return self._storage.load_ontology()

    def add_ontology_type(self, name, description, icon, color, shape, tag=None):
        """Add a new type to ontology.yaml and its tag inference rule."""
        ontology = self._storage.load_ontology()
        if not ontology:
            ontology = {'version': 1, 'types': {}, 'relationships': {}, 'inference': []}

        name = name.strip().lower().replace(' ', '_')
        if name in ontology.get('types', {}):
            return {'ok': False, 'error': f'Type "{name}" already exists'}

        # Add type definition
        ontology.setdefault('types', {})[name] = {
            'description': description.strip(),
            'icon': icon.strip() or name[0].upper(),
            'color': color.strip(),
            'shape': shape.strip(),
        }

        # Add tag inference rule
        infer_tag = (tag or name).strip().lower()
        ontology.setdefault('inference', []).append({'tag': infer_tag, 'type': name})

        self._storage.save_ontology(ontology)
        return {'ok': True, 'name': name, 'tag': infer_tag}

    def validate_ontology(self):
        """Check all notes against ontology rules. Returns advisory violations."""
        ontology = self._storage.load_ontology()
        if not ontology or 'types' not in ontology:
            return {'violations': [], 'stats': {'type_counts': {}, 'untyped': 0, 'total_violations': 0}}

        type_defs = ontology['types']
        violations = []
        type_counts = Counter()
        untyped_notes = []

        # Build full graph data
        notes_by_id = {}
        for n in self._storage.list_notes():
            note_type = infer_type({'type': n.type, 'tags': n.tags}, ontology)
            typed_links = extract_typed_wikilinks(n.raw_content)
            notes_by_id[n.id] = {
                'title': n.title,
                'type': note_type,
                'tags': n.tags,
                'typed_links': typed_links,
            }
            if note_type:
                type_counts[note_type] += 1
            else:
                untyped_notes.append({'id': n.id, 'title': n.title})

        title_to_id = {n['title'].lower(): nid for nid, n in notes_by_id.items()}

        for nid, note in notes_by_id.items():
            nt = note['type']
            if not nt or nt not in type_defs:
                continue
            td = type_defs[nt]

            # Check expected_links.min
            link_targets = [t for _, t in note['typed_links']]
            resolved = [title_to_id.get(t.lower()) for t in link_targets]
            resolved = [r for r in resolved if r]
            if 'expected_links' in td:
                min_links = td['expected_links'].get('min', 0)
                if len(resolved) < min_links:
                    violations.append({
                        'note_id': nid,
                        'title': note['title'],
                        'type': nt,
                        'severity': 'warn',
                        'message': f'{nt} should have at least {min_links} link(s) — has {len(resolved)}',
                    })

            # Check required typed relationships
            if 'relationships' in td:
                for rel_name, rel_def in td['relationships'].items():
                    cardinality = rel_def.get('cardinality', '0+')
                    if cardinality.startswith('1'):
                        # Required relationship — check if at least one typed link matches
                        matching = [t for r, t in note['typed_links'] if r == rel_name]
                        if not matching:
                            violations.append({
                                'note_id': nid,
                                'title': note['title'],
                                'type': nt,
                                'severity': 'info',
                                'message': f'Missing [[{rel_name}::...]] link (expected for {nt})',
                            })

        return {
            'violations': violations,
            'stats': {
                'type_counts': dict(type_counts),
                'untyped': len(untyped_notes),
                'untyped_notes': untyped_notes[:10],
                'total_violations': len(violations),
            },
        }

    # ===== Reasoning Engine =====

    def create_link(self, source_title, target_title):
        """Append bidirectional wikilinks between two notes identified by title."""
        notes_list = self._storage.list_notes()
        title_to_note = {n.title.lower(): n for n in notes_list}

        source = title_to_note.get(source_title.lower())
        target = title_to_note.get(target_title.lower())

        if not source:
            return {'success': False, 'error': f'Note not found: {source_title}'}
        if not target:
            return {'success': False, 'error': f'Note not found: {target_title}'}

        linked = []
        for note, link_title in [(source, target_title), (target, source_title)]:
            existing = [l.lower() for l in extract_wikilinks(note.raw_content)]
            if link_title.lower() not in existing:
                new_content = note.raw_content.rstrip() + f'\n\n[[{link_title}]]\n'
                self._storage.save_note(note.id, new_content)
                linked.append(note.file)

        return {'success': True, 'linked': linked}

    def run_reasoning(self, quiet=False):
        """Run the reasoning engine and return the report markdown.
        quiet=True skips Haiku API calls (free, structural only)."""
        import subprocess
        reason_script = os.path.join(_app_dir(), 'reason.py')
        cmd = [sys.executable, reason_script]
        if quiet:
            cmd.append('--quiet')
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                encoding='utf-8', errors='replace'
            )
            # Read the generated report from reports/ (not notes/)
            report_path = os.path.join(_app_dir(), 'reports', 'reasoning-report-latest.md')
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                _, body = parse_frontmatter(content)
                return {'success': True, 'report': body, 'log': result.stdout}
            return {'success': False, 'error': 'Report not generated', 'log': result.stderr}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Reasoning engine timed out (5 min limit)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ===== Multi-view Canvases =====

    def save_canvas(self, name, data):
        """Save a named canvas view (lens, layout positions, camera state)."""
        return self._storage.save_canvas(name, data)

    def load_canvas(self, name):
        """Load a named canvas view."""
        return self._storage.load_canvas(name)

    def list_canvases(self):
        """List all saved canvas names."""
        return self._storage.list_canvases()

    def delete_canvas(self, name):
        """Delete a named canvas."""
        return self._storage.delete_canvas(name)

    # ===== Semantic Clustering =====

    def get_clusters(self):
        """Compute semantic clusters using sentence embeddings and return cluster assignments + colors."""
        notes_list = self._storage.list_notes()
        note_data = [{'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime}
                     for n in notes_list]

        if len(note_data) < 3:
            return {'clusters': {}, 'colors': {}}

        embeddings = self._embedding_cache.get_embeddings(note_data)
        ids = [n['id'] for n in note_data if n['id'] in embeddings]
        n = len(ids)
        if n < 3:
            return {'clusters': {}, 'colors': {}}

        dim = len(embeddings[ids[0]])

        # Determine k: sqrt(n) capped at 8
        k = max(2, min(8, int(math.sqrt(n))))

        # Initialize centroids by picking k well-separated notes
        centroids = [embeddings[ids[0]]]
        used = {0}
        for _ in range(k - 1):
            best_idx = 0
            best_dist = -1
            for i in range(n):
                if i in used:
                    continue
                min_sim = min(_cosine_similarity(embeddings[ids[i]], c) for c in centroids)
                dist = 1 - min_sim
                if dist > best_dist:
                    best_dist = dist
                    best_idx = i
            used.add(best_idx)
            centroids.append(embeddings[ids[best_idx]])

        # Iterate assignments
        assignments = {}
        for iteration in range(15):
            new_assignments = {}
            for i, nid in enumerate(ids):
                best_c = 0
                best_sim = -1
                for ci in range(len(centroids)):
                    sim = _cosine_similarity(embeddings[nid], centroids[ci])
                    if sim > best_sim:
                        best_sim = sim
                        best_c = ci
                new_assignments[nid] = best_c

            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Recompute centroids as mean of cluster vectors
            new_centroids = []
            for ci in range(len(centroids)):
                cluster_ids = [nid for nid, c in assignments.items() if c == ci]
                if not cluster_ids:
                    new_centroids.append(centroids[ci])
                    continue
                # Average the dense vectors
                avg = [0.0] * dim
                for cid in cluster_ids:
                    for d in range(dim):
                        avg[d] += embeddings[cid][d] / len(cluster_ids)
                new_centroids.append(avg)
            centroids = new_centroids

        # Generate cluster colors
        cluster_colors = [
            '#4dc8ff', '#ff3cac', '#ffd700', '#00e89d', '#a855f7',
            '#ff6b4a', '#3a8ae8', '#ff2d95'
        ]

        colors = {}
        for nid, ci in assignments.items():
            colors[nid] = cluster_colors[ci % len(cluster_colors)]

        return {'clusters': assignments, 'colors': colors}

    # ===== Onboarding (First Launch) =====

    def is_first_launch(self):
        """Check if this is a first launch (no notes and onboarding not completed)."""
        if self._storage.get_config('onboarding_complete'):
            return {'first_launch': False}
        notes = self._storage.list_notes()
        return {'first_launch': len(notes) == 0}

    def detect_sources(self):
        """Auto-detect markdown sources for onboarding import."""
        try:
            from discover import detect_environment
            sources = detect_environment()
            return {'sources': sources}
        except Exception as e:
            return {'sources': [], 'error': str(e)}

    def preview_import(self, source_paths):
        """Build an import manifest from selected source directories.
        Args: source_paths - list of {path, category} dicts."""
        try:
            from discover import build_import_manifest
            manifest = build_import_manifest(source_paths)
            return manifest
        except Exception as e:
            return {'entries': [], 'stats': {}, 'error': str(e)}

    def execute_import(self, entries):
        """Import notes from the approved manifest.
        Args: entries - list of {source_path, proposed_filename, proposed_title, proposed_tags}."""
        try:
            from discover import create_note_content
            imported = 0
            for entry in entries:
                node_id = filename_to_id(entry['proposed_filename'])
                if self._storage.note_exists(node_id):
                    continue  # don't overwrite existing notes
                content = create_note_content(
                    entry['source_path'],
                    entry['proposed_title'],
                    entry['proposed_tags'],
                )
                self._storage.create_note(node_id, entry['proposed_filename'], content)
                imported += 1

            # Mark onboarding complete
            self._storage.set_config('onboarding_complete', True)

            return {'success': True, 'imported': imported}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_upload_import(self, files):
        """Import notes from browser-uploaded file contents.

        Args: files - list of {filename, content} dicts from drag-and-drop.
        """
        try:
            from discover import build_entry_from_content, create_note_content_from_body
            imported = 0
            seen = set()
            for f in files[:500]:
                entry = build_entry_from_content(f['filename'], f['content'])
                node_id = filename_to_id(entry['proposed_filename'])
                if node_id in seen or self._storage.note_exists(node_id):
                    continue
                seen.add(node_id)
                content = create_note_content_from_body(
                    entry['proposed_title'],
                    entry['proposed_tags'],
                    entry['body'],
                )
                self._storage.create_note(node_id, entry['proposed_filename'], content)
                imported += 1

            self._storage.set_config('onboarding_complete', True)
            return {'success': True, 'imported': imported}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ===== Batch Upload (post-onboarding) =====

    def batch_upload(self, files, thresholds=None):
        """Analyze uploaded files for batch import with merge/duplicate detection.

        Args:
            files - list of {filename, content} dicts (max 500).
            thresholds - optional dict with 'title', 'content', 'skip' floats (0-1).
        Returns: {
            auto_import: [...],    # new notes, no conflicts
            needs_review: [...],   # ambiguous matches requiring user decision
            skipped: [...],        # near-identical duplicates
            stats: {total, new, review, skipped}
        }
        """
        from difflib import SequenceMatcher
        from discover import build_entry_from_content

        thresholds = thresholds or {}
        thresh_title = thresholds.get('title', 0.85)
        thresh_content = thresholds.get('content', 0.70)
        thresh_skip = thresholds.get('skip', 0.95)

        files = files[:500]

        # Build lookup of all existing notes
        existing_notes = self._storage.list_notes()
        existing_by_id = {n.id: n for n in existing_notes}
        existing_titles = {n.title.lower(): n for n in existing_notes}

        # Pre-compute embeddings for existing notes (best-effort — skip on failure)
        existing_for_embed = [
            {'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime}
            for n in existing_notes
        ]
        try:
            existing_embeddings = self._embedding_cache.get_embeddings(existing_for_embed) if existing_for_embed else {}
        except Exception:
            existing_embeddings = {}

        auto_import = []
        needs_review = []
        skipped = []
        seen_ids = set()

        for f in files:
            entry = build_entry_from_content(f['filename'], f['content'])
            node_id = filename_to_id(entry['proposed_filename'])

            # Dedup within the upload batch
            if node_id in seen_ids:
                skipped.append({**entry, 'node_id': node_id, 'reason': 'Duplicate within upload batch'})
                continue
            seen_ids.add(node_id)

            # --- Check for exact ID collision ---
            if node_id in existing_by_id:
                existing = existing_by_id[node_id]
                content_sim = SequenceMatcher(None, entry['body'], existing.body).ratio()

                if content_sim >= thresh_skip:
                    skipped.append({
                        **entry, 'node_id': node_id,
                        'reason': f'Near-identical to existing "{existing.title}"',
                        'match_id': existing.id, 'match_title': existing.title,
                        'similarity': round(content_sim, 3),
                    })
                else:
                    needs_review.append({
                        **entry, 'node_id': node_id,
                        'match_type': 'id_collision',
                        'match_id': existing.id, 'match_title': existing.title,
                        'similarity': round(content_sim, 3),
                        'suggestion': 'merge' if content_sim >= 0.50 else 'create_new',
                    })
                continue

            # --- Fuzzy title match ---
            title_lower = entry['proposed_title'].lower()
            best_title_match = None
            best_title_ratio = 0
            for ex_title, ex_note in existing_titles.items():
                ratio = SequenceMatcher(None, title_lower, ex_title).ratio()
                if ratio >= thresh_title and ratio > best_title_ratio:
                    best_title_ratio = ratio
                    best_title_match = ex_note

            if best_title_match:
                content_sim = SequenceMatcher(None, entry['body'], best_title_match.body).ratio()
                if content_sim >= thresh_skip:
                    skipped.append({
                        **entry, 'node_id': node_id,
                        'reason': f'Near-identical to existing "{best_title_match.title}"',
                        'match_id': best_title_match.id, 'match_title': best_title_match.title,
                        'similarity': round(content_sim, 3),
                    })
                else:
                    needs_review.append({
                        **entry, 'node_id': node_id,
                        'match_type': 'title_similar',
                        'match_id': best_title_match.id, 'match_title': best_title_match.title,
                        'similarity': round(best_title_ratio, 3),
                        'suggestion': 'merge' if content_sim >= 0.50 else 'create_new',
                    })
                continue

            # --- Embedding similarity check ---
            if existing_embeddings:
                try:
                    upload_note = [{
                        'id': '__upload__', 'title': entry['proposed_title'],
                        'tags': entry['proposed_tags'], 'body': entry['body'], 'mtime': 0,
                    }]
                    upload_emb = self._embedding_cache.get_embeddings(upload_note)
                    if '__upload__' in upload_emb:
                        upload_vec = upload_emb['__upload__']
                        best_sim = 0
                        best_match = None
                        for ex_id, ex_vec in existing_embeddings.items():
                            sim = _cosine_similarity(upload_vec, ex_vec)
                            if sim >= thresh_content and sim > best_sim:
                                best_sim = sim
                                best_match = existing_by_id.get(ex_id)
                        if best_match:
                            needs_review.append({
                                **entry, 'node_id': node_id,
                                'match_type': 'semantic',
                                'match_id': best_match.id, 'match_title': best_match.title,
                                'similarity': round(best_sim, 3),
                                'suggestion': 'create_new',
                            })
                            continue
                        # Clean up temp embedding
                        if '__upload__' in self._embedding_cache._cache:
                            del self._embedding_cache._cache['__upload__']
                except Exception:
                    pass  # Skip embedding check on failure, fall through to auto-import

            # --- No match: auto-import ---
            auto_import.append({**entry, 'node_id': node_id})

        return {
            'auto_import': auto_import,
            'needs_review': needs_review,
            'skipped': skipped,
            'stats': {
                'total': len(files),
                'new': len(auto_import),
                'review': len(needs_review),
                'skipped': len(skipped),
            },
        }

    def execute_batch_upload(self, decisions):
        """Execute batch upload after user review.

        Args: decisions - list of {
            action: 'create' | 'merge' | 'skip',
            entry: {proposed_title, proposed_tags, body, node_id, ...},
            merge_target_id: optional string (for merge action)
        }
        Returns: {success, imported, merged, skipped, suggestions}
        """
        from discover import create_note_content_from_body

        imported = 0
        merged = 0
        skipped_count = 0
        new_ids = []

        for dec in decisions:
            action = dec.get('action', 'skip')
            entry = dec.get('entry', {})
            node_id = entry.get('node_id', '')

            if action == 'skip':
                skipped_count += 1
                continue

            if action == 'merge':
                target_id = dec.get('merge_target_id', '')
                if not target_id:
                    skipped_count += 1
                    continue
                try:
                    existing = self._storage.read_note(target_id)
                    if not existing:
                        skipped_count += 1
                        continue
                    # Append uploaded content to existing note
                    meta, existing_body = parse_frontmatter(existing.raw_content)
                    new_body = entry.get('body', '')
                    merged_body = existing_body + '\n\n---\n\n*Merged from batch upload:*\n\n' + new_body
                    # Union tags
                    existing_tags = set(meta.get('tags', []))
                    new_tags = set(entry.get('proposed_tags', []))
                    all_tags = sorted(existing_tags | new_tags)
                    meta['tags'] = all_tags
                    # Rebuild frontmatter in same style as existing notes
                    title = meta.get('title', target_id)
                    created = meta.get('created', '')
                    tags_str = ', '.join(all_tags)
                    fm_lines = [f'title: {title}', f'tags: [{tags_str}]']
                    if created:
                        fm_lines.append(f'created: {created}')
                    # Preserve extra fields (type, synced_from, etc.)
                    for k, v in meta.items():
                        if k not in ('title', 'tags', 'created'):
                            fm_lines.append(f'{k}: {v}')
                    merged_content = '---\n' + '\n'.join(fm_lines) + '\n---\n\n' + merged_body + '\n'
                    result = self._storage.save_note(target_id, merged_content)
                    if result.get('error'):
                        skipped_count += 1
                        continue
                    new_ids.append(target_id)
                    merged += 1
                except Exception:
                    skipped_count += 1
                    continue

            elif action == 'create':
                try:
                    if self._storage.note_exists(node_id):
                        skipped_count += 1
                        continue
                    content = create_note_content_from_body(
                        entry.get('proposed_title', node_id),
                        entry.get('proposed_tags', []),
                        entry.get('body', ''),
                    )
                    filename = entry.get('proposed_filename', node_id + '.md')
                    result = self._storage.create_note(node_id, filename, content)
                    if result.get('error'):
                        skipped_count += 1
                        continue
                    new_ids.append(node_id)
                    imported += 1
                except Exception:
                    skipped_count += 1
                    continue

        # Gather auto-link suggestions for all new/merged notes
        suggestions = []
        for nid in new_ids[:20]:  # cap to avoid slow-down
            note = self._storage.read_note(nid)
            if note:
                sug = self._get_auto_link_suggestions(nid, note.raw_content)
                for s in sug[:3]:
                    suggestions.append({'source_id': nid, 'source_title': note.title, **s})

        return {
            'success': True,
            'imported': imported,
            'merged': merged,
            'skipped': skipped_count,
            'suggestions': suggestions,
        }

    def batch_upload_from_folder(self, folder_path):
        """Desktop mode: scan a folder for .md files and run batch upload pipeline."""
        from discover import scan_directory
        entries = scan_directory(folder_path, depth=3, category='custom')
        files = []
        for entry in entries:
            try:
                with open(entry['source_path'], 'r', encoding='utf-8', errors='replace') as fh:
                    files.append({'filename': os.path.basename(entry['source_path']), 'content': fh.read()})
            except Exception:
                continue
        return self.batch_upload(files)

    def generate_ontology(self, preview_only=False):
        """Auto-generate an ontology tailored to the user's notes.

        Analyzes tags, types, and content themes, then calls Haiku to
        propose a custom type system with colors, shapes, and inference rules.

        If preview_only=True, generates the ontology and returns a diff but
        does NOT save it. The caller must call generate_ontology() again
        (or apply_ontology) to persist.

        Rate-limited: 5-minute cooldown between regenerations, max 10/day.
        Only the preview step (which calls Haiku) is gated — apply is free.
        """
        import anthropic
        import time as _time

        now = _time.time()
        if not hasattr(self, '_regen_cooldown_at'):
            self._regen_cooldown_at = 0.0
            self._regen_day_key = ''
            self._regen_day_count = 0

        # Daily cap — reset at midnight (keyed by date string)
        today = _time.strftime('%Y-%m-%d')
        if today != self._regen_day_key:
            self._regen_day_key = today
            self._regen_day_count = 0

        if self._regen_day_count >= 10:
            return {'success': False, 'error': 'Daily limit reached (10 regenerations per day). Try again tomorrow.'}

        # 5-minute cooldown
        wait = self._regen_cooldown_at - now
        if wait > 0:
            mins = int(wait // 60)
            secs = int(wait % 60)
            return {'success': False, 'error': f'Please wait {mins}m {secs}s before regenerating again.'}

        self._regen_cooldown_at = now + 300  # 5 minutes
        self._regen_day_count += 1

        DARK_BG_PALETTE = [
            "#4dc8ff", "#ff6b6b", "#96ceb4", "#ffd700", "#a855f7",
            "#f4a460", "#45b7d1", "#dda0dd", "#4ecdc4", "#e74c3c",
        ]
        SHAPES = [
            "box", "octahedron", "torusknot", "dodecahedron", "cone",
            "icosahedron", "cylinder", "torus", "tetrahedron", "sphere",
        ]

        notes = self._storage.list_notes()
        if not notes:
            return {'success': False, 'error': 'No notes to analyze'}

        # Collect tag frequencies
        tag_freq = {}
        explicit_types = set()
        titles = []
        previews = []
        for n in notes:
            titles.append(n.title)
            if n.preview:
                previews.append(f"{n.title}: {n.preview[:100]}")
            if n.type:
                explicit_types.add(n.type)
            for t in n.tags:
                tag_freq[t] = tag_freq.get(t, 0) + 1

        # Sort tags by frequency
        sorted_tags = sorted(tag_freq.items(), key=lambda x: -x[1])[:50]

        # Check for existing ontology (regeneration vs first-time)
        existing_ontology = self._storage.load_ontology()
        existing_types = existing_ontology.get('types', {}) if existing_ontology else {}

        if existing_types:
            # Regeneration: instruct Haiku to preserve existing types
            existing_block = f"""
IMPORTANT — This is a REGENERATION. The user already has an established ontology. Preserve continuity:
- Existing types (KEEP these — same name, color, shape, icon — unless truly wrong):
{json.dumps(existing_types, indent=2)}
- You may ADD new types if the notes clearly demand categories not covered above.
- You may REMOVE a type only if zero notes match it and it has no inference rules.
- You may RENAME a type only if the new name is strictly better and covers the same notes.
- PRESERVE exact colors and shapes for kept types — the user has built spatial memory around them.
"""
        else:
            existing_block = ''

        prompt = f"""You are configuring a personal knowledge graph. Analyze this user's notes and generate a custom ontology (type system) that fits their content.

Note landscape:
- {len(notes)} notes total
- Tags (with frequency): {json.dumps(sorted_tags)}
- Explicit types already set: {json.dumps(list(explicit_types))}
- Note titles: {json.dumps(titles[:30])}
- Sample content: {json.dumps(previews[:10])}

Available shapes (use each at most once): {json.dumps(SHAPES)}
{existing_block}
Generate 3-8 note types that naturally cover this user's content. For each type provide:
- name: lowercase (e.g. "project", "research", "concept")
- description: one sentence describing what this type represents
- icon: single uppercase letter that represents this type
- color: hex color visible on dark background (#0a0e17)
- shape: one of the available shapes

Also generate tag-to-type inference rules mapping existing tags to your types.

Return ONLY a JSON object:
{{
  "types": {{
    "type_name": {{
      "description": "...",
      "icon": "P",
      "color": "#hex",
      "shape": "box"
    }}
  }},
  "inference": [
    {{"tag": "existing_tag", "type": "type_name"}},
  ],
  "relationships": {{
    "related_to": {{"label": "related to", "reverse_label": "related to"}},
    "supports": {{"label": "supports", "reverse_label": "supported by"}},
    "part_of": {{"label": "part of", "reverse_label": "contains"}}
  }}
}}

No markdown fences or explanation."""

        try:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=2048,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
                if raw.endswith('```'):
                    raw = raw[:-3]
                raw = raw.strip()
            result = json.loads(raw)

            # Validate and fix colors/shapes
            types = result.get('types', {})
            used_shapes = set()
            for i, (name, tdef) in enumerate(types.items()):
                # Ensure valid shape
                shape = tdef.get('shape', 'sphere')
                if shape not in SHAPES or shape in used_shapes:
                    # Assign unused shape
                    for s in SHAPES:
                        if s not in used_shapes:
                            shape = s
                            break
                used_shapes.add(shape)
                tdef['shape'] = shape

                # Ensure color is a valid hex
                color = tdef.get('color', '')
                if not color or not color.startswith('#') or len(color) != 7:
                    tdef['color'] = DARK_BG_PALETTE[i % len(DARK_BG_PALETTE)]

            # Ensure every explicit type found on notes is in the ontology
            for et in explicit_types:
                if et not in types:
                    idx = len(types)
                    # Find an unused shape
                    shape = 'sphere'
                    for s in SHAPES:
                        if s not in used_shapes:
                            shape = s
                            break
                    used_shapes.add(shape)
                    types[et] = {
                        'description': f'Notes explicitly typed as {et}',
                        'icon': et[0].upper(),
                        'color': DARK_BG_PALETTE[idx % len(DARK_BG_PALETTE)],
                        'shape': shape,
                    }

            # Build ontology in the format the frontend expects
            ontology = {
                'version': 1,
                'types': types,
                'inference': result.get('inference', []),
                'relationships': result.get('relationships', {
                    'related_to': {'label': 'related to', 'reverse_label': 'related to'},
                    'supports': {'label': 'supports', 'reverse_label': 'supported by'},
                    'part_of': {'label': 'part of', 'reverse_label': 'contains'},
                }),
            }

            # Compute diff against existing ontology
            diff = self._ontology_diff(existing_types, types)

            if preview_only:
                # Cache the generated ontology for later apply
                self._pending_ontology = ontology
                return {
                    'success': True,
                    'preview': True,
                    'types_created': len(types),
                    'ontology': ontology,
                    'diff': diff,
                }

            self._storage.save_ontology(ontology)
            self._pending_ontology = None
            return {
                'success': True,
                'types_created': len(types),
                'ontology': ontology,
                'diff': diff,
            }
        except Exception as exc:
            print(f"[ontology] Haiku call failed, using fallback: {exc}")
            # Fallback: generate a basic ontology from tags without Haiku
            types = {}
            for i, (tag, count) in enumerate(sorted_tags[:8]):
                types[tag] = {
                    'description': f'Notes tagged with {tag}',
                    'icon': tag[0].upper(),
                    'color': DARK_BG_PALETTE[i % len(DARK_BG_PALETTE)],
                    'shape': SHAPES[i % len(SHAPES)],
                }
            ontology = {
                'version': 1,
                'types': types,
                'inference': [{'tag': tag, 'type': tag} for tag, _ in sorted_tags[:8]],
                'relationships': {
                    'related_to': {'label': 'related to', 'reverse_label': 'related to'},
                },
            }
            diff = self._ontology_diff(existing_types, types)

            if preview_only:
                self._pending_ontology = ontology
                return {
                    'success': True,
                    'preview': True,
                    'types_created': len(types),
                    'ontology': ontology,
                    'diff': diff,
                    'fallback': True,
                    'error': str(exc),
                }

            self._storage.save_ontology(ontology)
            return {
                'success': True,
                'types_created': len(types),
                'ontology': ontology,
                'diff': diff,
                'fallback': True,
                'error': str(exc),
            }

    def apply_pending_ontology(self):
        """Apply a previously previewed ontology."""
        if not getattr(self, '_pending_ontology', None):
            return {'success': False, 'error': 'No pending ontology to apply'}
        self._storage.save_ontology(self._pending_ontology)
        self._pending_ontology = None
        return {'success': True}

    @staticmethod
    def _ontology_diff(old_types, new_types):
        """Compute a human-readable diff between old and new ontology types."""
        old_names = set(old_types.keys())
        new_names = set(new_types.keys())
        added = sorted(new_names - old_names)
        removed = sorted(old_names - new_names)
        kept = sorted(old_names & new_names)
        changed = []
        for name in kept:
            old_t, new_t = old_types[name], new_types[name]
            diffs = []
            if old_t.get('color') != new_t.get('color'):
                diffs.append('color')
            if old_t.get('shape') != new_t.get('shape'):
                diffs.append('shape')
            if old_t.get('description') != new_t.get('description'):
                diffs.append('description')
            if diffs:
                changed.append({'name': name, 'fields': diffs})
        return {
            'added': added,
            'removed': removed,
            'kept': [n for n in kept if n not in [c['name'] for c in changed]],
            'changed': changed,
        }

    def set_note_type(self, note_id, new_type):
        """Change a note's type in its frontmatter and update the ontology if needed."""
        try:
            note = self._storage.read_note(note_id)
        except Exception as e:
            return {'error': f'Failed to read note: {e}'}
        if note is None:
            return {'error': 'Note not found'}

        raw = note.raw_content or ''
        if raw.startswith('---'):
            parts = raw.split('---', 2)
            if len(parts) >= 3:
                fm = parts[1]
                # Replace or add type field in frontmatter
                if re.search(r'^type\s*:', fm, re.MULTILINE):
                    if new_type:
                        fm = re.sub(r'^type\s*:.*$', f'type: {new_type}', fm, flags=re.MULTILINE)
                    else:
                        fm = re.sub(r'^type\s*:.*\n?', '', fm, flags=re.MULTILINE)
                elif new_type:
                    fm = fm.rstrip() + f'\ntype: {new_type}\n'
                new_content = '---' + fm + '---' + parts[2]
            else:
                new_content = raw
        else:
            # No frontmatter — wrap content with one
            if new_type:
                new_content = f'---\ntype: {new_type}\n---\n\n' + raw
            else:
                new_content = raw

        try:
            self._storage.save_note(note_id, new_content)
        except Exception as e:
            return {'error': f'Failed to save note: {e}'}

        # If the type is new, add it to the ontology
        if new_type:
            try:
                ontology = self._storage.load_ontology() or {'version': 1, 'types': {}, 'inference': [], 'relationships': {}}
                if new_type not in ontology.get('types', {}):
                    DARK_BG_PALETTE = [
                        "#4dc8ff", "#ff6b6b", "#96ceb4", "#ffd700", "#a855f7",
                        "#f4a460", "#45b7d1", "#dda0dd", "#4ecdc4", "#e74c3c",
                    ]
                    SHAPES = [
                        "box", "octahedron", "torusknot", "dodecahedron", "cone",
                        "icosahedron", "cylinder", "torus", "tetrahedron", "sphere",
                    ]
                    used_shapes = {t.get('shape') for t in ontology.get('types', {}).values()}
                    shape = 'sphere'
                    for s in SHAPES:
                        if s not in used_shapes:
                            shape = s
                            break
                    idx = len(ontology['types'])
                    ontology['types'][new_type] = {
                        'description': f'User-defined type: {new_type}',
                        'icon': new_type[0].upper(),
                        'color': DARK_BG_PALETTE[idx % len(DARK_BG_PALETTE)],
                        'shape': shape,
                    }
                    self._storage.save_ontology(ontology)
            except Exception as e:
                # Note was saved but ontology update failed — still report success
                return {'success': True, 'type': new_type, 'warning': f'Ontology update failed: {e}'}

        return {'success': True, 'type': new_type}

    def generate_profile(self, answers):
        """Generate a soul node + 15-30 starter nodes from onboarding interview.

        Sends all answers to Claude which returns structured JSON defining
        each node (title, tags, type, body, wikilinks). Creates them all in
        the graph with connections back to the soul node.
        """
        import anthropic
        import json as _json

        name = answers.get('name', '').strip()
        if not name:
            return {'success': False, 'error': 'Name is required'}

        today = time.strftime('%Y-%m-%d')

        # Build a structured summary of all answers for the prompt
        answer_block = f"Name: {name}\n"

        # Core questions (Q1-Q12)
        projects = answers.get('projects', [])
        if projects:
            answer_block += f"Projects/domains (up to 3): {', '.join(projects)}\n"
        brain_words = answers.get('brain_words', [])
        if brain_words:
            answer_block += f"Brain words (picked 3): {', '.join(brain_words)}\n"
        learn_style = answers.get('learn_style')
        if learn_style is not None:
            label = 'file it away' if learn_style < 30 else ('connect everything' if learn_style > 70 else 'somewhere in between')
            answer_block += f"Learning style: {learn_style}/100 ({label})\n"
        for key, label in [
            ('open_question', 'Unanswered question chewing on for months'),
            ('contrarian', 'Contrarian belief in their field'),
            ('never_do', 'What Heartwood should never do'),
            ('throughline', 'The throughline connecting everything they think about'),
        ]:
            val = answers.get(key, '').strip()
            if val:
                answer_block += f"{label}: {val}\n"

        borrowed = answers.get('borrowed_mind', '').strip()
        borrowed_why = answers.get('borrowed_why', '').strip()
        if borrowed:
            answer_block += f"Mind they'd borrow: {borrowed}" + (f" (why: {borrowed_why})" if borrowed_why else "") + "\n"

        domain_count = answers.get('domain_count', '')
        if domain_count:
            answer_block += f"Domain count: {domain_count}\n"
        stall_reason = answers.get('stall_reason', '')
        if stall_reason:
            answer_block += f"Why projects stall: {stall_reason}\n"

        # Sliders
        for key, labels in [
            ('slider_structure', ('structure', 'chaos')),
            ('slider_solo', ('solo', 'collaborative')),
            ('slider_breadth', ('breadth', 'depth')),
        ]:
            val = answers.get(key)
            if val is not None:
                answer_block += f"Spectrum {labels[0]}↔{labels[1]}: {val}/100\n"

        # Bonus round (Q13-Q24)
        has_bonus = bool(answers.get('superpower'))
        if has_bonus:
            for key, label in [
                ('superpower', 'Cognitive superpower chosen'),
                ('mind_shape', 'Mental model shape'),
                ('free_saturday', 'What happens on a free Saturday'),
                ('deadline', 'Relationship with deadlines'),
                ('hidden_skill', 'Hidden skill not on any resume'),
                ('first_hire', 'First hire for impossible project'),
                ('ted_talk', '20-minute unsolicited TED talk topic'),
                ('delete_tool', 'Tool they\'d delete forever'),
                ('rabbit_hole', 'Guilty-pleasure rabbit hole'),
                ('fifteen', 'What 15-year-old self expected'),
                ('smell', 'If their mind were a place, it would smell like'),
            ]:
                val = answers.get(key, '').strip() if isinstance(answers.get(key), str) else answers.get(key, '')
                if val:
                    answer_block += f"{label}: {val}\n"

            for key, labels in [
                ('slider_data', ('data', 'gut feeling')),
                ('slider_experts', ('experts', 'experience')),
                ('slider_plans', ('plans', 'improvisation')),
            ]:
                val = answers.get(key)
                if val is not None:
                    answer_block += f"Trust spectrum {labels[0]}↔{labels[1]}: {val}/100\n"

        target_count = '20-30' if has_bonus else '15-20'

        prompt = f"""You are generating starter nodes for a personal knowledge graph from an onboarding interview. The user answered questions about themselves. Generate {target_count} interconnected notes.

{answer_block}

Generate a JSON array of notes. Each note has: title, tags (array), type (one of: profile, project, concept, question, character-study, pattern, influence, rule), body (50-200 words of insightful content), wikilinks (array of other note titles from this batch to link to).

Required nodes:
1. SOUL NODE — title is exactly "{name}", type "profile", a synthesized character study (200-300 words). Reference patterns, tensions, and insights from their answers. Third person. Connect to most other nodes.
2. PROJECT NODES — one for each project/domain they listed. Include what's interesting about it, potential connections to other domains.
3. THINKING STYLE — a character study of how they think, synthesized from brain words, sliders, learning style, domain count.
4. OPEN QUESTION — their unanswered question, framed as a genuine inquiry node with connections to relevant projects/concepts.
5. INFLUENCE NODE — the mind they'd borrow and why, connected to relevant domains.
6. CONTRARIAN THESIS — their contrarian belief, framed as a position with implications.
7. MOTIVATION PATTERN — synthesized from stall reason, sliders, and any relevant answers.
8. COLLABORATION RULES — what they want and don't want from Heartwood, from their "never do" answer.
9. THROUGHLINE — the connecting thread, linked to every project and the soul node.
10. INFERRED CONCEPT NODES (3-5) — domains or ideas that emerge from the aggregate of their answers but they didn't name explicitly. These demonstrate Heartwood's ability to see what the user can't.
11. GAP NODES (1-2) — questions the graph implies but the user didn't answer. Frame as "Open Question: ..." with type "question".

{"BONUS NODES (if bonus round data present):" if has_bonus else ""}
{"12. RECHARGE PATTERN — from free Saturday answer" if has_bonus else ""}
{"13. HIDDEN SKILL — framed as an asset node" if has_bonus else ""}
{"14. PASSION DOMAIN — from TED talk topic" if has_bonus else ""}
{"15. ORIGIN STORY — from 15-year-old self answer, identity drift" if has_bonus else ""}
{"16. EPISTEMOLOGICAL PROFILE — from trust sliders" if has_bonus else ""}
{"17. SYNESTHETIC ANCHOR — the smell answer, poetic and memorable" if has_bonus else ""}

Every note body should be specific to THIS person — no generic filler. Wikilinks should use [[Title]] format inside the body text naturally. The soul node should be the densest hub.

Output ONLY valid JSON array. No markdown fences, no explanation."""

        try:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=8192,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
                if raw.endswith('```'):
                    raw = raw[:-3]
            nodes = _json.loads(raw)
        except Exception as e:
            print(f"[onboarding] Starter graph generation failed: {e}")
            # Fallback: create minimal soul node
            nodes = [{
                'title': name,
                'tags': ['user', 'profile'],
                'type': 'profile',
                'body': f'Identity node for {name}.',
                'wikilinks': []
            }]

        # Create all nodes in the graph
        soul_node_id = None
        created_count = 0

        for node in nodes:
            title = node.get('title', '').strip()
            if not title:
                continue

            tags = node.get('tags', [])
            note_type = node.get('type', 'concept')
            body = node.get('body', '')

            slug = title_to_filename(title)
            node_id = filename_to_id(slug)

            # Track soul node
            if title == name or note_type == 'profile':
                soul_node_id = node_id

            tag_str = ', '.join(tags) if tags else 'onboarding'
            content = f"""---
title: {title}
tags: [{tag_str}]
created: {today}
type: {note_type}
---

# {title}

{body}
"""
            try:
                if self._storage.note_exists(node_id):
                    self._storage.save_note(node_id, content)
                else:
                    self._storage.create_note(node_id, slug, content)
                created_count += 1
            except Exception as e:
                print(f"[onboarding] Failed to create node '{title}': {e}")

        # Set soul node
        if soul_node_id:
            self._storage.set_config('soul_node_id', soul_node_id)

        # Connect all nodes to soul node
        if soul_node_id:
            for node in nodes:
                title = node.get('title', '').strip()
                if not title or title == name:
                    continue
                node_id = filename_to_id(title_to_filename(title))
                try:
                    self.add_connection(soul_node_id, node_id)
                except Exception:
                    pass

        # Seed the FAQ note
        self._seed_faq_note()

        return {'success': True, 'node_id': soul_node_id, 'name': name, 'node_count': created_count}

    def _seed_faq_note(self):
        """Add the Heartwood FAQ note to the user's graph if it doesn't exist.

        Also creates a manual connection from the FAQ to the soul node.
        """
        faq_id = 'heartwood-faq'
        if self._storage.note_exists(faq_id):
            return

        # Load from the bundled file if available, otherwise use embedded content
        faq_content = None
        faq_path = os.path.join(os.path.dirname(__file__), 'notes', 'heartwood-faq.md')
        if os.path.exists(faq_path):
            with open(faq_path, 'r', encoding='utf-8') as f:
                faq_content = f.read()

        if not faq_content:
            today = time.strftime('%Y-%m-%d')
            faq_content = f"""---
title: Heartwood FAQ
tags: [heartwood, onboarding]
created: {today}
type: reference
---

# Heartwood FAQ

Welcome to Heartwood — your personal knowledge graph with an AI reasoning engine.

## Quick start

1. **Capture** — add notes via the editor, Chrome extension, Telegram bot, or markdown import
2. **Connect** — use [[wikilinks]] to link notes; semantic edges form automatically via embeddings
3. **Reason** — run the reasoning engine to surface patterns, gaps, and contradictions across your graph
4. **Compound** — each cycle makes the next one more valuable

## Key concepts

- **Soul Node** — your identity node at the center of the graph, created from your onboarding interview
- **Ontology** — a custom type system (colors + shapes) auto-generated from your content
- **Reasoning Engine** — 6 analytical passes that reveal your thinking patterns back to you
- **Semantic Edges** — automatic connections between notes based on meaning, not just explicit links

## Tips

- Capture liberally, curate later — the reasoning engine works better with more raw material
- Use [[wikilinks]] — they create explicit connections that strengthen the graph
- Run reasoning weekly after adding 5-10 new notes
- Let the ontology evolve — regenerate types when existing ones stop fitting

## Data & privacy

Your notes are stored with Row-Level Security — every query is scoped to your authenticated session. No other user or admin can access your content.
"""

        try:
            self._storage.create_note(faq_id, 'heartwood-faq.md', faq_content)
            # Link FAQ to soul node
            soul_id = self._storage.get_config('soul_node_id')
            if soul_id:
                self.add_connection(soul_id, faq_id)
        except Exception as e:
            print(f"[onboarding] Failed to seed FAQ note: {e}")

    def anchor_soul_node(self):
        """Connect the soul node to the most relevant imported notes via wikilinks.

        Uses semantic similarity to find the top matches, then appends a
        'Connected Notes' section with [[wikilinks]] to the soul note content.
        """
        soul_id = self._storage.get_config('soul_node_id')
        if not soul_id:
            return {'success': False, 'error': 'No soul node configured'}

        notes_list = self._storage.list_notes()
        soul_note = None
        other_notes = []
        for n in notes_list:
            if n.id == soul_id:
                soul_note = n
            else:
                other_notes.append(n)

        if not soul_note or not other_notes:
            return {'success': False, 'error': 'Soul note or other notes not found'}

        # Build embeddings for similarity matching
        note_data = [{'id': n.id, 'title': n.title, 'tags': n.tags, 'body': n.body, 'mtime': n.mtime}
                     for n in [soul_note] + other_notes]
        embeddings = self._embedding_cache.get_embeddings(note_data)

        soul_vec = embeddings.get(soul_id)
        if not soul_vec:
            return {'success': False, 'error': 'Could not compute soul node embedding'}

        # Score all other notes by similarity to soul node
        scored = []
        for n in other_notes:
            vec = embeddings.get(n.id)
            if vec:
                sim = _cosine_similarity(soul_vec, vec)
                scored.append((sim, n))
        scored.sort(key=lambda x: -x[0])

        # Take top matches (up to 15, or all if fewer) — threshold 0.2 to avoid noise
        top = [(sim, n) for sim, n in scored[:15] if sim >= 0.2]
        if not top:
            # Fallback: just link to all notes (small graph)
            top = [(0, n) for n in other_notes[:15]]

        # Build wikilink section
        links = [f'- [[{n.title}]]' for _, n in top]
        section = '\n\n## Connected Notes\n\n' + '\n'.join(links) + '\n'

        # Append to soul note content and save
        updated_content = soul_note.raw_content.rstrip() + section
        self._storage.save_note(soul_id, updated_content)

        return {'success': True, 'linked': len(top)}

    def skip_onboarding(self):
        """Mark onboarding as complete without importing anything."""
        self._storage.set_config('onboarding_complete', True)
        return {'success': True}

    def browse_folder(self):
        """Open a native folder picker dialog and scan the selected directory."""
        try:
            import webview as wv
            window = wv.windows[0] if wv.windows else None
            if not window:
                return {'error': 'No window available'}
            result = window.create_file_dialog(wv.FOLDER_DIALOG)
            if result and len(result) > 0:
                folder = result[0]
                from discover import scan_directory
                entries = scan_directory(folder, depth=3, category='custom')
                return {
                    'path': folder,
                    'file_count': len(entries),
                    'description': f'{len(entries)} markdown files found',
                }
            return {'cancelled': True}
        except Exception as e:
            return {'error': str(e)}

    # ===== Config =====

    def get_config(self, key):
        """Get a config value."""
        return {'value': self._storage.get_config(key)}

    def set_config(self, key, value):
        """Set a config value."""
        return self._storage.set_config(key, value)

    # ===== Daily Notes =====

    def create_note_with_title(self, title):
        """Create a note with a user-provided title, or open it if it already exists."""
        today = time.strftime('%Y-%m-%d')
        filename = title_to_filename(title)
        node_id = filename_to_id(filename)

        if self._storage.note_exists(node_id):
            return {'id': node_id, 'title': title, 'created': False}

        content = f"""---
title: "{title}"
tags: []
created: {today}
---

# {title}

"""
        self._storage.create_note(node_id, filename, content)
        return {'id': node_id, 'title': title, 'created': True}

    def get_or_create_daily_note(self):
        """Get today's daily note, creating it if it doesn't exist.
        Returns {id, title, created, is_new}."""
        today = time.strftime('%Y-%m-%d')
        weekday = time.strftime('%A')
        title = today
        filename = title_to_filename(title)
        node_id = filename_to_id(filename)

        if self._storage.note_exists(node_id):
            return {'id': node_id, 'title': title, 'created': False}

        content = f"""---
title: "{title}"
tags: ["daily"]
created: {today}
---

# {weekday}, {today}

## Captures

## Notes

"""
        self._storage.create_note(node_id, filename, content)
        return {'id': node_id, 'title': title, 'created': True}

    def _route_captures(self, captures):
        """Use Claude Haiku to cluster captures by topic and decide whether to
        create new notes or append to existing ones.
        Returns list of actions: {action: 'create'|'append', title, captures, links, backlinks}"""
        api_key = self._get_api_key()
        if not api_key:
            return None

        graph_index = self._get_graph_index()

        # Build rich context: title + tags + links for each note
        notes_lines = []
        for note in graph_index[:120]:
            tags_str = ', '.join(note['tags']) if note['tags'] else ''
            links_str = ', '.join(f'[[{l}]]' for l in note['links'][:5]) if note['links'] else ''
            parts = [f'- {note["title"]}']
            if tags_str:
                parts.append(f'[{tags_str}]')
            if links_str:
                parts.append(f'→ {links_str}')
            notes_lines.append(' '.join(parts))
        notes_list = '\n'.join(notes_lines) if notes_lines else 'None'

        capture_list = '\n'.join(f'{i+1}. {c}' for i, c in enumerate(captures))

        prompt = f"""You are a knowledge graph assistant. You have a list of quick captures sent via Telegram throughout the day. Your job is to route each capture to the right place in the graph.

For each capture (or group of related captures), decide:
- CREATE a new note if the topic is new — give it a terse title (2-4 words max)
- APPEND to an existing note if the capture clearly belongs there

Existing notes (with tags and links to show graph structure):
{notes_list}

Captures:
{capture_list}

Rules:
- Group related captures together (same action line)
- Titles must be 2-4 words, specific, no filler
- Only APPEND when the capture genuinely extends an existing note's topic
- LINKS: add [[wikilinks]] to semantically related notes — match by TOPIC, not just keywords. Use the tags and existing links above to find connections a title alone would miss.
- BACKLINKS: list existing notes that should link BACK to this new/updated note. If you CREATE a note about "Denim Sizing" that links to [[Denim Hunt]], then [[Denim Hunt]] should back-link to it.
- Each capture number must appear in exactly one action

Respond in EXACTLY this format (one action per line, no other text):
CREATE "Title Here" [1,3] LINKS: [[Note1]], [[Note2]] BACKLINKS: [[Note1]]
APPEND "Existing Note Title" [2] LINKS: [[Other Note]] BACKLINKS: none
CREATE "Another Title" [4,5] LINKS: none BACKLINKS: none"""

        payload = json.dumps({
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 1024,
            'messages': [{'role': 'user', 'content': prompt}]
        }).encode('utf-8')

        try:
            import http.client
            import ssl
            conn = http.client.HTTPSConnection('api.anthropic.com', timeout=30,
                                                context=ssl.create_default_context())
            conn.request('POST', '/v1/messages', body=payload, headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            })
            resp = conn.getresponse()
            raw = resp.read().decode('utf-8')
            conn.close()

            if resp.status != 200:
                print(f"[capture] Claude API HTTP {resp.status}: {raw[:300]}", flush=True)
                return None

            result = json.loads(raw)
            if not result.get('content'):
                return None

            text = result['content'][0]['text'].strip()
            actions = []

            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Parse: CREATE "Title" [1,3] LINKS: [[X]], [[Y]]
                # Parse: APPEND "Title" [2] LINKS: [[X]]
                action_type = None
                if line.startswith('CREATE'):
                    action_type = 'create'
                    line = line[6:].strip()
                elif line.startswith('APPEND'):
                    action_type = 'append'
                    line = line[6:].strip()
                else:
                    continue

                # Extract quoted title
                title_match = re.match(r'"([^"]+)"', line)
                if not title_match:
                    continue
                title = title_match.group(1)
                line = line[title_match.end():].strip()

                # Extract capture indices [1,3]
                idx_match = re.match(r'\[([^\]]+)\]', line)
                cap_indices = []
                if idx_match:
                    try:
                        cap_indices = [int(x.strip()) - 1 for x in idx_match.group(1).split(',')]
                    except ValueError:
                        pass
                    line = line[idx_match.end():].strip()

                # Extract links and backlinks
                links = []
                backlinks = []
                if 'BACKLINKS:' in line:
                    parts = line.split('BACKLINKS:')
                    backlink_text = parts[1].strip()
                    if backlink_text.lower() != 'none':
                        backlinks = WIKILINK_RE.findall(backlink_text)
                    line = parts[0].strip()

                if 'LINKS:' in line:
                    link_text = line.split('LINKS:')[1].strip()
                    if link_text.lower() != 'none':
                        links = WIKILINK_RE.findall(link_text)

                actions.append({
                    'action': action_type,
                    'title': title,
                    'capture_indices': cap_indices,
                    'links': links,
                    'backlinks': backlinks,
                })

            print(f"[capture] Routed {len(captures)} captures into {len(actions)} actions", flush=True)
            for a in actions:
                print(f"  {a['action'].upper()} \"{a['title']}\" caps={a['capture_indices']} links={a['links']} backlinks={a.get('backlinks', [])}", flush=True)
            return actions

        except Exception as e:
            print(f"[capture] Claude API error: {type(e).__name__}: {e}", flush=True)
            return None

    def _summarize_captures(self, capture_texts, title=''):
        """Use Claude Haiku to generate a structured markdown summary from raw captures.
        Returns a string of bullet points, or empty string on failure."""
        api_key = self._get_api_key()
        if not api_key or not capture_texts:
            return ''

        joined = '\n'.join(f'- {t}' for t in capture_texts)
        prompt = f"""You are a knowledge graph assistant. Below are raw captures sent via Telegram for a note titled "{title}".

Produce a concise markdown summary: 2-5 bullet points that distill the key ideas, decisions, or facts. Be terse and precise — no filler. Preserve any specific details (names, numbers, URLs). Do NOT reproduce the raw text — abstract it.

Raw captures:
{joined}

Respond with ONLY the bullet points (markdown list), no preamble."""

        payload = json.dumps({
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 512,
            'messages': [{'role': 'user', 'content': prompt}]
        }).encode('utf-8')

        try:
            import http.client
            import ssl
            conn = http.client.HTTPSConnection('api.anthropic.com', timeout=30,
                                                context=ssl.create_default_context())
            conn.request('POST', '/v1/messages', body=payload, headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            })
            resp = conn.getresponse()
            raw = resp.read().decode('utf-8')
            conn.close()

            if resp.status != 200:
                print(f"[capture] Summary API HTTP {resp.status}: {raw[:300]}", flush=True)
                return ''

            result = json.loads(raw)
            if not result.get('content'):
                return ''
            return result['content'][0]['text'].strip()

        except Exception as e:
            print(f"[capture] Summary API error: {type(e).__name__}: {e}", flush=True)
            return ''

    def _format_capture_lines(self, captures):
        """Format capture dicts into timestamped markdown lines."""
        lines = []
        for cap in captures:
            ts = cap.get('time', '')
            text = cap.get('text', '')
            if ts:
                try:
                    dt = datetime.datetime.fromisoformat(ts)
                    ts_fmt = dt.strftime('%H:%M')
                except Exception:
                    ts_fmt = ts[:5]
                lines.append(f'- **{ts_fmt}** — {text}')
            else:
                lines.append(f'- {text}')
        return lines

    def _write_backlinks(self, target_titles, source_title):
        """Add a [[wikilink]] to source_title inside each target note.
        Skips if the link already exists."""
        link_text = f'[[{source_title}]]'
        notes_list = self._storage.list_notes()
        for target_title in target_titles:
            for n in notes_list:
                if n.title.lower() != target_title.lower():
                    continue
                if link_text in n.raw_content:
                    break
                new_content = n.raw_content.rstrip() + f'\n\nSee also: {link_text}\n'
                self._storage.save_note(n.id, new_content)
                print(f"[capture] Back-link: {target_title} ← [[{source_title}]]", flush=True)
                break

    def pull_captures(self):
        """Pull pending captures and route them into the knowledge graph.

        Supabase mode: reads directly from the captures table.
        Filesystem mode: fetches from the Telegram bot HTTP API (legacy).
        Uses Claude Haiku to route captures: create new notes or append to existing ones.
        """
        try:
            from storage_supabase import SupabaseBackend
            _has_supabase = True
        except ImportError:
            _has_supabase = False

        if _has_supabase and isinstance(self._storage, SupabaseBackend):
            # ── Supabase: read from captures table directly ──
            resp = (self._storage._client.table('captures')
                    .select('*')
                    .eq('user_id', self._storage._user_id)
                    .eq('processed', False)
                    .order('captured_at')
                    .execute())
            rows = resp.data or []
            if not rows:
                return {'pulled': 0}
            captures = [{'text': r['text'], 'time': r['captured_at'],
                         'from': r.get('telegram_username', ''),
                         'id': r['id']} for r in rows]
        else:
            # ── Filesystem: fetch from bot HTTP API (legacy) ──
            bot_url = self._storage.get_config('capture_bot_url')
            bot_secret = self._storage.get_config('capture_bot_secret')
            if not bot_url:
                return {'pulled': 0, 'error': 'No capture_bot_url configured'}

            import urllib.request
            import urllib.error

            try:
                req = urllib.request.Request(
                    f'{bot_url}/captures',
                    headers={'Authorization': f'Bearer {bot_secret}'} if bot_secret else {}
                )
                with urllib.request.urlopen(req, timeout=5) as resp_http:
                    data = json.loads(resp_http.read().decode('utf-8'))
            except (urllib.error.URLError, Exception) as e:
                return {'pulled': 0, 'error': str(e)}

            captures = data.get('captures', [])
            if not captures:
                return {'pulled': 0}

        # Ask Haiku to route captures
        capture_texts = [c.get('text', '') for c in captures if c.get('text', '').strip()]
        actions = self._route_captures(capture_texts)

        created_notes = []
        appended_notes = []

        if actions:
            for action in actions:
                # Gather the capture dicts for this action's indices
                action_captures = []
                for idx in action.get('capture_indices', []):
                    if 0 <= idx < len(captures):
                        action_captures.append(captures[idx])

                if not action_captures:
                    continue

                cap_lines = self._format_capture_lines(action_captures)
                raw_cap_text = '\n'.join(cap_lines)

                # Generate summary from raw capture texts
                action_texts = [c.get('text', '') for c in action_captures if c.get('text', '').strip()]
                summary = self._summarize_captures(action_texts, title=action.get('title', ''))

                # Build wikilinks block
                wiki_lines = []
                if action.get('links'):
                    wiki_lines.append('')
                    for link in action['links']:
                        wiki_lines.append(f'[[{link}]]')
                wiki_text = '\n'.join(wiki_lines)

                today = time.strftime('%Y-%m-%d')

                if action['action'] == 'create':
                    title = action['title']
                    filename = title_to_filename(title)
                    node_id = filename_to_id(filename)

                    # Deduplicate slug: add suffix (-2, -3, ...) if note exists
                    if self._storage.note_exists(node_id):
                        base_id = node_id
                        base_filename = filename.replace('.md', '')
                        suffix = 2
                        while self._storage.note_exists(node_id):
                            node_id = f'{base_id}-{suffix}'
                            filename = f'{base_filename}-{suffix}.md'
                            title = f'{action["title"]} {suffix}'
                            suffix += 1
                    if True:
                        summary_section = f'## Summary\n{summary}\n\n' if summary else ''
                        content = f"""---
title: "{title}"
tags: ["capture"]
created: {today}
---

# {title}

{summary_section}## Raw Captures
{raw_cap_text}
{wiki_text}
"""
                        self._storage.create_note(node_id, filename, content)
                        created_notes.append(title)

                    # Write back-links: update linked-to notes to reference this note
                    if action.get('backlinks'):
                        self._write_backlinks(action['backlinks'], title)

                elif action['action'] == 'append':
                    # Find the existing note by title
                    target_title = action['title']
                    target_note = None
                    for n in self._storage.list_notes():
                        if n.title.lower() == target_title.lower():
                            target_note = n
                            break

                    if target_note:
                        append_block = f'\n## Capture ({today}) — Summary\n{summary}\n\n## Capture ({today}) — Raw\n{raw_cap_text}\n' if summary else f'\n## Capture ({today})\n{raw_cap_text}\n'
                        self._storage.save_note(target_note.id, target_note.raw_content + append_block + wiki_text + '\n')
                        appended_notes.append(target_title)
                        # Write back-links for appended content
                        if action.get('backlinks'):
                            self._write_backlinks(action['backlinks'], target_title)
                    else:
                        # Existing note not found — create instead
                        title = target_title
                        filename = title_to_filename(title)
                        node_id = filename_to_id(filename)

                        # Deduplicate slug
                        if self._storage.note_exists(node_id):
                            base_id = node_id
                            base_filename = filename.replace('.md', '')
                            suffix = 2
                            while self._storage.note_exists(node_id):
                                node_id = f'{base_id}-{suffix}'
                                filename = f'{base_filename}-{suffix}.md'
                                title = f'{target_title} {suffix}'
                                suffix += 1

                        summary_section = f'## Summary\n{summary}\n\n' if summary else ''
                        content = f"""---
title: "{title}"
tags: ["capture"]
created: {today}
---

# {title}

{summary_section}## Raw Captures
{raw_cap_text}
{wiki_text}
"""
                        self._storage.create_note(node_id, filename, content)
                        created_notes.append(title)
                        # Write back-links for fallback-created note
                        if action.get('backlinks'):
                            self._write_backlinks(action['backlinks'], title)

        else:
            # Fallback: no AI available, derive title from capture content
            today = time.strftime('%Y-%m-%d')
            first_text = next((c.get('text', '') for c in captures if c.get('text', '').strip()), '')
            derived_title = title_from_capture_text(first_text)
            title = derived_title if derived_title else today
            filename = title_to_filename(title)
            node_id = filename_to_id(filename)

            # Deduplicate slug if note already exists
            if self._storage.note_exists(node_id):
                base_id = node_id
                base_filename = filename.replace('.md', '')
                base_title = title
                suffix = 2
                while self._storage.note_exists(node_id):
                    node_id = f'{base_id}-{suffix}'
                    filename = f'{base_filename}-{suffix}.md'
                    title = f'{base_title} {suffix}'
                    suffix += 1

            cap_lines = self._format_capture_lines(captures)
            cap_text = '\n'.join(cap_lines)

            content = f"""---
title: "{title}"
tags: ["capture"]
created: {today}
---

# {title}

{cap_text}
"""
            self._storage.create_note(node_id, filename, content)
            created_notes.append(title)

        # Mark captures as processed
        if _has_supabase and isinstance(self._storage, SupabaseBackend):
            # Supabase: mark rows as processed
            capture_ids = [c['id'] for c in captures if 'id' in c]
            if capture_ids:
                (self._storage._client.table('captures')
                 .update({'processed': True})
                 .in_('id', capture_ids)
                 .execute())
        else:
            # Filesystem: clear from bot HTTP API
            try:
                import urllib.request
                clear_req = urllib.request.Request(
                    f'{bot_url}/captures',
                    method='DELETE',
                    headers={'Authorization': f'Bearer {bot_secret}'} if bot_secret else {}
                )
                urllib.request.urlopen(clear_req, timeout=5)
            except Exception:
                pass

        print(f"[capture] Created: {created_notes}, Appended: {appended_notes}", flush=True)
        return {
            'pulled': len(captures),
            'created': created_notes,
            'appended': appended_notes,
        }

    # ===== Web Clipper =====

    def clip_url(self, url):
        """Fetch a URL, extract text, summarize with Claude, find wikilinks."""
        import urllib.request

        html_content = None

        # Try basic fetch first
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Heartwood/1.0'
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or 'utf-8'
                html_content = raw.decode(charset, errors='replace')
        except Exception:
            html_content = None

        # Extract from basic fetch
        title, body = ('', '')
        if html_content:
            title, body = self._extract_text_from_html(html_content)

        # If basic fetch failed or got too little content, try Playwright
        if len(body.strip()) < 200:
            pw_html = self._fetch_with_playwright(url)
            if pw_html:
                pw_title, pw_body = self._extract_text_from_html(pw_html)
                if len(pw_body.strip()) > len(body.strip()):
                    title, body = pw_title or title, pw_body

        if not body.strip():
            return {'error': f'Could not extract content from {url}'}

        # Get existing note titles for wikilink context
        existing_notes = self._get_all_note_titles()

        # Try to summarize with Claude API
        api_key = self._get_api_key()
        print(f"[clip] Extracted {len(body)} chars, API key {'found' if api_key else 'NOT found'}", flush=True)
        if api_key:
            print(f"[clip] Sending to Claude for summarization...", flush=True)
            summary = self._summarize_with_claude(body, title, url, existing_notes, api_key)
            if summary:
                print(f"[clip] Got summary: {len(summary)} chars", flush=True)
                body = summary
            else:
                print(f"[clip] Summarization returned None, using raw text", flush=True)

        # Find wikilinks in the final body
        wikilinks = self._find_wikilink_matches(body)

        return {
            'title': title or url.split('//')[-1].split('/')[0],
            'body': body,
            'wikilinks': wikilinks,
            'source_url': url,
        }

    def _get_api_key(self):
        """Get Anthropic API key from user config, falling back to env var."""
        return self._storage.get_config('anthropic_api_key') or os.environ.get('ANTHROPIC_API_KEY', '')

    def _get_all_note_titles(self):
        """Get all existing note titles for wikilink suggestions."""
        return [n.title for n in self._storage.list_notes() if n.title]

    def _get_graph_index(self):
        """Get titles, tags, and wikilinks for every note — used for smart capture routing."""
        index = []
        for n in self._storage.list_notes():
            if not n.title:
                continue
            tags = n.tags
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            links = extract_wikilinks(n.raw_content)
            index.append({
                'title': n.title,
                'tags': tags,
                'links': links,
                'filename': n.file,
            })
        return index

    def _clean_extracted_text(self, text):
        """Strip garbage characters, collapse whitespace, remove non-content noise."""
        # Remove null bytes and control characters (keep newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Remove common HTML artifacts
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        # Remove lines that are just symbols/noise (no letters)
        lines = text.split('\n')
        clean_lines = [l for l in lines if re.search(r'[a-zA-Z]{2,}', l)]
        text = '\n'.join(clean_lines)
        # Collapse excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()

    def _summarize_with_claude(self, body, title, url, existing_notes, api_key):
        """Send page text to Claude Haiku for markdown summarization."""
        import urllib.request

        # Clean the text first
        text = self._clean_extracted_text(body)
        # Cap input to ~50K chars to stay within context limits
        text = text[:50000]

        if len(text) < 50:
            print(f"[clip] Text too short after cleaning ({len(text)} chars), skipping summarization")
            return None

        notes_list = ', '.join(existing_notes[:100]) if existing_notes else 'None'
        # Ensure all strings are safe for encoding
        notes_list = notes_list.encode('ascii', 'replace').decode('ascii')
        safe_title = title.encode('ascii', 'replace').decode('ascii')
        safe_text = text.encode('ascii', 'replace').decode('ascii')

        prompt = f"""You are creating a knowledge base note from a web page. Your job is to read all the content and produce a clean, useful markdown document that captures what matters.

Page title: {safe_title}
Source URL: {url}

Existing notes in the knowledge base (create [[Note Title]] wikilinks where the content genuinely connects):
{notes_list}

Instructions:
- Read the ENTIRE page content below and understand what it's about
- Ignore navigation text, ads, cookie notices, boilerplate, and gibberish
- Produce a well-structured markdown note with:
  - ## section headers for major topics
  - Bullet points for key facts, arguments, and data
  - > blockquotes for notable quotes worth preserving
  - **Bold** for important terms and concepts
- For long content: distill into the key insights, facts, and takeaways (aim for 500-1500 words)
- For short/reference content: preserve the substance in clean markdown
- Add [[wikilinks]] only where there's a genuine connection to existing notes
- Write in a neutral, informative tone — this is reference material
- Do NOT include the page title as a heading (added separately)
- Do NOT include any preamble or meta-commentary
- Start directly with the content

Page content:
{safe_text}"""

        payload = json.dumps({
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 4096,
            'messages': [{'role': 'user', 'content': prompt}]
        }).encode('utf-8')

        try:
            import http.client
            import ssl
            conn = http.client.HTTPSConnection('api.anthropic.com', timeout=60,
                                                context=ssl.create_default_context())
            conn.request('POST', '/v1/messages', body=payload, headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            })
            resp = conn.getresponse()
            raw = resp.read().decode('utf-8')
            conn.close()

            if resp.status != 200:
                print(f"[clip] Claude API HTTP {resp.status}: {raw[:500]}", flush=True)
                return None

            result = json.loads(raw)
            if result.get('content') and len(result['content']) > 0:
                summary = result['content'][0]['text']
                print(f"[clip] Claude summarized {len(text)} chars to {len(summary)} chars", flush=True)
                return summary
            elif result.get('error'):
                print(f"[clip] Claude API returned error: {result['error']}", flush=True)
        except Exception as e:
            print(f"[clip] Claude API error: {type(e).__name__}: {e}", flush=True)
        return None

    def _fetch_with_playwright(self, url):
        """Fetch a URL using Playwright for JS-rendered pages. Returns HTML or None."""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until='networkidle', timeout=20000)
                html_content = page.content()
                browser.close()
                return html_content
        except Exception:
            return None

    def _extract_text_from_html(self, html_content):
        """Extract page title and readable text from HTML."""
        import html.parser

        class TextExtractor(html.parser.HTMLParser):
            def __init__(self):
                super().__init__()
                self.title = ''
                self.text_parts = []
                self._in_title = False
                self._skip_tags = {'script', 'style', 'noscript', 'nav', 'footer', 'header'}
                self._skip_depth = 0

            def handle_starttag(self, tag, attrs):
                if tag == 'title':
                    self._in_title = True
                if tag in self._skip_tags:
                    self._skip_depth += 1
                if tag in ('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'blockquote'):
                    self.text_parts.append('\n')

            def handle_endtag(self, tag):
                if tag == 'title':
                    self._in_title = False
                if tag in self._skip_tags and self._skip_depth > 0:
                    self._skip_depth -= 1

            def handle_data(self, data):
                if self._in_title:
                    self.title += data
                elif self._skip_depth == 0:
                    stripped = data.strip()
                    if stripped:
                        self.text_parts.append(stripped)

        parser = TextExtractor()
        try:
            parser.feed(html_content)
        except Exception:
            pass

        title = parser.title.strip()
        body = ' '.join(parser.text_parts)
        # Collapse whitespace
        body = re.sub(r'\n\s*\n', '\n\n', body)
        body = re.sub(r' +', ' ', body).strip()

        return title, body

    def _find_wikilink_matches(self, text):
        """Find existing note titles that appear in the given text."""
        text_lower = text.lower()
        matches = []
        for n in self._storage.list_notes():
            if len(n.title) >= 4 and n.title.lower() in text_lower:
                matches.append(n.title)
        return matches

    def extract_clip(self, text, source_url=''):
        """Use Haiku to extract structured note data from raw captured text.

        Returns: {title, body, tags, wikilinks, type}
        """
        import anthropic

        # Gather existing tags and note titles for context
        existing_tags = self.get_all_tags()
        existing_titles = [n.title for n in self._storage.list_notes()]

        # Load ontology types for category inference
        ontology = self.get_ontology()
        type_names = list(ontology.get('types', {}).keys()) if isinstance(ontology, dict) else []

        prompt = f"""You are a knowledge graph assistant. Extract structured note data from the following captured text.

Existing tags in the graph: {json.dumps(existing_tags[:100])}
Existing note titles (for wikilink suggestions): {json.dumps(existing_titles[:200])}
Valid note types: {json.dumps(type_names)}

Source URL: {source_url or 'unknown'}

Raw captured text:
---
{text[:4000]}
---

Return a JSON object with these fields:
- "title": concise, descriptive title (max 80 chars)
- "body": cleaned/summarized version of the text (keep key information, remove noise)
- "tags": array of 2-5 tags (prefer existing tags when they fit, can add new ones)
- "wikilinks": array of existing note titles this text relates to (only from the list above, max 5)
- "type": the most fitting ontology type from the valid types list

Return ONLY the JSON object, no markdown fences or explanation."""

        try:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=1024,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
                if raw.endswith('```'):
                    raw = raw[:-3]
                raw = raw.strip()
            result = json.loads(raw)
            # Validate and sanitize
            return {
                'title': str(result.get('title', 'Untitled'))[:80],
                'body': str(result.get('body', text)),
                'tags': [str(t) for t in result.get('tags', ['clip'])][:10],
                'wikilinks': [t for t in result.get('wikilinks', []) if t in existing_titles][:5],
                'type': str(result.get('type', '')) if result.get('type') in type_names else '',
            }
        except Exception as exc:
            # Fallback: return raw text with minimal structure
            first_line = text.split('\n')[0].replace('#', '').strip()[:80] or 'Untitled Capture'
            return {
                'title': first_line,
                'body': text,
                'tags': ['clip'],
                'wikilinks': [],
                'type': '',
                'error': str(exc),
            }

    def save_clip(self, title, body, wikilinks, source_url, tags=None):
        """Save clipped content as a new Heartwood note + dual-write to Claude Code memory."""
        filename = title_to_filename(title)
        node_id = filename_to_id(filename)

        if self._storage.note_exists(node_id):
            return {'error': 'Note already exists', 'id': node_id}

        today = time.strftime('%Y-%m-%d')
        tag_list = tags if tags else ['clip']

        # Build wikilink references
        links_section = ''
        if wikilinks:
            links_section = '\n\n## Related Notes\n\n' + '\n'.join(
                f'- [[{w}]]' for w in wikilinks
            )

        tag_str = ', '.join(tag_list)
        content = f"""---
title: {title}
tags: [{tag_str}]
created: {today}
---

# {title}

> Clipped from [{source_url}]({source_url})

{body}{links_section}
"""
        result = self._storage.create_note(node_id, filename, content)

        # Dual-write: also save to Claude Code memory as a reference
        self._dual_write_clip(title, source_url, body, wikilinks, today)

        return result

    def _dual_write_clip(self, title, source_url, body, wikilinks, date):
        """Write a clip to Claude Code memory directory as a reference file."""
        try:
            memory_dir = os.environ.get('HEARTWOOD_MEMORY_DIR', '') or os.path.join(
                os.path.expanduser('~'), '.heartwood', 'memory'
            )
            memory_dir = os.path.normpath(memory_dir)
            if not os.path.isdir(memory_dir):
                return

            # Build memory filename with clip_ prefix for easy identification
            safe_name = 'clip_' + re.sub(r'[^\w\s-]', '', title.lower()).strip()
            safe_name = re.sub(r'[\s-]+', '_', safe_name)[:80]
            mem_filename = safe_name + '.md'
            mem_path = os.path.join(memory_dir, mem_filename)

            # Summary: first 300 chars of body
            summary = body[:300].strip()
            if len(body) > 300:
                summary += '...'

            # Wikilink references
            related = ''
            if wikilinks:
                related = '\n\nRelated notes: ' + ', '.join(f'[[{w}]]' for w in wikilinks)

            mem_content = f"""---
name: {title}
description: Web clip from {source_url}
type: reference
source: heartwood-clip
---

# {title}

Clipped from: {source_url}
Date: {date}

{summary}{related}
"""
            with open(mem_path, 'w', encoding='utf-8') as fh:
                fh.write(mem_content)

            # Append to MEMORY.md index
            memory_md = os.path.join(memory_dir, 'MEMORY.md')
            if os.path.exists(memory_md):
                with open(memory_md, 'r', encoding='utf-8') as fh:
                    existing = fh.read()
                # Add clip reference if not already there
                entry = f'\n- [{title}]({mem_filename}) — web clip from {source_url}'
                if mem_filename not in existing:
                    # Add under a Clips section, create it if needed
                    if '## Clips' not in existing:
                        existing = existing.rstrip() + '\n\n## Clips\n'
                    existing += entry + '\n'
                    with open(memory_md, 'w', encoding='utf-8') as fh:
                        fh.write(existing)
        except Exception as e:
            print(f"[dual-write] Error writing clip to memory: {e}")


def run_sync_quiet():
    """Run sync.py --apply silently on startup and return count of changes."""
    try:
        from sync import discover_sources, get_existing_notes, check_sync_status, create_note_from_source
        sources = discover_sources()
        existing = get_existing_notes()
        actions = check_sync_status(sources, existing)
        if not actions:
            return 0
        notes_path = os.path.join(_app_dir(), 'notes')
        os.makedirs(notes_path, exist_ok=True)
        for a in actions:
            content = create_note_from_source(a['source'])
            filepath = os.path.join(notes_path, a['note_filename'])
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        return len(actions)
    except Exception as e:
        print(f"[sync] Error: {e}")
        return 0


class FileWatcher:
    """Watches directories for file changes and triggers frontend refresh."""

    def __init__(self, window, api_instance):
        self.window = window
        self.api = api_instance
        self._running = False
        self._thread = None
        self._snapshot = {}

    def _get_watch_dirs(self):
        """Directories to watch for changes."""
        dirs = [self.api._notes_path]
        # Also watch source directories that sync.py scans
        try:
            from sync import PARENT_DIR, MEMORY_DIR
            candidates = [
                os.path.join(PARENT_DIR, '.claude', 'skills'),
                os.path.join(PARENT_DIR, '.claude', 'agents'),
                MEMORY_DIR,
            ]
            for d in candidates:
                if os.path.isdir(d):
                    dirs.append(d)
        except ImportError:
            pass
        return dirs

    def _snapshot_dirs(self):
        """Take a snapshot of all watched .md files and their mtimes."""
        snap = {}
        for d in self._get_watch_dirs():
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith('.md'):
                        fp = os.path.join(root, f)
                        try:
                            snap[fp] = os.path.getmtime(fp)
                        except OSError:
                            pass
        return snap

    def _check_changes(self):
        """Compare current state to snapshot, return True if changed."""
        new_snap = self._snapshot_dirs()
        changed = new_snap != self._snapshot
        self._snapshot = new_snap
        return changed

    def _watch_loop(self):
        """Background polling loop — checks every 5 seconds."""
        import time as _time
        self._snapshot = self._snapshot_dirs()
        while self._running:
            _time.sleep(5)
            if not self._running:
                break
            try:
                if self._check_changes():
                    # Re-run sync to pull in any new external files
                    count = run_sync_quiet()
                    if count > 0:
                        self._snapshot = self._snapshot_dirs()
                    # Tell the frontend to refresh
                    try:
                        self.window.evaluate_js('if(typeof loadGraph==="function") loadGraph();')
                    except Exception:
                        pass
            except Exception as e:
                print(f"[watcher] Error: {e}")

    def start(self):
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


if __name__ == '__main__':
    # Run sync on startup
    synced = run_sync_quiet()
    if synced:
        print(f"[sync] {synced} notes synced on startup")

    api = Api()

    # Pull captures from Telegram bot on startup
    try:
        result = api.pull_captures()
        if result.get('pulled', 0) > 0:
            print(f"[captures] Pulled {result['pulled']} captures into daily note")
    except Exception as e:
        print(f"[captures] {e}")

    html_path = resource('index.html')
    window = webview.create_window(
        'Heartwood',
        url=f'file:///{html_path}',
        js_api=api,
        width=1400,
        height=900,
        min_size=(900, 600),
        resizable=True,
    )

    # Start file watcher after window is ready
    watcher = FileWatcher(window, api)

    def on_loaded():
        watcher.start()

    window.events.loaded += on_loaded
    webview.start(debug=True)
    watcher.stop()
