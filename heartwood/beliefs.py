"""Heartwood Belief Extraction — Phase 4a of Reasoning Engine v2.

Extracts atomic claims from knowledge graph notes. Each claim is a single,
falsifiable statement with confidence, temporal scope, and source type.

Claims are cached by note mtime in .graph/beliefs.json. Re-extracted only
when the source note is modified.

Usage:
    python cerebro/beliefs.py                        # extract from all notes
    python cerebro/beliefs.py --pilot                # 20-note pilot run
    python cerebro/beliefs.py --pilot --model sonnet # pilot with Sonnet
    python cerebro/beliefs.py --note firm-intel      # single note
    python cerebro/beliefs.py --stats                # show extraction stats
"""

import os
import sys
import json
import hashlib
import logging
import datetime
import argparse
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Add parent dir so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import parse_frontmatter, filename_to_id, notes_dir

logger = logging.getLogger('cerebro.beliefs')


# ── Pydantic Models ──────────────────────────────────────────────────────

class Claim(BaseModel):
    id: str                                  # hash of (source_note, text)
    text: str                                # atomic claim
    source_note: str                         # note ID
    source_type: str = 'direct'              # direct|research|hearsay|capture|inferred
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: Optional[str] = None         # ISO date or None
    valid_until: Optional[str] = None        # ISO date or None
    extracted_at: str = ''                   # ISO datetime
    status: str = 'in'                       # in|out|contested

    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v):
        allowed = {'direct', 'research', 'hearsay', 'capture', 'inferred'}
        if v not in allowed:
            return 'direct'
        return v

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        allowed = {'in', 'out', 'contested'}
        if v not in allowed:
            return 'in'
        return v


class NoteBeliefs(BaseModel):
    """Claims extracted from a single note, cached by mtime."""
    note_id: str
    mtime: float                             # source note mtime at extraction
    model: str = ''                          # model used for extraction
    claims: list[Claim] = Field(default_factory=list)
    extraction_error: Optional[str] = None   # non-None if extraction failed


class BeliefsStore(BaseModel):
    """All extracted claims, keyed by note_id."""
    version: int = 1
    notes: dict[str, NoteBeliefs] = Field(default_factory=dict)


# ── Paths ────────────────────────────────────────────────────────────────

def _graph_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.graph')


def _beliefs_path():
    return os.path.join(_graph_path(), 'beliefs.json')


def _notes_path():
    return notes_dir()


# ── Store I/O ────────────────────────────────────────────────────────────

def load_store() -> BeliefsStore:
    path = _beliefs_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return BeliefsStore.model_validate(json.load(f))
    return BeliefsStore()


def save_store(store: BeliefsStore):
    path = _beliefs_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(store.model_dump(), f, indent=2, default=str)


# ── Claim ID ─────────────────────────────────────────────────────────────

def claim_id(source_note: str, text: str) -> str:
    """Deterministic hash from source note + claim text."""
    raw = f"{source_note}::{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── API Caller ───────────────────────────────────────────────────────────

def _call_claude(prompt: str, api_key: str, model: str = 'claude-haiku-4-5-20251001',
                 max_tokens: int = 2048) -> str:
    """Call Claude API and return text response. Works with both Haiku and Sonnet."""
    import http.client
    import ssl

    payload = json.dumps({
        'model': model,
        'max_tokens': max_tokens,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode('utf-8')

    conn = http.client.HTTPSConnection(
        'api.anthropic.com', timeout=90,
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
        raise RuntimeError(f'Claude API returned HTTP {resp.status}: {raw[:300]}')

    result = json.loads(raw)
    if result.get('content') and len(result['content']) > 0:
        return result['content'][0]['text']
    raise RuntimeError('Empty Claude response')


# ── Extraction ───────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract atomic claims from this note. Each claim should be:
- A single, falsifiable statement (not a question, not a wish, not a section header)
- Attributed with confidence (0.0-1.0) based on the author's certainty level:
  - 0.9-1.0: stated as fact, verifiable
  - 0.7-0.8: stated with conviction but not cited
  - 0.4-0.6: uncertain, hedged, or speculative language
  - 0.1-0.3: feelings, hunches, or explicitly uncertain
- Scoped temporally if the note implies a time range (ISO date format YYYY-MM-DD)
- Typed by source:
  - direct: first-hand experience or decision by the author
  - research: cited from external sources
  - hearsay: heard from others, secondhand
  - capture: quick note or raw thought capture
  - inferred: derived or concluded by the author from other information

IMPORTANT NUANCE RULES:
- Preserve hedging language. "I'm thinking about X" → confidence 0.4-0.5, NOT "I will do X"
- "feels like" / "seems like" → confidence 0.2-0.3, source_type: direct
- Separate compound statements into individual claims
- Do NOT extract questions, wishes, section headers, or meta-commentary about the note itself
- Do NOT hallucinate claims — every claim must be directly supported by the note text
- For strategy/plan notes, extract the claims about what IS, not what SHOULD be done

Note title: {title}
Note type: {note_type}
Note tags: {tags}
Note content:
{body}

Return at most 25 claims. Prioritize the most important and distinctive claims.
Return ONLY a JSON array. Each element: {{"text": "...", "confidence": 0.X, "source_type": "...", "valid_from": "YYYY-MM-DD or null", "valid_until": "YYYY-MM-DD or null"}}
No markdown fences, no explanation — just the JSON array."""


def extract_claims_from_note(note_id: str, title: str, note_type: str,
                             tags: list[str], body: str, api_key: str,
                             model: str = 'claude-haiku-4-5-20251001') -> list[Claim]:
    """Extract claims from a single note via Claude API."""
    prompt = EXTRACTION_PROMPT.format(
        title=title,
        note_type=note_type or 'unknown',
        tags=', '.join(tags) if tags else 'none',
        body=body[:6000],  # cap body to avoid token limits
    )

    raw = _call_claude(prompt, api_key, model=model, max_tokens=4096)

    # Parse JSON — strip markdown fences if model adds them anyway
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

    # Try to parse, with fallback repair for common LLM JSON issues
    try:
        claims_data = json.loads(text)
    except json.JSONDecodeError:
        import re
        # Repair: remove trailing commas before ] or }
        repaired = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            claims_data = json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: find the outermost JSON array
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                chunk = text[start:end+1]
                chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
                claims_data = json.loads(chunk)
            else:
                raise

    if not isinstance(claims_data, list):
        raise ValueError(f'Expected JSON array, got {type(claims_data).__name__}')

    now = datetime.datetime.now().isoformat()
    claims = []
    for item in claims_data:
        if not isinstance(item, dict) or 'text' not in item:
            continue
        cid = claim_id(note_id, item['text'])
        claim = Claim(
            id=cid,
            text=item['text'],
            source_note=note_id,
            source_type=item.get('source_type', 'direct'),
            confidence=max(0.0, min(1.0, float(item.get('confidence', 0.5)))),
            valid_from=item.get('valid_from'),
            valid_until=item.get('valid_until'),
            extracted_at=now,
            status='in',
        )
        claims.append(claim)

    return claims


# ── Batch Extraction ─────────────────────────────────────────────────────

def get_api_key():
    cfg_path = os.path.join(_graph_path(), 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            key = json.load(f).get('anthropic_api_key', '')
            if key:
                return key
    return os.environ.get('ANTHROPIC_API_KEY', '')


def load_note(filepath: str):
    """Load a note file, return (meta, body) or None on error."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        meta, body = parse_frontmatter(content)
        return meta, body
    except Exception as e:
        logger.warning(f'Failed to load {filepath}: {e}')
        return None


def extract_all(note_ids: Optional[list[str]] = None, model: str = 'claude-haiku-4-5-20251001',
                force: bool = False) -> dict:
    """Extract claims from notes. Returns stats dict.

    Args:
        note_ids: specific notes to process (None = all)
        model: Claude model to use
        force: if True, re-extract even if cached
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError('No API key found in .graph/config.json or ANTHROPIC_API_KEY env var')

    store = load_store()
    npath = _notes_path()

    stats = {'extracted': 0, 'cached': 0, 'failed': 0, 'total_claims': 0}

    files = []
    for f in sorted(os.listdir(npath)):
        if not f.endswith('.md'):
            continue
        nid = filename_to_id(f)
        if note_ids and nid not in note_ids:
            continue
        files.append((f, nid))

    for i, (fname, nid) in enumerate(files):
        filepath = os.path.join(npath, fname)
        mtime = os.path.getmtime(filepath)

        # Check cache
        if not force and nid in store.notes:
            cached = store.notes[nid]
            if abs(cached.mtime - mtime) < 1.0 and cached.model == model:
                stats['cached'] += 1
                stats['total_claims'] += len(cached.claims)
                continue

        # Load and extract
        result = load_note(filepath)
        if not result:
            stats['failed'] += 1
            continue

        meta, body = result
        title = str(meta.get('title', nid.replace('-', ' ').title()))
        tags = meta.get('tags', [])
        note_type = meta.get('type', '')
        if not note_type and tags:
            note_type = tags[0]

        print(f'  [{i+1}/{len(files)}] {nid} ...', end=' ', flush=True)

        try:
            claims = extract_claims_from_note(
                nid, title, note_type, tags, body, api_key, model=model
            )
            store.notes[nid] = NoteBeliefs(
                note_id=nid,
                mtime=mtime,
                model=model,
                claims=claims,
            )
            stats['extracted'] += 1
            stats['total_claims'] += len(claims)
            print(f'{len(claims)} claims')
        except Exception as e:
            store.notes[nid] = NoteBeliefs(
                note_id=nid,
                mtime=mtime,
                model=model,
                claims=[],
                extraction_error=str(e)[:200],
            )
            stats['failed'] += 1
            print(f'FAILED: {e}')

    save_store(store)
    return stats


# ── Pilot ────────────────────────────────────────────────────────────────

PILOT_NOTES = [
    'feedback-aesthetic-direction',
    'user-discernment-fear',
    'lidia-goals-and-credits',
    'lidia-research-koryo-saram-soviet-life',
    'user-poetry-as-release',
    'firm-intel',
    'kg-design-resources',
    'ontology-as-abstraction-layer',
    'reasoning-reflection-2026-03-25',
    'scrutinize-skill',
    'visual-contrast-moscow-vs-navrongo',
    'lidia-anecdote-catholicism',
    'ruthie',
    'tee-hunt',
    'hunt-common',
    'building-to-learn',
    'ai-native-rm',
    'cerebro-batch-upload',
    'feedback-giftedness-collaboration',
    'dobbs-ferry-preference',
]


def run_pilot(model: str = 'claude-haiku-4-5-20251001'):
    """Run extraction on 20 diverse notes with specified model."""
    print(f'\n== Pilot Extraction ({model}) ==')
    print(f'Notes: {len(PILOT_NOTES)}\n')
    stats = extract_all(note_ids=PILOT_NOTES, model=model, force=True)
    print(f'\n== Results ==')
    print(f'  Extracted: {stats["extracted"]}')
    print(f'  Failed:    {stats["failed"]}')
    print(f'  Claims:    {stats["total_claims"]}')
    avg = stats['total_claims'] / max(1, stats['extracted'])
    print(f'  Avg/note:  {avg:.1f}')
    return stats


def show_stats():
    """Print stats from existing beliefs store."""
    store = load_store()
    if not store.notes:
        print('No claims extracted yet.')
        return

    total_claims = sum(len(nb.claims) for nb in store.notes.values())
    failed = sum(1 for nb in store.notes.values() if nb.extraction_error)
    models = set(nb.model for nb in store.notes.values())

    print(f'\n== Beliefs Store Stats ==')
    print(f'  Notes processed: {len(store.notes)}')
    print(f'  Total claims:    {total_claims}')
    print(f'  Failed:          {failed}')
    print(f'  Models used:     {", ".join(models)}')
    print(f'  Avg claims/note: {total_claims / max(1, len(store.notes) - failed):.1f}')

    # Confidence distribution
    all_conf = [c.confidence for nb in store.notes.values() for c in nb.claims]
    if all_conf:
        buckets = {'0.0-0.3': 0, '0.3-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
        for c in all_conf:
            if c < 0.3: buckets['0.0-0.3'] += 1
            elif c < 0.6: buckets['0.3-0.6'] += 1
            elif c < 0.8: buckets['0.6-0.8'] += 1
            else: buckets['0.8-1.0'] += 1
        print(f'\n  Confidence distribution:')
        for k, v in buckets.items():
            pct = v / len(all_conf) * 100
            print(f'    {k}: {v} ({pct:.0f}%)')

    # Source type distribution
    all_types = [c.source_type for nb in store.notes.values() for c in nb.claims]
    if all_types:
        from collections import Counter
        print(f'\n  Source types:')
        for st, count in Counter(all_types).most_common():
            print(f'    {st}: {count}')


def show_note_claims(note_id: str):
    """Print claims for a specific note."""
    store = load_store()
    if note_id not in store.notes:
        print(f'No claims for {note_id}')
        return
    nb = store.notes[note_id]
    if nb.extraction_error:
        print(f'Extraction error: {nb.extraction_error}')
        return
    print(f'\n== Claims from {note_id} ({len(nb.claims)} claims, model: {nb.model}) ==\n')
    for i, c in enumerate(nb.claims, 1):
        temporal = ''
        if c.valid_from or c.valid_until:
            temporal = f' [{c.valid_from or "?"} -> {c.valid_until or "ongoing"}]'
        print(f'  {i}. [{c.confidence:.1f}] [{c.source_type}]{temporal}')
        print(f'     {c.text}\n')


# ── CLI ──────────────────────────────────────────────────────────────────

MODEL_MAP = {
    'haiku': 'claude-haiku-4-5-20251001',
    'sonnet': 'claude-3-5-sonnet-20241022',
}


def main():
    parser = argparse.ArgumentParser(description='Heartwood Claim Extraction')
    parser.add_argument('--pilot', action='store_true', help='Run 20-note pilot')
    parser.add_argument('--model', default='haiku', choices=['haiku', 'sonnet'],
                        help='Model to use (default: haiku)')
    parser.add_argument('--note', type=str, help='Extract from a single note')
    parser.add_argument('--stats', action='store_true', help='Show extraction stats')
    parser.add_argument('--show', type=str, help='Show claims for a specific note')
    parser.add_argument('--force', action='store_true', help='Force re-extraction')
    parser.add_argument('--all', action='store_true', help='Extract from all notes')
    args = parser.parse_args()

    model = MODEL_MAP[args.model]

    if args.stats:
        show_stats()
    elif args.show:
        show_note_claims(args.show)
    elif args.pilot:
        run_pilot(model=model)
    elif args.note:
        stats = extract_all(note_ids=[args.note], model=model, force=args.force)
        show_note_claims(args.note)
    elif args.all:
        print(f'\n== Full Extraction ({model}) ==\n')
        stats = extract_all(model=model, force=args.force)
        print(f'\n== Results ==')
        print(f'  Extracted: {stats["extracted"]}')
        print(f'  Cached:    {stats["cached"]}')
        print(f'  Failed:    {stats["failed"]}')
        print(f'  Claims:    {stats["total_claims"]}')
    else:
        parser.print_help()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
