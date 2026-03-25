"""Heartwood Belief Revision — Phase 4b of Reasoning Engine v2.

Three-layer contradiction detection with formal belief revision:
  Layer 1: Structural — temporal overlap, duplicate claims (no LLM)
  Layer 2: Embedding — high similarity + polarity inversion (no LLM)
  Layer 3: LLM confirmation — Claude classifies candidates (costs ~$0.003/pair)

Plus AGM-style entrenchment ordering and justification tracking.

Usage:
    python cerebro/revision.py                  # detect contradictions across all claims
    python cerebro/revision.py --note X         # check claims from note X against all others
    python cerebro/revision.py --stats          # show revision stats
    python cerebro/revision.py --dry-run        # layers 1-2 only, no LLM cost
    python cerebro/revision.py --reverse         # preview reversal of pre-fix revisions
    python cerebro/revision.py --reverse --apply # execute reversal
    python cerebro/revision.py --reverse 2026-03-25T15:00  # custom cutoff
"""

import os
import sys
import json
import logging
import datetime
import argparse
import re
from typing import Optional
from collections import defaultdict

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import EmbeddingCache, _cosine_similarity
from beliefs import load_store, save_store, Claim, BeliefsStore

logger = logging.getLogger('cerebro.revision')


# ── Pydantic Models ──────────────────────────────────────────────────────

class Justification(BaseModel):
    """Links a claim to its supporting evidence (JTMS-inspired)."""
    claim_id: str
    supporting_claims: list[str] = Field(default_factory=list)
    supporting_notes: list[str] = Field(default_factory=list)
    type: str = 'direct'  # direct|inferred|corroborated


class ContradictionCandidate(BaseModel):
    """A pair of claims suspected of contradicting."""
    claim_a_id: str
    claim_b_id: str
    claim_a_text: str = ''
    claim_b_text: str = ''
    source_a: str = ''
    source_b: str = ''
    layer: int = 1                      # which layer detected it
    reason: str = ''                    # why it's a candidate
    similarity: float = 0.0
    status: str = 'candidate'           # candidate|confirmed|dismissed|unconfirmed
    resolution: Optional[str] = None    # which claim to retract, or None


class RevisionLog(BaseModel):
    """Record of a belief revision event."""
    timestamp: str
    retracted_claim_id: str
    retracted_text: str
    retained_claim_id: str
    retained_text: str
    reason: str
    entrenchment_retracted: float
    entrenchment_retained: float


class RevisionStore(BaseModel):
    """Persisted revision state."""
    version: int = 1
    contradictions: list[ContradictionCandidate] = Field(default_factory=list)
    revisions: list[RevisionLog] = Field(default_factory=list)
    last_run: str = ''


# ── Paths ────────────────────────────────────────────────────────────────

def _graph_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.graph')


def _revision_path():
    return os.path.join(_graph_path(), 'revisions.json')


def load_revision_store() -> RevisionStore:
    path = _revision_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return RevisionStore.model_validate(json.load(f))
    return RevisionStore()


def save_revision_store(store: RevisionStore):
    path = _revision_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(store.model_dump(), f, indent=2, default=str)


# ── Entrenchment ─────────────────────────────────────────────────────────

# Tunable weights — initial values per architecture doc section 9.2
ENTRENCHMENT_WEIGHTS = {
    'source_type': {
        'direct': 0.9,
        'research': 0.7,
        'inferred': 0.5,
        'hearsay': 0.4,
        'capture': 0.3,
    },
    'corroboration_boost': 0.05,   # per corroborating note
    'recency_decay': 0.001,        # per day old
    'degree_boost': 0.02,          # per connection (capped at 10)
}


def compute_entrenchment(claim: Claim, adjacency: dict, all_claims: list[Claim]) -> float:
    """Compute how entrenched a claim is — higher = harder to retract.

    Factors: source type base score + corroboration + recency - decay + connectivity.
    """
    w = ENTRENCHMENT_WEIGHTS

    # Base score from source type
    score = w['source_type'].get(claim.source_type, 0.5)

    # Corroboration: count other notes with similar claims (same text fuzzy match)
    claim_lower = claim.text.lower().strip()
    corroborating = 0
    for other in all_claims:
        if other.id == claim.id or other.source_note == claim.source_note:
            continue
        # Simple word overlap as proxy for corroboration
        words_a = set(claim_lower.split())
        words_b = set(other.text.lower().strip().split())
        if len(words_a) > 2 and len(words_b) > 2:
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap > 0.6:
                corroborating += 1
    score += w['corroboration_boost'] * min(corroborating, 5)

    # Recency decay
    try:
        extracted = datetime.datetime.fromisoformat(claim.extracted_at)
        age_days = (datetime.datetime.now() - extracted).days
    except Exception:
        age_days = 30
    score -= w['recency_decay'] * age_days

    # Connection density of source note
    source_degree = len(adjacency.get(claim.source_note, set()))
    score += w['degree_boost'] * min(source_degree, 10)

    # Confidence as a factor
    score *= claim.confidence

    return max(0.0, min(1.0, score))


# ── Layer 1: Structural Detection ────────────────────────────────────────

def _parse_date(s: Optional[str]) -> Optional[datetime.date]:
    if not s or s == 'null':
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except Exception:
        return None


def _temporal_overlap(a: Claim, b: Claim) -> bool:
    """Check if two claims have overlapping temporal windows."""
    a_from = _parse_date(a.valid_from)
    a_until = _parse_date(a.valid_until)
    b_from = _parse_date(b.valid_from)
    b_until = _parse_date(b.valid_until)

    # If neither has temporal scope, they could overlap (assume current)
    if not a_from and not b_from:
        return True
    # If one has no start, assume it started long ago
    if not a_from:
        a_from = datetime.date(2000, 1, 1)
    if not b_from:
        b_from = datetime.date(2000, 1, 1)
    # If no end, assume ongoing
    if not a_until:
        a_until = datetime.date(2099, 12, 31)
    if not b_until:
        b_until = datetime.date(2099, 12, 31)

    return a_from <= b_until and b_from <= a_until


def _extract_entities(text: str) -> set[str]:
    """Extract potential entity mentions from claim text (capitalized phrases, quoted terms)."""
    entities = set()
    # Capitalized multi-word phrases (e.g., "Two Sigma", "Dobbs Ferry")
    for match in re.finditer(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text):
        entities.add(match.group().lower())
    # Single capitalized words that aren't sentence starters (heuristic)
    words = text.split()
    for i, w in enumerate(words):
        if i > 0 and w[0:1].isupper() and len(w) > 2:
            entities.add(w.lower().rstrip('.,;:'))
    return entities


def layer1_structural(claims: list[Claim]) -> list[ContradictionCandidate]:
    """Detect contradictions via structural signals — no LLM needed.

    Checks:
    1. Same entity + overlapping time windows + conflicting attributes
    2. Negation patterns (X is Y vs X is not Y)
    """
    candidates = []
    seen_pairs = set()

    # Index claims by entity mentions for efficient pairing
    entity_index = defaultdict(list)
    for claim in claims:
        for entity in _extract_entities(claim.text):
            entity_index[entity].append(claim)

    # Check entity-sharing claim pairs
    for entity, entity_claims in entity_index.items():
        for i in range(len(entity_claims)):
            for j in range(i + 1, len(entity_claims)):
                a, b = entity_claims[i], entity_claims[j]
                if a.source_note == b.source_note:
                    continue  # same note, not a contradiction
                pair_key = tuple(sorted([a.id, b.id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if not _temporal_overlap(a, b):
                    continue  # different time periods — temporal sequence, not contradiction

                # Check for negation patterns
                a_lower = a.text.lower()
                b_lower = b.text.lower()

                # Direct negation: "X is Y" vs "X is not Y"
                is_negation = False
                if 'not ' in a_lower and a_lower.replace('not ', '', 1).strip() == b_lower.strip():
                    is_negation = True
                elif 'not ' in b_lower and b_lower.replace('not ', '', 1).strip() == a_lower.strip():
                    is_negation = True

                # Opposing value patterns: "X is A" vs "X is B" where A != B
                # Simple heuristic: shared prefix + different suffix
                words_a = a_lower.split()
                words_b = b_lower.split()
                shared_prefix = 0
                for wa, wb in zip(words_a, words_b):
                    if wa == wb:
                        shared_prefix += 1
                    else:
                        break

                has_conflict = False
                if shared_prefix >= 3 and (len(words_a) > shared_prefix or len(words_b) > shared_prefix):
                    # Significant shared prefix but different ending
                    has_conflict = True

                if is_negation or has_conflict:
                    candidates.append(ContradictionCandidate(
                        claim_a_id=a.id,
                        claim_b_id=b.id,
                        claim_a_text=a.text,
                        claim_b_text=b.text,
                        source_a=a.source_note,
                        source_b=b.source_note,
                        layer=1,
                        reason='negation' if is_negation else f'shared entity "{entity}" + conflicting suffix',
                    ))

    return candidates


# ── Layer 2: Embedding-Based Detection ───────────────────────────────────

def layer2_embedding(claims: list[Claim], emb_cache: EmbeddingCache,
                     similarity_threshold: float = 0.75,
                     max_candidates: int = 50) -> list[ContradictionCandidate]:
    """Detect contradictions via high embedding similarity + polarity signals.

    Claims that are very similar but from different notes are candidates —
    they discuss the same topic and may disagree.
    """
    candidates = []

    # Embed all claims (batched via cache)
    claim_embeddings = {}
    for claim in claims:
        emb = emb_cache.embed_query(claim.text)
        if emb:
            claim_embeddings[claim.id] = (claim, emb)

    items = list(claim_embeddings.values())

    # Polarity inversion words
    negators = {'not', 'no', 'never', 'neither', 'nor', 'without',
                'unlikely', 'impossible', 'false', 'incorrect', 'wrong'}
    opposites = {
        ('increase', 'decrease'), ('grow', 'shrink'), ('rise', 'fall'),
        ('gain', 'lose'), ('up', 'down'), ('high', 'low'),
        ('success', 'failure'), ('strong', 'weak'), ('more', 'less'),
        ('better', 'worse'), ('first', 'last'), ('easy', 'hard'),
        ('safe', 'dangerous'), ('active', 'inactive'), ('open', 'closed'),
    }
    opposite_map = {}
    for a, b in opposites:
        opposite_map[a] = b
        opposite_map[b] = a

    # Compare all pairs — O(n^2) but claims are <4000 so ~8M comparisons max
    # Optimize: only compare cross-note pairs
    for i in range(len(items)):
        claim_a, emb_a = items[i]
        for j in range(i + 1, len(items)):
            claim_b, emb_b = items[j]

            # Skip same-note pairs
            if claim_a.source_note == claim_b.source_note:
                continue

            sim = _cosine_similarity(emb_a, emb_b)
            if sim < similarity_threshold:
                continue

            # Check polarity inversion
            words_a = set(claim_a.text.lower().split())
            words_b = set(claim_b.text.lower().split())

            # One has a negator the other doesn't
            neg_a = words_a & negators
            neg_b = words_b & negators
            has_polarity_flip = bool(neg_a) != bool(neg_b)

            # Opposite words present
            has_opposite = False
            for w in words_a:
                if w in opposite_map and opposite_map[w] in words_b:
                    has_opposite = True
                    break

            if has_polarity_flip or has_opposite:
                reason_parts = []
                if has_polarity_flip:
                    reason_parts.append(f'polarity flip (neg: {neg_a or neg_b})')
                if has_opposite:
                    reason_parts.append('opposite terms detected')
                candidates.append(ContradictionCandidate(
                    claim_a_id=claim_a.id,
                    claim_b_id=claim_b.id,
                    claim_a_text=claim_a.text,
                    claim_b_text=claim_b.text,
                    source_a=claim_a.source_note,
                    source_b=claim_b.source_note,
                    layer=2,
                    reason=', '.join(reason_parts),
                    similarity=round(sim, 3),
                ))

            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    # Sort by similarity descending
    candidates.sort(key=lambda c: -c.similarity)
    return candidates[:max_candidates]


# ── Layer 3: LLM Confirmation ────────────────────────────────────────────

CONFIRMATION_PROMPT = """You are evaluating potential contradictions between claims extracted from a personal knowledge graph. Be STRICT — only mark pairs as CONTRADICTORY when the claims genuinely cannot both be true.

Classify each pair:

(a) CONTRADICTORY — the claims directly conflict and cannot both be true as stated. One must be wrong or outdated. Example: "X is a manager" vs "X is an associate" (different roles).
(b) TENSION — the claims pull in different directions but COULD both be true with context. Do NOT confirm these as contradictions. Example: "She accepted the situation" vs "She was unhappy about it" (both can be true simultaneously).
(c) COMPLEMENTARY — different details or perspectives on the same topic. No conflict. Example: "X grew up in Accra" vs "X grew up in Navrongo and Accra" (one is more complete). Also applies when one claim is a more detailed version of the other, or when claims describe different events/aspects of the same subject.
(d) UNRELATED — false positive, claims are not about the same thing.

Common false positives to watch for:
- Narrative/screenplay material where multiple interpretations of the same event are INTENTIONAL (mark TENSION or COMPLEMENTARY, not CONTRADICTORY)
- One claim adds detail to another without conflicting (mark COMPLEMENTARY)
- Strategic perspectives that represent different framings of the same reality (mark TENSION)
- Claims about the same entity from different time periods (mark TENSION unless clearly superseded)

For CONTRADICTORY pairs only, suggest which claim to retain.

{pairs}

Respond with one line per pair:
[number]. [CONTRADICTORY/TENSION/COMPLEMENTARY/UNRELATED] | [brief explanation] | [retain: A or B or neither]"""


def layer3_llm(candidates: list[ContradictionCandidate], api_key: str,
               call_llm=None) -> list[ContradictionCandidate]:
    """Confirm contradiction candidates via LLM.

    Only candidates from layers 1-2 reach here. Returns updated candidates
    with status set to confirmed/dismissed.
    """
    if not candidates or not api_key:
        return candidates

    # Batch into groups of 10 to stay within token limits
    batch_size = 10
    confirmed = []

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]

        pairs_text = []
        for idx, c in enumerate(batch):
            pairs_text.append(
                f"Pair {idx+1}:\n"
                f"  A: [{c.source_a}] \"{c.claim_a_text}\"\n"
                f"  B: [{c.source_b}] \"{c.claim_b_text}\"\n"
                f"  Detection: Layer {c.layer} — {c.reason}"
            )

        prompt = CONFIRMATION_PROMPT.format(pairs='\n\n'.join(pairs_text))

        try:
            if call_llm:
                response = call_llm(prompt, api_key, max_tokens=1024)
            else:
                from beliefs import _call_claude
                response = _call_claude(prompt, api_key, max_tokens=1024)

            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            for idx, c in enumerate(batch):
                if idx < len(lines):
                    line = lines[idx].upper()
                    if 'CONTRADICTORY' in line:
                        c.status = 'confirmed'
                        c.resolution = _extract_resolution(lines[idx])
                    elif 'TENSION' in line:
                        c.status = 'contested'
                        c.resolution = None  # tensions don't get auto-resolved
                    else:
                        c.status = 'dismissed'
                else:
                    c.status = 'unconfirmed'
                confirmed.append(c)

        except Exception as e:
            logger.warning(f'Layer 3 LLM failed: {e}')
            for c in batch:
                c.status = 'unconfirmed'
                confirmed.append(c)

    return confirmed


def _extract_resolution(line: str) -> Optional[str]:
    """Extract retain suggestion from LLM response line."""
    parts = line.split('|')
    if len(parts) >= 3:
        retain = parts[2].strip().lower()
        if 'a' in retain and 'b' not in retain:
            return 'retain_a'
        elif 'b' in retain and 'a' not in retain:
            return 'retain_b'
    return None


# ── AGM Belief Revision ──────────────────────────────────────────────────

def revise_beliefs(confirmed: list[ContradictionCandidate],
                   store: BeliefsStore,
                   adjacency: dict) -> list[RevisionLog]:
    """Apply AGM-style revision: retract the least-entrenched claim in each contradiction."""
    all_claims = [c for nb in store.notes.values() for c in nb.claims]
    claims_by_id = {c.id: c for c in all_claims}
    revisions = []

    for contradiction in confirmed:
        if contradiction.status != 'confirmed':
            continue

        claim_a = claims_by_id.get(contradiction.claim_a_id)
        claim_b = claims_by_id.get(contradiction.claim_b_id)
        if not claim_a or not claim_b:
            continue

        ent_a = compute_entrenchment(claim_a, adjacency, all_claims)
        ent_b = compute_entrenchment(claim_b, adjacency, all_claims)

        # Use LLM resolution hint if available
        if contradiction.resolution == 'retain_a':
            to_retract, to_retain = claim_b, claim_a
            ent_retracted, ent_retained = ent_b, ent_a
        elif contradiction.resolution == 'retain_b':
            to_retract, to_retain = claim_a, claim_b
            ent_retracted, ent_retained = ent_a, ent_b
        else:
            # Fall back to entrenchment ordering
            if ent_a >= ent_b:
                to_retract, to_retain = claim_b, claim_a
                ent_retracted, ent_retained = ent_b, ent_a
            else:
                to_retract, to_retain = claim_a, claim_b
                ent_retracted, ent_retained = ent_a, ent_b

        # Don't auto-retract if entrenchment is very close — mark as contested
        if abs(ent_a - ent_b) < 0.1 and contradiction.resolution is None:
            claim_a.status = 'contested'
            claim_b.status = 'contested'
            revisions.append(RevisionLog(
                timestamp=datetime.datetime.now().isoformat(),
                retracted_claim_id=claim_a.id,
                retracted_text=claim_a.text,
                retained_claim_id=claim_b.id,
                retained_text=claim_b.text,
                reason='contested — entrenchment too close to auto-resolve',
                entrenchment_retracted=round(ent_a, 3),
                entrenchment_retained=round(ent_b, 3),
            ))
        else:
            to_retract.status = 'out'
            revisions.append(RevisionLog(
                timestamp=datetime.datetime.now().isoformat(),
                retracted_claim_id=to_retract.id,
                retracted_text=to_retract.text,
                retained_claim_id=to_retain.id,
                retained_text=to_retain.text,
                reason=f'lower entrenchment ({ent_retracted:.3f} vs {ent_retained:.3f})',
                entrenchment_retracted=round(ent_retracted, 3),
                entrenchment_retained=round(ent_retained, 3),
            ))

    return revisions


# ── Full Pipeline ────────────────────────────────────────────────────────

def detect_contradictions(beliefs: BeliefsStore, adjacency: dict,
                          emb_cache: EmbeddingCache,
                          api_key: str = '', call_llm=None,
                          dry_run: bool = False,
                          note_filter: Optional[str] = None) -> dict:
    """Run the full 3-layer contradiction detection pipeline.

    Args:
        beliefs: The beliefs store with extracted claims
        adjacency: Graph adjacency dict from load_graph()
        emb_cache: EmbeddingCache for Layer 2
        api_key: For Layer 3 LLM confirmation
        call_llm: Optional callable(prompt, api_key, max_tokens) for LLM
        dry_run: If True, skip Layer 3 (no LLM cost)
        note_filter: If set, only check claims from this note

    Returns dict with layer results, confirmed contradictions, and revisions.
    """
    # Collect active claims
    all_claims = []
    for nid, nb in beliefs.notes.items():
        for claim in nb.claims:
            if claim.status != 'out':  # skip already-retracted
                all_claims.append(claim)

    if note_filter:
        # Only check this note's claims against all others
        filter_claims = [c for c in all_claims if c.source_note == note_filter]
        if not filter_claims:
            return {'error': f'No claims found for note {note_filter}'}
        # For layers, we still need all claims but focus on pairs involving the filter
        check_claims = all_claims
    else:
        check_claims = all_claims

    print(f'  Active claims: {len(all_claims)}')

    # Layer 1: Structural
    print(f'  Layer 1: Structural detection...', end=' ', flush=True)
    l1_candidates = layer1_structural(check_claims)
    if note_filter:
        l1_candidates = [c for c in l1_candidates
                         if c.source_a == note_filter or c.source_b == note_filter]
    print(f'{len(l1_candidates)} candidates')

    # Layer 2: Embedding
    print(f'  Layer 2: Embedding detection...', end=' ', flush=True)
    l2_candidates = layer2_embedding(check_claims, emb_cache)
    if note_filter:
        l2_candidates = [c for c in l2_candidates
                         if c.source_a == note_filter or c.source_b == note_filter]
    print(f'{len(l2_candidates)} candidates')

    # Deduplicate across layers
    seen = set()
    all_candidates = []
    for c in l1_candidates + l2_candidates:
        pair_key = tuple(sorted([c.claim_a_id, c.claim_b_id]))
        if pair_key not in seen:
            seen.add(pair_key)
            all_candidates.append(c)

    print(f'  Combined: {len(all_candidates)} unique candidates')

    # Layer 3: LLM confirmation
    confirmed = all_candidates
    if not dry_run and api_key and all_candidates:
        print(f'  Layer 3: LLM confirmation...', end=' ', flush=True)
        confirmed = layer3_llm(all_candidates, api_key, call_llm)
        actual_confirmed = [c for c in confirmed if c.status == 'confirmed']
        contested = [c for c in confirmed if c.status == 'contested']
        dismissed = [c for c in confirmed if c.status == 'dismissed']
        print(f'{len(actual_confirmed)} confirmed, {len(contested)} contested, {len(dismissed)} dismissed')

    # Revision
    revisions = []
    if not dry_run:
        actual_confirmed = [c for c in confirmed if c.status == 'confirmed']
        if actual_confirmed:
            print(f'  Revising beliefs...', end=' ', flush=True)
            revisions = revise_beliefs(actual_confirmed, beliefs, adjacency)
            print(f'{len(revisions)} revisions')

            # Save updated beliefs (with status changes)
            save_store(beliefs)

    # Save revision store
    rev_store = load_revision_store()
    rev_store.contradictions = confirmed
    rev_store.revisions.extend(revisions)
    rev_store.last_run = datetime.datetime.now().isoformat()
    save_revision_store(rev_store)

    return {
        'layer1': len(l1_candidates),
        'layer2': len(l2_candidates),
        'total_candidates': len(all_candidates),
        'confirmed': len([c for c in confirmed if c.status == 'confirmed']),
        'contested': len([c for c in confirmed if c.status == 'contested']),
        'dismissed': len([c for c in confirmed if c.status == 'dismissed']),
        'unconfirmed': len([c for c in confirmed if c.status == 'unconfirmed']),
        'revisions': len(revisions),
        'contradictions': [c.model_dump() for c in confirmed if c.status == 'confirmed'],
        'revision_log': [r.model_dump() for r in revisions],
    }


# ── Integration ──────────────────────────────────────────────────────────

def run_belief_revision(graph: dict, api_key: str = '', call_llm=None,
                        dry_run: bool = False) -> dict:
    """Entry point for reason.py integration.

    Loads the beliefs store, runs 3-layer detection, applies revision.
    Returns results dict for inclusion in the reasoning report.
    """
    beliefs = load_store()
    if not beliefs.notes:
        return {'skipped': True, 'reason': 'No claims extracted yet'}

    adjacency = graph.get('adjacency', {})
    emb_cache = graph.get('emb_cache')
    if not emb_cache:
        emb_cache = EmbeddingCache()

    return detect_contradictions(
        beliefs, adjacency, emb_cache,
        api_key=api_key, call_llm=call_llm,
        dry_run=dry_run,
    )


# ── CLI ──────────────────────────────────────────────────────────────────

def show_stats():
    rev_store = load_revision_store()
    if not rev_store.contradictions and not rev_store.revisions:
        print('No revision data yet.')
        return

    confirmed = [c for c in rev_store.contradictions if c.status == 'confirmed']
    dismissed = [c for c in rev_store.contradictions if c.status == 'dismissed']
    unconfirmed = [c for c in rev_store.contradictions if c.status == 'unconfirmed']

    print(f'\n== Revision Stats ==')
    print(f'  Last run: {rev_store.last_run}')
    print(f'  Contradictions: {len(confirmed)} confirmed, {len(dismissed)} dismissed, {len(unconfirmed)} unconfirmed')
    print(f'  Revisions: {len(rev_store.revisions)}')

    if confirmed:
        print(f'\n  Confirmed contradictions:')
        for i, c in enumerate(confirmed, 1):
            print(f'    {i}. [{c.source_a}] "{c.claim_a_text[:60]}..."')
            print(f'       vs [{c.source_b}] "{c.claim_b_text[:60]}..."')
            print(f'       Layer {c.layer}: {c.reason}')

    if rev_store.revisions:
        print(f'\n  Revision log:')
        for r in rev_store.revisions[-10:]:
            print(f'    RETRACTED: "{r.retracted_text[:60]}..." (ent: {r.entrenchment_retracted})')
            print(f'    RETAINED:  "{r.retained_text[:60]}..." (ent: {r.entrenchment_retained})')
            print(f'    Reason: {r.reason}')
            print()


def reverse_revisions(before: str = '', reason_prefix: str = 'reversal',
                      dry_run: bool = True) -> dict:
    """Reverse revisions made before a given timestamp.

    Restores retracted claims from 'out' back to 'active' and logs each
    reversal. This is the recovery half of AGM revision — needed when
    the detection pipeline improves and old decisions are no longer valid.

    Args:
        before: ISO timestamp — reverse revisions with timestamp < before.
                If empty, interactively lists candidates.
        reason_prefix: Prefix for the reversal reason in the log.
        dry_run: If True, report what would change without modifying data.

    Returns dict with counts and details.
    """
    rev_store = load_revision_store()
    beliefs = load_store()

    # Build claim lookup
    claims_by_id = {}
    for note_data in beliefs.notes.values():
        for claim in note_data.claims:
            claims_by_id[claim.id] = claim

    # Find revisions to reverse
    if before:
        targets = [r for r in rev_store.revisions if r.timestamp < before]
    else:
        targets = rev_store.revisions

    if not targets:
        print('No revisions match the filter.')
        return {'reversed': 0, 'skipped': 0, 'already_active': 0}

    reversed_count = 0
    skipped = 0
    already_active = 0
    details = []

    for rev in targets:
        claim = claims_by_id.get(rev.retracted_claim_id)
        if not claim:
            skipped += 1
            continue

        if claim.status == 'active':
            already_active += 1
            continue

        if claim.status != 'out' and claim.status != 'contested':
            skipped += 1
            continue

        old_status = claim.status
        details.append({
            'claim_id': claim.id,
            'text': claim.text[:80],
            'old_status': old_status,
            'new_status': 'active',
            'original_reason': rev.reason,
        })

        if not dry_run:
            claim.status = 'active'
            # Log the reversal as a new revision entry
            rev_store.revisions.append(RevisionLog(
                timestamp=datetime.datetime.now().isoformat(),
                retracted_claim_id=rev.retracted_claim_id,
                retracted_text=rev.retracted_text,
                retained_claim_id=rev.retained_claim_id,
                retained_text=rev.retained_text,
                reason=f'{reason_prefix}: restored from {old_status} '
                       f'(original: {rev.reason})',
                entrenchment_retracted=rev.entrenchment_retracted,
                entrenchment_retained=rev.entrenchment_retained,
            ))
            reversed_count += 1

    if not dry_run and reversed_count > 0:
        save_store(beliefs)
        save_revision_store(rev_store)

    return {
        'reversed': reversed_count if not dry_run else len(details),
        'skipped': skipped,
        'already_active': already_active,
        'dry_run': dry_run,
        'details': details,
    }


def get_api_key():
    cfg_path = os.path.join(_graph_path(), 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            key = json.load(f).get('anthropic_api_key', '')
            if key:
                return key
    return os.environ.get('ANTHROPIC_API_KEY', '')


def main():
    parser = argparse.ArgumentParser(description='Heartwood Belief Revision')
    parser.add_argument('--note', type=str, help='Check claims from a specific note')
    parser.add_argument('--dry-run', action='store_true', help='Layers 1-2 only, no LLM')
    parser.add_argument('--stats', action='store_true', help='Show revision stats')
    parser.add_argument('--reverse', type=str, nargs='?', const='auto',
                        help='Reverse old revisions. Pass a timestamp cutoff or "auto" '
                             'to reverse pre-fix revisions (before 2026-03-25T15:00)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply reversals (default is dry run)')
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.reverse is not None:
        if args.reverse == 'auto':
            before = '2026-03-25T15:00'
        else:
            before = args.reverse
        print(f'\n== Revision Reversal {"(DRY RUN)" if not args.apply else ""} ==')
        print(f'  Reversing revisions before: {before}\n')
        results = reverse_revisions(
            before=before,
            reason_prefix='layer3-fix-reversal',
            dry_run=not args.apply,
        )
        for d in results['details']:
            label = 'WOULD RESTORE' if not args.apply else 'RESTORED'
            print(f'  {label}: "{d["text"]}..."')
            print(f'           {d["old_status"]} -> active  (was: {d["original_reason"]})')
            print()
        print(f'  Total: {results["reversed"]} {"to reverse" if not args.apply else "reversed"}, '
              f'{results["skipped"]} skipped, {results["already_active"]} already active')
        if not args.apply:
            print(f'\n  Use --reverse --apply to execute.')
        return

    api_key = get_api_key()
    beliefs = load_store()
    if not beliefs.notes:
        print('No claims extracted. Run: python cerebro/beliefs.py --all')
        return

    from reason import load_graph
    graph = load_graph()

    print(f'\n== Belief Revision ==\n')
    results = detect_contradictions(
        beliefs,
        graph.get('adjacency', {}),
        graph['emb_cache'],
        api_key=api_key,
        dry_run=args.dry_run,
        note_filter=args.note,
    )

    print(f'\n== Results ==')
    print(f'  Layer 1 candidates: {results["layer1"]}')
    print(f'  Layer 2 candidates: {results["layer2"]}')
    print(f'  Total unique:       {results["total_candidates"]}')
    print(f'  Confirmed:          {results["confirmed"]}')
    print(f'  Dismissed:          {results["dismissed"]}')
    print(f'  Revisions:          {results["revisions"]}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
