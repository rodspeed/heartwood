"""Entrenchment tuning harness — evaluates revision accuracy against human judgments.

Reads human-annotated ground truth from eval_ground_truth.json and compares
against the entrenchment ordering. Outputs agreement score and per-case analysis.

Usage:
    python heartwood/tests/eval_entrenchment.py --generate   # generate review sheet
    python heartwood/tests/eval_entrenchment.py --evaluate   # run evaluation
    python heartwood/tests/eval_entrenchment.py --tune       # grid search weights
"""

import os
import sys
import json
import argparse
import itertools
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from revision import (
    load_revision_store, compute_entrenchment, ENTRENCHMENT_WEIGHTS,
)
from beliefs import load_store as load_beliefs_store


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, 'eval_ground_truth.json')


def generate_review_sheet():
    """Print a review sheet for manual annotation."""
    store = load_revision_store()
    confirmed = [c for c in store.contradictions if c.status == 'confirmed']

    print(f'=== ENTRENCHMENT TUNING: {len(confirmed)} contradiction pairs ===\n')
    print('For each pair, annotate:')
    print('  judgment: TRUE | TENSION | COMPLEMENTARY')
    print('  retain:   A | B | BOTH (if TENSION/COMPLEMENTARY)\n')
    print('=' * 70)

    pairs = []
    for i, c in enumerate(confirmed, 1):
        print(f'\n--- Pair {i} ---')
        print(f'Source A: [{c.source_a}]')
        print(f'Claim A:  {c.claim_a_text}')
        print(f'Source B: [{c.source_b}]')
        print(f'Claim B:  {c.claim_b_text}')
        print(f'LLM hint: {c.resolution or "none"}')
        print(f'Judgment: ___  Retain: ___')

        pairs.append({
            'index': i,
            'claim_a_id': c.claim_a_id,
            'claim_b_id': c.claim_b_id,
            'source_a': c.source_a,
            'source_b': c.source_b,
            'claim_a_text': c.claim_a_text[:200],
            'claim_b_text': c.claim_b_text[:200],
            'llm_resolution': c.resolution,
            'judgment': '',    # TRUE | TENSION | COMPLEMENTARY
            'retain': '',      # A | B | BOTH
        })

    # Write template JSON for annotation
    template_path = os.path.join(EVAL_DIR, 'eval_template.json')
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f'\n\nTemplate written to {template_path}')
    print('Fill in "judgment" and "retain" fields, then save as eval_ground_truth.json')


def evaluate(weights=None):
    """Compare entrenchment ordering against ground truth."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f'Ground truth not found at {GROUND_TRUTH_PATH}')
        print('Run --generate first, annotate, then save as eval_ground_truth.json')
        return None

    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    beliefs = load_beliefs_store()
    all_claims = [c for nb in beliefs.notes.values() for c in nb.claims]
    claims_by_id = {c.id: c for c in all_claims}

    # Build adjacency from beliefs store
    adjacency = {}
    for nid, nb in beliefs.notes.items():
        adjacency[nid] = set()  # simplified — would need graph edges

    if weights:
        # Temporarily override weights for grid search
        import revision
        original = revision.ENTRENCHMENT_WEIGHTS.copy()
        revision.ENTRENCHMENT_WEIGHTS.update(weights)

    correct = 0
    total_true = 0
    false_positives = 0  # TENSION/COMPLEMENTARY marked as confirmed
    results = []

    for pair in ground_truth:
        judgment = pair.get('judgment', '').upper()
        human_retain = pair.get('retain', '').upper()

        if not judgment:
            continue

        if judgment in ('TENSION', 'COMPLEMENTARY'):
            false_positives += 1
            results.append({
                'index': pair['index'],
                'judgment': judgment,
                'correct': False,
                'reason': f'False positive: {judgment}',
            })
            continue

        if judgment != 'TRUE':
            continue

        total_true += 1
        claim_a = claims_by_id.get(pair['claim_a_id'])
        claim_b = claims_by_id.get(pair['claim_b_id'])

        if not claim_a or not claim_b:
            results.append({
                'index': pair['index'],
                'judgment': 'TRUE',
                'correct': False,
                'reason': 'Claim not found in store',
            })
            continue

        ent_a = compute_entrenchment(claim_a, adjacency, all_claims)
        ent_b = compute_entrenchment(claim_b, adjacency, all_claims)

        # Determine system's choice
        if ent_a >= ent_b:
            system_retain = 'A'
        else:
            system_retain = 'B'

        # Check LLM resolution override
        llm_res = pair.get('llm_resolution')
        if llm_res == 'retain_a':
            system_retain = 'A'
        elif llm_res == 'retain_b':
            system_retain = 'B'

        if human_retain == 'BOTH':
            # Human says both valid — system retaining either is fine
            is_correct = True
        else:
            is_correct = (system_retain == human_retain)

        if is_correct:
            correct += 1

        results.append({
            'index': pair['index'],
            'judgment': 'TRUE',
            'human_retain': human_retain,
            'system_retain': system_retain,
            'ent_a': round(ent_a, 4),
            'ent_b': round(ent_b, 4),
            'correct': is_correct,
        })

    if weights:
        import revision
        revision.ENTRENCHMENT_WEIGHTS = original

    total_annotated = total_true + false_positives
    true_agreement = correct / total_true if total_true > 0 else 0
    fp_rate = false_positives / total_annotated if total_annotated > 0 else 0

    return {
        'total_annotated': total_annotated,
        'true_contradictions': total_true,
        'false_positives': false_positives,
        'fp_rate': round(fp_rate, 3),
        'correct': correct,
        'agreement': round(true_agreement, 3),
        'results': results,
    }


def tune():
    """Grid search over weight parameters to maximize agreement."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print('Ground truth not found. Run --generate and annotate first.')
        return

    best_agreement = 0
    best_weights = None

    # Grid search over key parameters
    for corr_boost in [0.03, 0.05, 0.07, 0.10]:
        for recency_decay in [0.0005, 0.001, 0.002, 0.005]:
            for degree_boost in [0.01, 0.02, 0.03, 0.05]:
                weights = {
                    'source_type': ENTRENCHMENT_WEIGHTS['source_type'],
                    'corroboration_boost': corr_boost,
                    'recency_decay': recency_decay,
                    'degree_boost': degree_boost,
                }
                result = evaluate(weights)
                if result and result['agreement'] > best_agreement:
                    best_agreement = result['agreement']
                    best_weights = weights.copy()

    print(f'\nBest agreement: {best_agreement:.1%}')
    if best_weights:
        print(f'Best weights:')
        print(f'  corroboration_boost: {best_weights["corroboration_boost"]}')
        print(f'  recency_decay: {best_weights["recency_decay"]}')
        print(f'  degree_boost: {best_weights["degree_boost"]}')

    return best_weights


def main():
    parser = argparse.ArgumentParser(description='Entrenchment tuning harness')
    parser.add_argument('--generate', action='store_true', help='Generate review sheet')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--tune', action='store_true', help='Grid search weights')
    args = parser.parse_args()

    if args.generate:
        generate_review_sheet()
    elif args.evaluate:
        result = evaluate()
        if result:
            print(f'\n=== EVALUATION RESULTS ===')
            print(f'Total annotated: {result["total_annotated"]}')
            print(f'True contradictions: {result["true_contradictions"]}')
            print(f'False positives: {result["false_positives"]} ({result["fp_rate"]:.0%})')
            print(f'Agreement on TRUE: {result["correct"]}/{result["true_contradictions"]} '
                  f'({result["agreement"]:.0%})')
            print(f'\nDetails:')
            for r in result['results']:
                status = 'OK' if r['correct'] else 'MISS'
                if r['judgment'] in ('TENSION', 'COMPLEMENTARY'):
                    print(f'  #{r["index"]}: FALSE POSITIVE ({r["judgment"]})')
                else:
                    print(f'  #{r["index"]}: {status} — human={r.get("human_retain","?")} '
                          f'system={r.get("system_retain","?")} '
                          f'(ent_a={r.get("ent_a","?")}, ent_b={r.get("ent_b","?")})')
    elif args.tune:
        tune()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
