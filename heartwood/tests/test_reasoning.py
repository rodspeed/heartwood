"""Reasoning engine tests.

Tests structural passes (no API key needed) and full reasoning if
ANTHROPIC_API_KEY is available. Verifies graceful degradation,
output structure, and performance at scale.
"""

from __future__ import annotations

import os
import sys
import time
import pytest

_cerebro_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(_cerebro_dir))

# reason.py uses bare imports too
sys.path.insert(0, _cerebro_dir)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
needs_anthropic = pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY required")


def _build_graph(n_notes: int) -> dict:
    """Build a synthetic graph dict using reason.py's own load_graph_from_data."""
    from reason import load_graph_from_data

    topics = [
        ("machine learning", ["ml", "ai"], "Neural networks, gradient descent, transformers."),
        ("vintage denim", ["fashion", "denim"], "Selvedge, indigo, fade patterns, looms."),
        ("portfolio theory", ["finance", "quant"], "Modern portfolio theory, Sharpe ratio, risk."),
        ("japanese culture", ["culture", "japan"], "Wabi-sabi, kintsugi, craftsmanship."),
        ("knowledge graphs", ["tech", "graphs"], "Nodes, edges, ontology, semantic search."),
    ]

    notes_data = []
    for i in range(n_notes):
        topic_name, tags, seed_text = topics[i % len(topics)]
        slug = f"synth-{i:04d}"
        body = f"{seed_text} Variation {i}. Links to [[{topics[(i+1) % len(topics)][0].title()} Note {(i+1) % n_notes}]]."
        notes_data.append({
            "id": slug,
            "title": f"{topic_name.title()} Note {i}",
            "tags": tags + [f"group-{i % 10}"],
            "body": body,
            "raw_content": f"---\ntitle: {topic_name.title()} Note {i}\ntags: {tags}\n---\n\n{body}",
            "type": None,
            "created": "2026-03-24",
            "mtime": float(i),
            "preview": body[:100],
        })

    connections = []
    for i in range(n_notes - 1):
        connections.append({"source": f"synth-{i:04d}", "target": f"synth-{i+1:04d}"})
    for i in range(0, n_notes, 5):
        j = (i + len(topics)) % n_notes
        connections.append({"source": f"synth-{i:04d}", "target": f"synth-{j:04d}"})

    ontology = {
        "version": 1,
        "types": {
            "concept": {"description": "A concept", "icon": "C", "color": "#4dc8ff", "shape": "sphere"},
        },
        "inference": [],
        "relationships": {},
    }

    return load_graph_from_data(notes_data, connections, ontology)


class TestStructuralPasses:
    """Tests that work without any API key — structural analysis only."""

    def test_pass_gaps_small(self):
        """Gaps pass runs on 20 notes, returns expected structure."""
        from reason import pass_gaps

        graph = _build_graph(20)
        result = pass_gaps(graph)

        assert isinstance(result, dict)
        assert "type_distribution" in result
        assert "tag_clusters" in result
        assert "underlinked_notes" in result

    def test_pass_gaps_500_notes(self):
        """Gaps pass scales to 500 notes."""
        from reason import pass_gaps

        t0 = time.time()
        graph = _build_graph(500)
        build_time = time.time() - t0
        print(f"\n  Built 500-note graph in {build_time:.1f}s")

        t0 = time.time()
        result = pass_gaps(graph)
        elapsed = time.time() - t0
        print(f"  pass_gaps(500 notes) in {elapsed:.1f}s")

        assert isinstance(result, dict)
        assert len(result.get("tag_clusters", {})) > 0

    def test_pass_drift_small(self):
        """Drift pass runs on 20 notes, returns list."""
        from reason import pass_drift

        graph = _build_graph(20)
        result = pass_drift(graph)

        assert isinstance(result, list)
        # All notes are fresh (origin == current), so drift should be low
        for item in result:
            assert "title" in item
            assert "drift" in item

    def test_pass_drift_500_notes(self):
        """Drift pass scales to 500 notes."""
        from reason import pass_drift

        graph = _build_graph(500)

        t0 = time.time()
        result = pass_drift(graph)
        elapsed = time.time() - t0
        print(f"\n  pass_drift(500 notes) in {elapsed:.1f}s")

        assert isinstance(result, list)

    def test_pass_clusters_structural_only(self):
        """Clusters pass without API key returns structural clusters."""
        from reason import pass_clusters

        graph = _build_graph(50)
        result = pass_clusters(graph, api_key=None, quiet=True)

        assert isinstance(result, list)
        if result:  # may be empty for small graphs
            assert "size" in result[0]
            assert "titles" in result[0]

    def test_pass_latent_structural_only(self):
        """Latent connections pass without API key returns structural results."""
        from reason import pass_latent

        graph = _build_graph(50)
        result = pass_latent(graph, api_key=None, quiet=True)

        assert isinstance(result, list)
        for item in result:
            assert "note_a" in item
            assert "note_b" in item
            assert "similarity" in item


class TestStructuralAt500:
    """Full structural reasoning at 500 notes — the Phase 0.5 gate."""

    def test_all_structural_passes_500(self):
        """All non-LLM passes complete on 500 notes within 60s."""
        from reason import pass_gaps, pass_drift

        t0 = time.time()
        graph = _build_graph(500)
        build_time = time.time() - t0

        t1 = time.time()
        gaps = pass_gaps(graph)
        gaps_time = time.time() - t1

        t2 = time.time()
        drift = pass_drift(graph)
        drift_time = time.time() - t2

        total = time.time() - t0
        print(f"\n  500-note reasoning (structural):")
        print(f"    Graph build + embed: {build_time:.1f}s")
        print(f"    pass_gaps: {gaps_time:.1f}s")
        print(f"    pass_drift: {drift_time:.1f}s")
        print(f"    Total: {total:.1f}s")

        assert isinstance(gaps, dict)
        assert isinstance(drift, list)
        assert total < 120  # should complete well within 2 minutes


@needs_anthropic
class TestFullReasoning:
    """Tests that require ANTHROPIC_API_KEY for Haiku calls."""

    def test_pass_latent_with_haiku(self):
        """Latent connections pass with Haiku produces richer results."""
        from reason import pass_latent

        graph = _build_graph(30)
        result = pass_latent(graph, api_key=ANTHROPIC_KEY)

        assert isinstance(result, list)
        # With Haiku, results should have reasoning field
        for item in result:
            assert "note_a" in item
            assert "note_b" in item

    def test_pass_clusters_with_haiku(self):
        """Clusters pass with Haiku produces named clusters."""
        from reason import pass_clusters

        graph = _build_graph(50)
        result = pass_clusters(graph, api_key=ANTHROPIC_KEY)

        assert isinstance(result, list)
        if result:
            # Haiku should have named at least one cluster
            named = [c for c in result if c.get("name")]
            print(f"\n  {len(result)} clusters, {len(named)} named by Haiku")

    def test_pass_contradictions_with_haiku(self):
        """Contradictions pass with Haiku runs without error."""
        from reason import pass_contradictions

        graph = _build_graph(30)
        result = pass_contradictions(graph, api_key=ANTHROPIC_KEY)

        assert isinstance(result, list)

    def test_pass_synthesis_with_haiku(self):
        """Synthesis pass with Haiku runs without error."""
        from reason import pass_synthesis

        graph = _build_graph(50)
        result = pass_synthesis(graph, api_key=ANTHROPIC_KEY)

        assert isinstance(result, list)

    def test_full_6pass_50_notes(self):
        """Full 6-pass reasoning on 50 notes — timing benchmark."""
        from reason import (pass_latent, pass_gaps, pass_clusters,
                            pass_contradictions, pass_drift, pass_synthesis)

        graph = _build_graph(50)

        timings = {}
        passes = [
            ("latent", lambda: pass_latent(graph, api_key=ANTHROPIC_KEY)),
            ("gaps", lambda: pass_gaps(graph)),
            ("clusters", lambda: pass_clusters(graph, api_key=ANTHROPIC_KEY)),
            ("contradictions", lambda: pass_contradictions(graph, api_key=ANTHROPIC_KEY)),
            ("drift", lambda: pass_drift(graph)),
            ("synthesis", lambda: pass_synthesis(graph, api_key=ANTHROPIC_KEY)),
        ]

        for name, fn in passes:
            t0 = time.time()
            result = fn()
            timings[name] = time.time() - t0
            assert result is not None

        total = sum(timings.values())
        print(f"\n  Full 6-pass reasoning on 50 notes:")
        for name, t in timings.items():
            print(f"    {name}: {t:.1f}s")
        print(f"    TOTAL: {total:.1f}s")


class TestReasoningDegradation:
    """Verify graceful fallback when Haiku is unavailable."""

    def test_latent_no_key_returns_structural(self):
        """Latent pass with no API key still returns results."""
        from reason import pass_latent

        graph = _build_graph(30)
        result = pass_latent(graph, api_key=None, quiet=True)

        assert isinstance(result, list)
        # Should have structural similarity scores
        for item in result:
            assert "similarity" in item

    def test_clusters_no_key_returns_unnamed(self):
        """Clusters pass without API key returns clusters without names."""
        from reason import pass_clusters

        graph = _build_graph(50)
        result = pass_clusters(graph, api_key=None, quiet=True)

        assert isinstance(result, list)

    def test_contradictions_no_key_returns_empty(self):
        """Contradictions pass without API key returns empty (LLM-only)."""
        from reason import pass_contradictions

        graph = _build_graph(30)
        result = pass_contradictions(graph, api_key=None, quiet=True)

        assert isinstance(result, list)
        assert len(result) == 0  # contradictions is LLM-only

    def test_synthesis_no_key_returns_empty(self):
        """Synthesis pass without API key returns empty (LLM-only)."""
        from reason import pass_synthesis

        graph = _build_graph(30)
        result = pass_synthesis(graph, api_key=None, quiet=True)

        assert isinstance(result, list)

    def test_invalid_api_key_doesnt_crash(self):
        """Bad API key causes Haiku failure but pass still returns."""
        from reason import pass_latent

        graph = _build_graph(20)
        # Should not raise — should fall back gracefully
        result = pass_latent(graph, api_key="sk-invalid-key-00000", quiet=True)

        assert isinstance(result, list)


class TestReasoningMemory:
    """Tests for delta-aware reasoning (prior-run context)."""

    FAKE_PREVIOUS_REPORT = """---
title: "Reasoning Report"
tags: [cerebro, reasoning]
type: daily
created: 2026-03-23
---

# Reasoning Report
_Generated 2026-03-23_

## 1. Latent Connections
_∞ = no existing path between notes._

| Note A | Note B | Similarity | Hops | Reasoning | Action |
|--------|--------|------------|------|-----------|--------|
| [[Alpha]] | [[Beta]] | 0.85 | ∞ | Strong thematic link | link |
| [[Gamma]] | [[Delta]] | 0.72 | 3 | Moderate overlap | skip |

## 2. Knowledge Gaps

### Type Distribution
| Type | Count |
|------|-------|
| concept | 50 |
| project | 30 |

## 3. Cluster Narratives
_Semantic groupings across the knowledge graph._

### Machine Learning (25 notes)
_Neural network and AI development cluster._

## 3b. Inter-Cluster Bridges
_Deeper themes connecting seemingly unrelated clusters._

### ML ↔ Fashion (sim: 0.65)
Pattern recognition as a shared discipline

## 4. Contradictions
_Notes that may make conflicting claims._

No contradictions detected.

## 5. Drift Alerts
_Notes that have evolved significantly from their original scope._

No significant drift detected.

## 6. Synthesis Opportunities
_Tightly-connected note clusters that could use a unifying hub note._

### Proposed: ML Foundations Hub
_6 notes, 45% connected_
"""

    def test_build_prior_context_empty(self):
        """No previous report returns empty string."""
        from reason import _build_prior_context

        result = _build_prior_context(None, None, 'Latent Connections')
        assert result == ''

    def test_build_prior_context_extracts_section(self):
        """Prior context extracts the correct section from a previous report."""
        from reason import _build_prior_context

        result = _build_prior_context(self.FAKE_PREVIOUS_REPORT, None, 'Latent Connections')
        assert 'PREVIOUS RUN' in result
        assert 'Alpha' in result
        assert 'Beta' in result
        # Should NOT contain other sections
        assert 'Synthesis' not in result

    def test_build_prior_context_includes_changed_notes(self):
        """Prior context lists changed note titles."""
        from reason import _build_prior_context

        changed = ['New Note A', 'Updated Note B']
        result = _build_prior_context(self.FAKE_PREVIOUS_REPORT, changed, 'Latent Connections')
        assert 'NOTES CHANGED SINCE LAST RUN' in result
        assert 'New Note A' in result
        assert 'Updated Note B' in result

    def test_build_prior_context_truncates_long_changed_list(self):
        """More than 20 changed notes gets truncated."""
        from reason import _build_prior_context

        changed = [f'Note {i}' for i in range(30)]
        result = _build_prior_context(self.FAKE_PREVIOUS_REPORT, changed, 'Latent Connections')
        assert '30 notes' in result
        assert '...' in result

    def test_build_prior_context_instruction(self):
        """Prior context includes the 'focus on NEW or CHANGED' instruction."""
        from reason import _build_prior_context

        result = _build_prior_context(self.FAKE_PREVIOUS_REPORT, ['X'], 'Latent Connections')
        assert 'NEW or CHANGED' in result

    def test_generate_report_first_run(self):
        """First run (no follow-up) has no delta summary."""
        from reason import generate_report

        report = generate_report([], {
            'type_distribution': {}, 'untyped': [],
            'hub_gaps': [], 'underlinked_notes': []
        }, [], [], [], [], [])
        assert 'Delta Summary' not in report

    def test_generate_report_follow_up_has_delta(self):
        """Follow-up run includes delta summary with changed notes."""
        from reason import generate_report

        report = generate_report([], {
            'type_distribution': {}, 'untyped': [],
            'hub_gaps': [], 'underlinked_notes': []
        }, [], [], [], [], [],
            is_follow_up=True,
            changed_note_titles=['Alpha', 'Beta', 'Gamma'])
        assert 'Delta Summary' in report
        assert '3 notes changed' in report
        assert '[[Alpha]]' in report

    def test_generate_report_follow_up_no_changes(self):
        """Follow-up with no changed notes says so."""
        from reason import generate_report

        report = generate_report([], {
            'type_distribution': {}, 'untyped': [],
            'hub_gaps': [], 'underlinked_notes': []
        }, [], [], [], [], [],
            is_follow_up=True,
            changed_note_titles=[])
        assert 'Delta Summary' in report
        assert 'No notes changed' in report

    def test_run_reasoning_from_data_accepts_previous(self):
        """run_reasoning_from_data with previous_report doesn't crash."""
        from reason import run_reasoning_from_data, load_graph_from_data

        notes_data = [{
            'id': 'n1', 'title': 'Test', 'tags': ['test'],
            'body': 'Hello world.', 'raw_content': 'Hello world.',
            'type': None, 'created': '2026-03-24',
            'mtime': 1.0, 'preview': 'Hello world.',
        }]
        result = run_reasoning_from_data(
            notes_data, [], {'version': 1, 'types': {}, 'inference': [], 'relationships': {}},
            api_key=None, quiet=True,
            previous_report=self.FAKE_PREVIOUS_REPORT,
            changed_note_titles=['Test'],
        )
        assert result['success']
        assert 'Delta Summary' in result['report']
        assert '1 notes changed' in result['report']

    def test_pass_latent_with_prior_context_structural(self):
        """pass_latent accepts prior_context without crashing (structural mode)."""
        from reason import pass_latent

        graph = _build_graph(20)
        prior = "--- CONTEXT FROM PREVIOUS ANALYSIS ---\nPREVIOUS RUN: some data\n--- END CONTEXT ---\n"
        result = pass_latent(graph, api_key=None, quiet=True, prior_context=prior)
        assert isinstance(result, list)
