"""Shared fixtures for Heartwood tests.

Sets up filesystem-based testing environment. Supabase/multi-tenant
tests are not included in the open-source package.
"""

from __future__ import annotations

import os
import sys
import pytest

# Allow bare imports from heartwood/
_heartwood_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(_heartwood_dir))


@pytest.fixture
def tmp_notes_dir(tmp_path):
    """Create a temporary notes directory with sample notes."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    # Create a sample note
    sample = notes_dir / "sample-note.md"
    sample.write_text(
        "---\n"
        "title: Sample Note\n"
        "tags: [test, example]\n"
        "created: 2026-01-01\n"
        "---\n\n"
        "# Sample Note\n\n"
        "This is a test note with a [[wikilink]].\n"
    )
    return notes_dir


@pytest.fixture
def tmp_graph_dir(tmp_path):
    """Create a temporary .graph directory."""
    graph_dir = tmp_path / ".graph"
    graph_dir.mkdir()
    (graph_dir / "run-counter.json").write_text("{}")
    return graph_dir
