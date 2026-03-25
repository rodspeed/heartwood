"""Pydantic models for Heartwood API requests/responses and storage abstraction."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Data objects (used by Api + both backends) ──────────────────────────


@dataclass
class NoteData:
    """In-memory representation of a note, decoupled from storage format."""
    id: str
    title: str
    tags: list[str] = field(default_factory=list)
    body: str = ""
    raw_content: str = ""          # full markdown (frontmatter + body)
    type: str | None = None
    created: str = ""
    mtime: float = 0.0             # last-modified epoch (filesystem) or updated_at
    synced_from: str = ""
    preview: str = ""
    file: str = ""                 # filename (filesystem-only, ignored by Supabase)


@dataclass
class EdgeData:
    source: str
    target: str
    type: str = "wikilink"         # wikilink | manual | semantic
    rel: str | None = None         # typed relationship prefix
    score: float | None = None     # semantic similarity score


@dataclass
class CanvasData:
    name: str
    data: dict[str, Any] = field(default_factory=dict)


# ── Storage backend contract ────────────────────────────────────────────


class StorageBackend(abc.ABC):
    """Abstract interface for note storage.

    FileSystemBackend implements this interface.
    Every method that writes returns a result dict for API compatibility.
    """

    # ── Notes ───────────────────────────────────────────────────────

    @abc.abstractmethod
    def list_notes(self) -> list[NoteData]:
        """Return all notes (metadata + body) for this user."""

    @abc.abstractmethod
    def read_note(self, note_id: str) -> NoteData | None:
        """Read a single note by slug/id. Returns None if not found."""

    @abc.abstractmethod
    def save_note(self, note_id: str, raw_content: str) -> dict:
        """Write full markdown content for a note. Returns {success: True}."""

    @abc.abstractmethod
    def create_note(self, note_id: str, filename: str, raw_content: str) -> dict:
        """Create a new note. Returns {success, id, file} or {error}."""

    @abc.abstractmethod
    def delete_note(self, note_id: str) -> dict:
        """Delete a note. Returns {success} or {error}."""

    @abc.abstractmethod
    def rename_note(self, old_id: str, new_id: str, new_filename: str,
                    new_content: str) -> dict:
        """Rename (re-slug) a note. Deletes old if slug changed."""

    @abc.abstractmethod
    def note_exists(self, note_id: str) -> bool:
        """Check if a note with this id/slug exists."""

    # ── Connections (manual edges) ──────────────────────────────────

    @abc.abstractmethod
    def load_connections(self) -> list[dict]:
        """Return all manual connections [{source, target}, ...]."""

    @abc.abstractmethod
    def save_connections(self, connections: list[dict]) -> dict:
        """Overwrite all manual connections."""

    # ── Layout ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def load_layout(self) -> dict:
        """Return saved node positions {node_id: {x,y,z}}."""

    @abc.abstractmethod
    def save_layout(self, positions: dict) -> dict:
        """Save node positions."""

    # ── Constellations ──────────────────────────────────────────────

    @abc.abstractmethod
    def load_constellations(self) -> dict:
        """Return constellation name mapping."""

    @abc.abstractmethod
    def save_constellations(self, data: dict) -> dict:
        """Save constellation names."""

    # ── Canvases ────────────────────────────────────────────────────

    @abc.abstractmethod
    def list_canvases(self) -> list[str]:
        """Return all canvas names."""

    @abc.abstractmethod
    def load_canvas(self, name: str) -> dict:
        """Load a named canvas. Returns canvas data or {error}."""

    @abc.abstractmethod
    def save_canvas(self, name: str, data: dict) -> dict:
        """Save a named canvas."""

    @abc.abstractmethod
    def delete_canvas(self, name: str) -> dict:
        """Delete a named canvas."""

    # ── Config ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_config(self, key: str) -> str:
        """Get a config value. Returns empty string if not set."""

    @abc.abstractmethod
    def set_config(self, key: str, value: str) -> dict:
        """Set a config value."""

    @abc.abstractmethod
    def get_all_config(self) -> dict:
        """Return the full config dict."""

    # ── Ontology ────────────────────────────────────────────────────

    @abc.abstractmethod
    def load_ontology(self) -> dict:
        """Return the ontology definition."""

    @abc.abstractmethod
    def save_ontology(self, ontology: dict) -> dict:
        """Save the ontology definition."""

    # ── Embeddings cache ────────────────────────────────────────────

    @abc.abstractmethod
    def load_embeddings_cache(self) -> dict:
        """Return raw embeddings cache {note_id: {mtime, vector, chunks, origin_vector}}."""

    @abc.abstractmethod
    def save_embeddings_cache(self, cache: dict) -> dict:
        """Persist the embeddings cache."""
