"""FileSystem storage backend — preserves exact existing behavior for desktop app."""

from __future__ import annotations

import json
import os
import yaml
from pathlib import Path

from models import StorageBackend, NoteData


def _parse_frontmatter(content: str) -> tuple[dict, str]:
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


def _filename_to_id(filename: str) -> str:
    return Path(filename).stem


class FileSystemBackend(StorageBackend):
    """Reads/writes notes and graph state from the local filesystem.

    This is the original storage model — notes are .md files in notes_dir,
    graph state lives in .graph/ as JSON files.
    """

    def __init__(self, notes_path: str, graph_path: str, ontology_path: str | None = None):
        self._notes_path = notes_path
        self._graph_path = graph_path
        self._ontology_path = ontology_path or os.path.join(
            os.path.dirname(notes_path), 'ontology.yaml'
        )
        os.makedirs(self._notes_path, exist_ok=True)
        os.makedirs(self._graph_path, exist_ok=True)

    # ── Notes ───────────────────────────────────────────────────────

    def list_notes(self) -> list[NoteData]:
        notes = []
        for f in os.listdir(self._notes_path):
            if not f.endswith('.md'):
                continue
            try:
                filepath = os.path.join(self._notes_path, f)
                with open(filepath, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                meta, body = _parse_frontmatter(content)
                node_id = _filename_to_id(f)
                title = str(meta.get('title', node_id.replace('-', ' ').title()))
                tags = meta.get('tags', [])
                created = str(meta.get('created', ''))
                mtime = os.path.getmtime(filepath)

                from app import make_preview
                preview = make_preview(body)

                notes.append(NoteData(
                    id=node_id, title=title, tags=tags, body=body,
                    raw_content=content, created=created, mtime=mtime,
                    type=meta.get('type'),
                    synced_from=meta.get('synced_from', ''),
                    preview=preview, file=f,
                ))
            except Exception as e:
                print(f'Warning: skipping {f}: {e}')
        return notes

    def read_note(self, note_id: str) -> NoteData | None:
        filepath = os.path.join(self._notes_path, note_id + '.md')
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()
        meta, body = _parse_frontmatter(content)
        return NoteData(
            id=note_id,
            title=meta.get('title', note_id.replace('-', ' ').title()),
            tags=meta.get('tags', []),
            body=body,
            raw_content=content,
            type=meta.get('type'),
            created=str(meta.get('created', '')),
            mtime=os.path.getmtime(filepath),
            synced_from=meta.get('synced_from', ''),
            file=note_id + '.md',
        )

    def save_note(self, note_id: str, raw_content: str) -> dict:
        filepath = os.path.join(self._notes_path, note_id + '.md')
        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(raw_content)
        return {'success': True}

    def create_note(self, note_id: str, filename: str, raw_content: str) -> dict:
        filepath = os.path.join(self._notes_path, filename)
        if os.path.exists(filepath):
            return {'error': 'Note already exists', 'id': note_id}
        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(raw_content)
        return {'success': True, 'id': note_id, 'file': filename}

    def delete_note(self, note_id: str) -> dict:
        filepath = os.path.join(self._notes_path, note_id + '.md')
        if os.path.exists(filepath):
            os.remove(filepath)
            return {'success': True}
        return {'error': 'Note not found'}

    def rename_note(self, old_id: str, new_id: str, new_filename: str,
                    new_content: str) -> dict:
        old_path = os.path.join(self._notes_path, old_id + '.md')
        if not os.path.exists(old_path):
            return {'error': 'Note not found'}
        new_path = os.path.join(self._notes_path, new_filename)
        with open(new_path, 'w', encoding='utf-8') as fh:
            fh.write(new_content)
        if old_path != new_path:
            os.remove(old_path)
        return {'success': True, 'id': new_id, 'file': new_filename}

    def note_exists(self, note_id: str) -> bool:
        return os.path.exists(os.path.join(self._notes_path, note_id + '.md'))

    # ── Connections ─────────────────────────────────────────────────

    def load_connections(self) -> list[dict]:
        path = os.path.join(self._graph_path, 'connections.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return []

    def save_connections(self, connections: list[dict]) -> dict:
        path = os.path.join(self._graph_path, 'connections.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(connections, fh, indent=2)
        return {'success': True}

    # ── Layout ──────────────────────────────────────────────────────

    def load_layout(self) -> dict:
        path = os.path.join(self._graph_path, 'layout.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return {}

    def save_layout(self, positions: dict) -> dict:
        path = os.path.join(self._graph_path, 'layout.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(positions, fh, indent=2)
        return {'success': True}

    # ── Constellations ──────────────────────────────────────────────

    def load_constellations(self) -> dict:
        path = os.path.join(self._graph_path, 'constellations.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return {}

    def save_constellations(self, data: dict) -> dict:
        path = os.path.join(self._graph_path, 'constellations.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        return {'success': True}

    # ── Canvases ────────────────────────────────────────────────────

    def _canvases_path(self) -> str:
        return os.path.join(self._graph_path, 'canvases.json')

    def _load_all_canvases(self) -> dict:
        path = self._canvases_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return {}

    def _save_all_canvases(self, canvases: dict):
        with open(self._canvases_path(), 'w', encoding='utf-8') as fh:
            json.dump(canvases, fh, indent=2)

    def list_canvases(self) -> list[str]:
        return list(self._load_all_canvases().keys())

    def load_canvas(self, name: str) -> dict:
        canvases = self._load_all_canvases()
        return canvases.get(name, {'error': 'Canvas not found'})

    def save_canvas(self, name: str, data: dict) -> dict:
        canvases = self._load_all_canvases()
        canvases[name] = data
        self._save_all_canvases(canvases)
        return {'success': True}

    def delete_canvas(self, name: str) -> dict:
        canvases = self._load_all_canvases()
        if name in canvases:
            del canvases[name]
            self._save_all_canvases(canvases)
            return {'success': True}
        return {'error': 'Canvas not found'}

    # ── Config ──────────────────────────────────────────────────────

    def _config_path(self) -> str:
        return os.path.join(self._graph_path, 'config.json')

    def _load_config(self) -> dict:
        path = self._config_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _save_config(self, cfg: dict):
        with open(self._config_path(), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, indent=2)

    def get_config(self, key: str) -> str:
        return self._load_config().get(key, '')

    def set_config(self, key: str, value: str) -> dict:
        cfg = self._load_config()
        cfg[key] = value
        self._save_config(cfg)
        return {'success': True}

    def get_all_config(self) -> dict:
        return self._load_config()

    # ── Ontology ────────────────────────────────────────────────────

    def load_ontology(self) -> dict:
        if os.path.exists(self._ontology_path):
            with open(self._ontology_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_ontology(self, ontology: dict) -> dict:
        with open(self._ontology_path, 'w', encoding='utf-8') as f:
            yaml.dump(ontology, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        return {'success': True}

    # ── Embeddings cache ────────────────────────────────────────────

    def load_embeddings_cache(self) -> dict:
        path = os.path.join(self._graph_path, 'embeddings.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def save_embeddings_cache(self, cache: dict) -> dict:
        path = os.path.join(self._graph_path, 'embeddings.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        return {'success': True}
