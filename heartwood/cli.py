#!/usr/bin/env python3
"""Heartwood CLI — quick capture, search, and graph management from the terminal.

Usage:
    python cerebro/cli.py add "My new thought"
    python cerebro/cli.py add --title "Idea Title" "Body text goes here"
    python cerebro/cli.py search "portfolio theory"
    python cerebro/cli.py link "Note A" "Note B"
    python cerebro/cli.py recent
    python cerebro/cli.py read <note-id>

Backend: auto-detects from environment variables.
  - If SUPABASE_URL, SUPABASE_KEY, and SUPABASE_ACCESS_TOKEN are set → Supabase
  - Otherwise → local filesystem (cerebro/notes/)
"""

import sys
import os
import argparse
import time

# Resolve cerebro package imports
_cerebro_dir = os.path.dirname(os.path.abspath(__file__))
if _cerebro_dir not in sys.path:
    sys.path.insert(0, _cerebro_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(_cerebro_dir, '.env'))

# Config file location
_CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.cerebro')
_CONFIG_FILE = os.path.join(_CONFIG_DIR, 'config.json')


def _load_config():
    """Load CLI config from ~/.cerebro/config.json."""
    if os.path.exists(_CONFIG_FILE):
        import json
        with open(_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_config(cfg):
    """Save CLI config to ~/.cerebro/config.json."""
    import json
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


def _make_api():
    """Create an Api instance with the appropriate backend.

    Priority: env vars > ~/.cerebro/config.json > local filesystem.
    """
    cfg = _load_config()
    url = os.environ.get('SUPABASE_URL', '') or cfg.get('supabase_url', '')
    key = os.environ.get('SUPABASE_KEY', '') or cfg.get('supabase_key', '')
    token = os.environ.get('SUPABASE_ACCESS_TOKEN', '') or cfg.get('token', '')

    if url and key and token:
        from storage_supabase import SupabaseBackend
        import jwt as pyjwt
        payload = pyjwt.decode(token, options={"verify_signature": False})
        user_id = payload.get('sub', '')
        backend = SupabaseBackend(url, key, token, user_id)
        from app import Api
        return Api(storage=backend)

    from app import Api
    return Api()


def cmd_add(args):
    api = _make_api()
    title = args.title
    body = args.text

    if title:
        # Title provided explicitly — create note then append body
        result = api.create_note_with_title(title)
        note_id = result['id']
        if body and result.get('created', True):
            note = api.read_note(note_id)
            new_content = note['content'].rstrip() + '\n\n' + body + '\n'
            api.save_note(note_id, new_content)
        status = 'created' if result.get('created', True) else 'already exists'
        print(f"  {result['title']} ({status})")
    elif body:
        # No title — use first line or timestamp
        lines = body.strip().split('\n', 1)
        first_line = lines[0].strip()
        if len(first_line) <= 60 and '.' not in first_line:
            title = first_line
            remaining = lines[1].strip() if len(lines) > 1 else ''
        else:
            title = time.strftime('capture-%Y%m%d-%H%M%S')
            remaining = body

        result = api.create_note_with_title(title)
        note_id = result['id']
        if remaining and result.get('created', True):
            note = api.read_note(note_id)
            new_content = note['content'].rstrip() + '\n\n' + remaining + '\n'
            api.save_note(note_id, new_content)
        print(f"  {result['title']}")
    else:
        print("Nothing to add. Provide text or --title.")
        sys.exit(1)


def cmd_search(args):
    api = _make_api()
    results = api.search_notes(args.query)
    if not results:
        print(f"  No results for \"{args.query}\"")
        return
    for r in results:
        match_badge = f"[{r['match_type']}]"
        print(f"  {match_badge:10s} {r['title']}")
        if r.get('snippet'):
            print(f"             {r['snippet']}")


def cmd_link(args):
    api = _make_api()
    result = api.create_link(args.source, args.target)
    if result.get('success'):
        print(f"  Linked: {args.source} <-> {args.target}")
    else:
        print(f"  Error: {result.get('error', 'unknown')}")


def cmd_recent(args):
    api = _make_api()
    notes = api._storage.list_notes()
    notes.sort(key=lambda n: n.mtime, reverse=True)
    count = args.count or 10
    for n in notes[:count]:
        date = time.strftime('%Y-%m-%d', time.localtime(n.mtime))
        tags = ', '.join(n.tags) if n.tags else ''
        tag_str = f"  [{tags}]" if tags else ''
        print(f"  {date}  {n.title}{tag_str}")


def cmd_read(args):
    api = _make_api()
    result = api.read_note(args.note_id)
    if 'error' in result:
        print(f"  Error: {result['error']}")
        sys.exit(1)
    print(result['content'])


def cmd_config(args):
    cfg = _load_config()
    if args.token:
        cfg['token'] = args.token
        _save_config(cfg)
        print(f"  Token saved to {_CONFIG_FILE}")
    elif args.url:
        cfg['supabase_url'] = args.url
        _save_config(cfg)
        print(f"  Supabase URL saved")
    elif args.key:
        cfg['supabase_key'] = args.key
        _save_config(cfg)
        print(f"  Supabase key saved")
    elif args.show:
        if not cfg:
            print("  No config found. Run: heartwood config --token <your-token>")
        else:
            for k, v in cfg.items():
                display = v[:12] + '...' if len(v) > 15 else v
                print(f"  {k}: {display}")
    else:
        print(f"  Config location: {_CONFIG_FILE}")
        print("  Use --token, --url, --key to set values, --show to view")


def main():
    parser = argparse.ArgumentParser(
        prog='heartwood',
        description='Heartwood CLI — knowledge graph from the terminal',
    )
    sub = parser.add_subparsers(dest='command')

    # add
    p_add = sub.add_parser('add', help='Create a new note')
    p_add.add_argument('text', nargs='?', default='', help='Note text')
    p_add.add_argument('--title', '-t', default='', help='Explicit title (text becomes body)')

    # search
    p_search = sub.add_parser('search', help='Search notes')
    p_search.add_argument('query', help='Search query')

    # link
    p_link = sub.add_parser('link', help='Link two notes by title')
    p_link.add_argument('source', help='Source note title')
    p_link.add_argument('target', help='Target note title')

    # recent
    p_recent = sub.add_parser('recent', help='Show recently modified notes')
    p_recent.add_argument('-n', '--count', type=int, default=10, help='Number of notes')

    # read
    p_read = sub.add_parser('read', help='Print a note\'s content')
    p_read.add_argument('note_id', help='Note ID (filename without .md)')

    # config
    p_config = sub.add_parser('config', help='Configure CLI credentials')
    p_config.add_argument('--token', default='', help='API token from Heartwood web app')
    p_config.add_argument('--url', default='', help='Supabase URL')
    p_config.add_argument('--key', default='', help='Supabase anon key')
    p_config.add_argument('--show', action='store_true', help='Show current config')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {'add': cmd_add, 'search': cmd_search, 'link': cmd_link,
     'recent': cmd_recent, 'read': cmd_read, 'config': cmd_config}[args.command](args)


if __name__ == '__main__':
    main()
