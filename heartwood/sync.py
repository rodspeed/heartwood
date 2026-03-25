"""
Sync native markdown files <-> Heartwood notes (bidirectional).

Forward sync: native files -> Heartwood notes (create/update).
Reverse sync: when a Heartwood note is deleted, the native file is moved to
heartwood/trash/ and held for 30 days before permanent deletion.

Usage:
    python heartwood/sync.py          # dry run (shows what would change)
    python heartwood/sync.py --apply  # actually write changes

Native locations scanned:
    - .claude/skills/*/SKILL.md          -> skills
    - .claude/agents/*.md                -> agents
    - .claude/rules/*.md                 -> rules
    - Memory files (auto-memory)         -> memories
"""

import os
import sys
import re
import shutil
import yaml
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(SCRIPT_DIR, 'notes')
TRASH_DIR = os.path.join(SCRIPT_DIR, 'trash')
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
MEMORY_DIR = os.environ.get('HEARTWOOD_MEMORY_DIR', '') or os.path.join(
    os.path.expanduser('~'), '.heartwood', 'memory'
)
MEMORY_INDEX = os.path.join(MEMORY_DIR, 'MEMORY.md')
TRASH_RETENTION_DAYS = 365


def title_to_filename(title):
    """Convert a note title to a filename slug."""
    slug = re.sub(r'[^\w\s-]', '', title.lower().strip())
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug + '.md'


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


def get_existing_notes():
    """Get set of existing note filenames in Heartwood."""
    if not os.path.exists(NOTES_DIR):
        return set()
    return {f for f in os.listdir(NOTES_DIR) if f.endswith('.md')}


def discover_sources():
    """Discover all native markdown files that should be synced."""
    sources = []

    # 1. Skills (.claude/skills/*/SKILL.md)
    skills_dir = os.path.join(PARENT_DIR, '.claude', 'skills')
    if os.path.exists(skills_dir):
        for name in os.listdir(skills_dir):
            skill_file = os.path.join(skills_dir, name, 'SKILL.md')
            if os.path.isfile(skill_file):
                sources.append({
                    'path': skill_file,
                    'category': 'skill',
                    'native_name': name,
                })
            # Also check for standalone .md files in skills/
            standalone = os.path.join(skills_dir, name)
            if standalone.endswith('.md') and os.path.isfile(standalone):
                sources.append({
                    'path': standalone,
                    'category': 'skill',
                    'native_name': Path(standalone).stem,
                })

    # 2. Agents (.claude/agents/*.md)
    agents_dir = os.path.join(PARENT_DIR, '.claude', 'agents')
    if os.path.exists(agents_dir):
        for f in os.listdir(agents_dir):
            if f.endswith('.md'):
                sources.append({
                    'path': os.path.join(agents_dir, f),
                    'category': 'agent',
                    'native_name': Path(f).stem,
                })

    # 3. Memory files
    if os.path.exists(MEMORY_DIR):
        for f in os.listdir(MEMORY_DIR):
            if f.endswith('.md') and f != 'MEMORY.md':
                sources.append({
                    'path': os.path.join(MEMORY_DIR, f),
                    'category': 'memory',
                    'native_name': Path(f).stem,
                })

    # Additional source directories can be configured via HEARTWOOD_SYNC_DIRS env var
    # (comma-separated list of dir:category pairs, e.g. "/path/to/skills:skill")
    extra = os.environ.get('HEARTWOOD_SYNC_DIRS', '')
    if extra:
        for entry in extra.split(','):
            entry = entry.strip()
            if ':' not in entry:
                continue
            dir_path, cat = entry.rsplit(':', 1)
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    if f.endswith('.md'):
                        sources.append({
                            'path': os.path.join(dir_path, f),
                            'category': cat.strip(),
                            'native_name': Path(f).stem,
                        })

    return sources


def source_to_note_filename(source):
    """Determine the Heartwood note filename for a native source."""
    return title_to_filename(source['native_name'].replace('_', ' ').replace('-', ' ').title())


def check_sync_status(sources, existing_notes):
    """Compare native sources against existing Heartwood notes.
    Returns list of actions needed."""
    actions = []

    for source in sources:
        note_filename = source_to_note_filename(source)
        note_path = os.path.join(NOTES_DIR, note_filename)

        if note_filename not in existing_notes:
            actions.append({
                'action': 'create',
                'source': source,
                'note_filename': note_filename,
                'reason': 'New file not yet in Heartwood',
            })
        else:
            # Check if native file is newer
            source_mtime = os.path.getmtime(source['path'])
            note_mtime = os.path.getmtime(note_path)
            if source_mtime > note_mtime:
                actions.append({
                    'action': 'update',
                    'source': source,
                    'note_filename': note_filename,
                    'reason': f'Native file modified ({time.ctime(source_mtime)})',
                })

    return actions


def create_note_from_source(source):
    """Generate Heartwood note content from a native source file."""
    with open(source['path'], 'r', encoding='utf-8') as f:
        content = f.read()

    meta, body = parse_frontmatter(content)

    # Determine title
    title = meta.get('title') or meta.get('name') or \
            source['native_name'].replace('_', ' ').replace('-', ' ').title()

    # Determine tags based on category
    tags = meta.get('tags', [])
    if isinstance(tags, str):
        tags = [tags]

    cat = source['category']
    native_name = source['native_name'].lower()

    if cat == 'skill' and 'skills' not in tags:
        tags.append('skills')
    elif cat == 'agent' and 'agent' not in tags:
        tags.append('agent')
    elif cat == 'memory':
        # Assign tags based on naming conventions
        if 'user' in native_name or 'role' in native_name:
            if 'user' not in tags:
                tags.append('user')
        elif 'feedback' in native_name:
            if 'feedback' not in tags:
                tags.append('feedback')
        elif 'project' in native_name:
            if 'project' not in tags:
                tags.append('project')

    # Add source reference
    rel_path = os.path.relpath(source['path'], PARENT_DIR).replace('\\', '/')
    source_line = f"\n\n---\n*Native source: `{rel_path}`*\n"

    # Build the note
    today = time.strftime('%Y-%m-%d')
    note_content = f"""---
title: {title}
tags: [{', '.join(tags)}]
created: {today}
synced_from: {rel_path}
---

{body}{source_line}"""

    return note_content


def check_reverse_sync(sources, existing_notes):
    """Find native files whose Heartwood note has been deleted.
    Returns list of delete actions."""
    actions = []

    for source in sources:
        note_filename = source_to_note_filename(source)
        if note_filename not in existing_notes:
            # The note is missing -- but only flag for deletion if it was
            # previously synced (i.e., this isn't a brand-new native file).
            # We detect this by checking if a trash copy already exists or
            # if the native file has a synced_from marker, but the simplest
            # heuristic: if the note doesn't exist AND the native file does,
            # it could be new or deleted. We distinguish by checking whether
            # a CREATE action would also be generated -- if so, it's new.
            # That logic lives in run_sync; here we just flag all missing notes.
            actions.append({
                'action': 'delete',
                'source': source,
                'note_filename': note_filename,
                'reason': 'Heartwood note deleted -- removing native file',
            })

    return actions


def trash_native_file(source, apply=False):
    """Move a native file to heartwood/trash/ with a timestamp prefix."""
    src_path = source['path']
    if not os.path.isfile(src_path):
        return None

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    basename = os.path.basename(src_path)
    trash_name = f"{timestamp}_{source['category']}_{basename}"
    trash_path = os.path.join(TRASH_DIR, trash_name)

    if apply:
        os.makedirs(TRASH_DIR, exist_ok=True)
        shutil.move(src_path, trash_path)

        # If it was a skill directory with SKILL.md, remove the empty dir
        parent = os.path.dirname(src_path)
        if os.path.basename(src_path) == 'SKILL.md' and os.path.isdir(parent):
            try:
                os.rmdir(parent)  # only removes if empty
            except OSError:
                pass

    return trash_name


def remove_memory_index_entry(native_filename, apply=False):
    """Remove a memory file's entry from MEMORY.md index."""
    if not os.path.isfile(MEMORY_INDEX):
        return False

    stem = Path(native_filename).stem  # e.g. 'feedback_example'
    with open(MEMORY_INDEX, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    removed = False
    for line in lines:
        # Match lines that link to this memory file, e.g.:
        # - [Example feedback](feedback_example.md) -- ...
        if f'({stem}.md)' in line or f'({native_filename})' in line:
            removed = True
            continue
        new_lines.append(line)

    if removed and apply:
        with open(MEMORY_INDEX, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return removed


def purge_old_trash(apply=False):
    """Delete trash files older than TRASH_RETENTION_DAYS."""
    if not os.path.exists(TRASH_DIR):
        return []

    purged = []
    cutoff = time.time() - (TRASH_RETENTION_DAYS * 86400)

    for f in os.listdir(TRASH_DIR):
        fpath = os.path.join(TRASH_DIR, f)
        if os.path.getmtime(fpath) < cutoff:
            purged.append(f)
            if apply:
                os.remove(fpath)

    return purged


def check_title_sync(sources, existing_notes):
    """Find title mismatches between native files and their Heartwood notes.
    The most recently modified file wins. Only syncs memory files — skill/agent
    native names are machine identifiers, not display titles."""
    actions = []

    for source in sources:
        # Only sync titles for memory files (which use name: as a display title)
        if source['category'] not in ('memory',):
            continue

        note_filename = source_to_note_filename(source)
        note_path = os.path.join(NOTES_DIR, note_filename)

        if note_filename not in existing_notes:
            continue  # no note to compare against

        # Read native title
        with open(source['path'], 'r', encoding='utf-8') as f:
            native_content = f.read()
        native_meta, _ = parse_frontmatter(native_content)
        native_title = native_meta.get('name') or native_meta.get('title') or ''

        # Read Heartwood title
        with open(note_path, 'r', encoding='utf-8') as f:
            note_content = f.read()
        note_meta, _ = parse_frontmatter(note_content)
        note_title = note_meta.get('title') or ''

        if not native_title or not note_title:
            continue
        if native_title.strip() == note_title.strip():
            continue

        # Titles differ — most recently modified wins
        native_mtime = os.path.getmtime(source['path'])
        note_mtime = os.path.getmtime(note_path)

        if note_mtime >= native_mtime:
            # Heartwood note is newer — push title to native file
            actions.append({
                'action': 'title-to-native',
                'source': source,
                'note_filename': note_filename,
                'old_title': native_title,
                'new_title': note_title,
                'reason': f'Heartwood title "{note_title}" is newer than native "{native_title}"',
            })
        else:
            # Native file is newer — push title to Heartwood note
            actions.append({
                'action': 'title-to-heartwood',
                'source': source,
                'note_filename': note_filename,
                'old_title': note_title,
                'new_title': native_title,
                'reason': f'Native title "{native_title}" is newer than Heartwood "{note_title}"',
            })

    return actions


def apply_title_to_native(source, new_title, apply=False):
    """Update the name:/title: field in a native file's frontmatter."""
    with open(source['path'], 'r', encoding='utf-8') as f:
        content = f.read()

    meta, body = parse_frontmatter(content)
    if not meta:
        return False

    # Determine which key the native file uses
    key = 'name' if 'name' in meta else 'title'
    meta[key] = new_title

    if apply:
        # Rebuild frontmatter
        lines = ['---']
        for k, v in meta.items():
            if isinstance(v, list):
                lines.append(f'{k}: [{", ".join(str(x) for x in v)}]')
            else:
                lines.append(f'{k}: {v}')
        lines.append('---')
        new_content = '\n'.join(lines) + '\n\n' + body + '\n'
        with open(source['path'], 'w', encoding='utf-8') as f:
            f.write(new_content)

    return True


def apply_title_to_heartwood(note_filename, new_title, apply=False):
    """Update the title: field in a Heartwood note's frontmatter."""
    note_path = os.path.join(NOTES_DIR, note_filename)
    with open(note_path, 'r', encoding='utf-8') as f:
        content = f.read()

    meta, body = parse_frontmatter(content)
    if not meta:
        return False

    meta['title'] = new_title

    if apply:
        lines = ['---']
        for k, v in meta.items():
            if isinstance(v, list):
                lines.append(f'tags: [{", ".join(str(x) for x in v)}]')
            elif k == 'tags':
                lines.append(f'tags: [{", ".join(str(x) for x in v)}]')
            else:
                lines.append(f'{k}: {v}')
        lines.append('---')
        new_content = '\n'.join(lines) + '\n\n' + body + '\n'
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    return True


def run_sync(apply=False):
    """Main sync logic."""
    print("=== Heartwood Sync ===\n")

    sources = discover_sources()
    print(f"Found {len(sources)} native markdown files\n")

    existing_notes = get_existing_notes()
    print(f"Existing Heartwood notes: {len(existing_notes)}\n")

    # Title sync: detect mismatches between native and Heartwood titles
    title_actions = check_title_sync(sources, existing_notes)

    # Forward sync: native -> Heartwood
    forward_actions = check_sync_status(sources, existing_notes)

    # Reverse sync: detect native files whose Heartwood note was deleted
    reverse_actions = check_reverse_sync(sources, existing_notes)

    # Filter out conflicts: if a note is missing from Heartwood, it shows up
    # in both forward (create) and reverse (delete). Reverse wins -- the user
    # intentionally deleted the note.
    forward_note_files = {a['note_filename'] for a in forward_actions if a['action'] == 'create'}
    reverse_note_files = {a['note_filename'] for a in reverse_actions}
    conflict_files = forward_note_files & reverse_note_files

    if conflict_files:
        forward_actions = [a for a in forward_actions if a['note_filename'] not in conflict_files]

    actions = forward_actions + reverse_actions

    # Purge old trash
    purged = purge_old_trash(apply=apply)
    if purged:
        print(f"Purged {len(purged)} trash files older than {TRASH_RETENTION_DAYS} days:\n")
        for p in purged:
            print(f"  x  {p}")
        print()

    # Title sync
    if title_actions:
        print(f"{len(title_actions)} title mismatches:\n")
        for ta in title_actions:
            direction = 'Heartwood -> Native' if ta['action'] == 'title-to-native' else 'Native -> Heartwood'
            print(f"  [TITLE] {ta['note_filename']}  ({direction})")
            print(f"    \"{ta['old_title']}\" -> \"{ta['new_title']}\"")
            print()

        if apply:
            for ta in title_actions:
                if ta['action'] == 'title-to-native':
                    if apply_title_to_native(ta['source'], ta['new_title'], apply=True):
                        print(f"  Synced title to native: {os.path.basename(ta['source']['path'])}")
                elif ta['action'] == 'title-to-heartwood':
                    if apply_title_to_heartwood(ta['note_filename'], ta['new_title'], apply=True):
                        print(f"  Synced title to Heartwood: {ta['note_filename']}")
            print()

    if not actions and not title_actions:
        print("Everything is in sync!")
        return

    if actions:
        print(f"{len(actions)} file actions needed:\n")
        for a in actions:
            src = a['source']
            label = a['action'].upper()
            print(f"  [{label}] {a['note_filename']}")
            if a['action'] == 'delete':
                print(f"    Native: {src['path']}")
            else:
                print(f"    From: {src['path']}")
            print(f"    Reason: {a['reason']}")
            print()

    if not apply:
        print("Dry run -- no changes made. Use --apply to write changes.")
        return

    os.makedirs(NOTES_DIR, exist_ok=True)

    created = updated = deleted = 0
    for a in actions:
        if a['action'] in ('create', 'update'):
            note_content = create_note_from_source(a['source'])
            note_path = os.path.join(NOTES_DIR, a['note_filename'])
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)
            print(f"  Written: {a['note_filename']}")
            if a['action'] == 'create':
                created += 1
            else:
                updated += 1

        elif a['action'] == 'delete':
            trash_name = trash_native_file(a['source'], apply=True)
            if trash_name:
                print(f"  Trashed: {os.path.basename(a['source']['path'])} -> trash/{trash_name}")
                # Clean up MEMORY.md if this was a memory file
                if a['source']['category'] == 'memory':
                    basename = os.path.basename(a['source']['path'])
                    if remove_memory_index_entry(basename, apply=True):
                        print(f"  Updated: MEMORY.md (removed entry for {basename})")
                deleted += 1

    parts = []
    if created:
        parts.append(f"{created} created")
    if updated:
        parts.append(f"{updated} updated")
    if deleted:
        parts.append(f"{deleted} trashed")
    if title_actions:
        parts.append(f"{len(title_actions)} titles synced")
    print(f"\nSync complete! {', '.join(parts)}.")


CATEGORY_RESTORE_DIRS = {
    'skill': os.path.join(PARENT_DIR, '.claude', 'skills'),
    'agent': os.path.join(PARENT_DIR, '.claude', 'agents'),
    'memory': MEMORY_DIR,
    # Additional restore dirs can be added via HEARTWOOD_SYNC_DIRS
}


def parse_trash_name(filename):
    """Parse a trash filename into (timestamp, category, original_name)."""
    # Format: 20260318-210841_agent_tell-me-the-time.md
    parts = filename.split('_', 2)
    if len(parts) < 3:
        return None, None, filename
    return parts[0], parts[1], parts[2]


def list_trash():
    """List all files in the trash directory."""
    if not os.path.exists(TRASH_DIR):
        print("Trash is empty.")
        return

    files = sorted(os.listdir(TRASH_DIR))
    if not files:
        print("Trash is empty.")
        return

    print(f"=== Heartwood Trash ({len(files)} files) ===\n")
    for f in files:
        ts, category, original = parse_trash_name(f)
        fpath = os.path.join(TRASH_DIR, f)
        age_days = int((time.time() - os.path.getmtime(fpath)) / 86400)
        expires_in = TRASH_RETENTION_DAYS - age_days
        print(f"  {original}")
        print(f"    Category: {category}  |  Trashed: {ts}  |  Expires in: {expires_in} days")
        print()


def restore_from_trash(search_term):
    """Restore a file from trash back to its native location."""
    if not os.path.exists(TRASH_DIR):
        print("Trash is empty.")
        return

    # Find matching files
    matches = []
    for f in os.listdir(TRASH_DIR):
        ts, category, original = parse_trash_name(f)
        if search_term.lower() in original.lower() or search_term.lower() in f.lower():
            matches.append((f, ts, category, original))

    if not matches:
        print(f"No trash files matching '{search_term}'.")
        return

    if len(matches) > 1:
        print(f"Multiple matches for '{search_term}':\n")
        for f, ts, category, original in matches:
            print(f"  {f}")
        print(f"\nBe more specific, or use the full trash filename.")
        return

    trash_file, ts, category, original = matches[0]
    trash_path = os.path.join(TRASH_DIR, trash_file)

    # Determine restore destination
    restore_dir = CATEGORY_RESTORE_DIRS.get(category)
    if not restore_dir:
        print(f"Unknown category '{category}' -- cannot determine restore path.")
        return

    # Skills with SKILL.md go into a subdirectory
    if category == 'skill' and original == 'SKILL.md':
        # Can't determine original skill dir name from trash alone
        print(f"Skill SKILL.md files need a directory name.")
        print(f"Manually move: {trash_path}")
        return

    restore_path = os.path.join(restore_dir, original)

    if os.path.exists(restore_path):
        print(f"Destination already exists: {restore_path}")
        return

    os.makedirs(restore_dir, exist_ok=True)
    shutil.move(trash_path, restore_path)
    print(f"Restored: {original} -> {restore_path}")

    # Also re-create the Heartwood note via forward sync
    print(f"\nRun 'python heartwood/sync.py --apply' to re-create the Heartwood note.")


def print_help():
    print("""Heartwood Sync -- bidirectional sync between native files and Heartwood notes.

Commands:
  python heartwood/sync.py              Dry run (show what would change)
  python heartwood/sync.py --apply      Apply changes (create/update/trash)
  python heartwood/sync.py --trash      List files in the trash
  python heartwood/sync.py --restore X  Restore a trashed file (fuzzy match on name)
  python heartwood/sync.py --help       Show this help
""")


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
    elif '--trash' in sys.argv:
        list_trash()
    elif '--restore' in sys.argv:
        idx = sys.argv.index('--restore')
        if idx + 1 < len(sys.argv):
            restore_from_trash(sys.argv[idx + 1])
        else:
            print("Usage: python heartwood/sync.py --restore <search_term>")
    else:
        apply = '--apply' in sys.argv
        run_sync(apply=apply)
