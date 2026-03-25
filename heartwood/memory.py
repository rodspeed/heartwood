"""Heartwood Memory System — Phase 3 of Reasoning Engine v2.

Four output channels that create compounding across reasoning runs:
1. Reflection nodes — first-class graph notes with importance scores
2. Living summary — Pydantic-enforced persistent doc (MemGPT pattern)
3. Extracted rules — structured heuristics with confidence weights
4. Reasoning strategies — named analytical approaches with usage tracking

Pre-run: loads memory context for injection into Haiku prompts.
Post-run: writes memory artifacts (each step independent, partial failure OK).
"""

import os
import json
import math
import logging
import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger('cerebro.memory')


# ── Pydantic Models ──────────────────────────────────────────────────────

class ThematicMap(BaseModel):
    macro_themes: list[str] = Field(default_factory=list, max_length=6)
    emerging_themes: list[str] = Field(default_factory=list, max_length=4)
    declining_themes: list[str] = Field(default_factory=list, max_length=4)


class GraphEvolution(BaseModel):
    growth_rate: str = ""
    structural_changes: list[str] = Field(default_factory=list, max_length=5)
    stability_note: str = ""


class LivingSummary(BaseModel):
    version: int = 1
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    thematic_map: ThematicMap = Field(default_factory=ThematicMap)
    active_hypotheses: list[str] = Field(default_factory=list, max_length=8)
    known_blind_spots: list[str] = Field(default_factory=list, max_length=8)
    graph_evolution: GraphEvolution = Field(default_factory=GraphEvolution)
    meta_observations: list[str] = Field(default_factory=list, max_length=8)


class ReasoningRule(BaseModel):
    id: str
    rule: str
    confidence: float = Field(ge=0.0, le=1.0)
    discovered_at_run: int
    last_validated_run: int
    times_applied: int = 0
    times_useful: int = 0


class RulesStore(BaseModel):
    rules: list[ReasoningRule] = Field(default_factory=list)


class ReasoningStrategy(BaseModel):
    id: str
    name: str
    description: str
    applicable_when: str = ""
    discovered_at_run: int
    times_applied: int = 0
    times_useful: int = 0


class StrategiesStore(BaseModel):
    strategies: list[ReasoningStrategy] = Field(default_factory=list)


# ── File I/O ─────────────────────────────────────────────────────────────

def _graph_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.graph')


def _notes_path():
    from app import notes_dir
    return notes_dir()


def _load_json(filename, default=None):
    path = os.path.join(_graph_path(), filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default if default is not None else {}


def _save_json(filename, data):
    path = os.path.join(_graph_path(), filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def load_living_summary() -> LivingSummary:
    data = _load_json('living-summary.json')
    if data:
        try:
            return LivingSummary.model_validate(data)
        except Exception as e:
            logger.warning(f'Failed to parse living summary, using default: {e}')
    return LivingSummary()


def save_living_summary(summary: LivingSummary):
    _save_json('living-summary.json', summary.model_dump(mode='json'))


def load_rules() -> RulesStore:
    data = _load_json('reasoning-rules.json', {'rules': []})
    try:
        return RulesStore.model_validate(data)
    except Exception as e:
        logger.warning(f'Failed to parse rules store, using default: {e}')
        return RulesStore()


def save_rules(store: RulesStore):
    _save_json('reasoning-rules.json', store.model_dump(mode='json'))


def load_strategies() -> StrategiesStore:
    data = _load_json('reasoning-strategies.json', {'strategies': []})
    try:
        return StrategiesStore.model_validate(data)
    except Exception as e:
        logger.warning(f'Failed to parse strategies store, using default: {e}')
        return StrategiesStore()


def save_strategies(store: StrategiesStore):
    _save_json('reasoning-strategies.json', store.model_dump(mode='json'))


def _get_run_number() -> int:
    """Get the next run number from a simple counter file."""
    path = os.path.join(_graph_path(), 'run-counter.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            n = data.get('run_number', 0) + 1
    else:
        n = 1
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'run_number': n}, f)
    return n


# ── Reflection Nodes ─────────────────────────────────────────────────────

def _load_recent_reflections(limit=3) -> list[dict]:
    """Load the most recent reflection notes from the notes directory."""
    npath = _notes_path()
    reflections = []
    for f in os.listdir(npath):
        if not f.startswith('reasoning-reflection-') or not f.endswith('.md'):
            continue
        filepath = os.path.join(npath, f)
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()
        # Quick parse for importance and created date
        importance = 5
        created = ''
        for line in content.split('\n')[:15]:
            if line.startswith('importance:'):
                try:
                    importance = int(line.split(':')[1].strip())
                except ValueError:
                    pass
            if line.startswith('created:'):
                created = line.split(':', 1)[1].strip()
        reflections.append({
            'file': f,
            'content': content,
            'importance': importance,
            'created': created,
        })
    # Sort by created date descending, take most recent
    reflections.sort(key=lambda r: r['created'], reverse=True)
    return reflections[:limit]


def generate_reflection(report: str, run_number: int, api_key: str,
                        call_haiku) -> Optional[str]:
    """Generate a reflection node from the reasoning report.

    Returns the filepath of the written note, or None on failure.
    """
    today = datetime.date.today().isoformat()

    # Load last reflection for delta context
    recent = _load_recent_reflections(limit=1)
    prior_context = ''
    if recent:
        prior_text = recent[0]['content']
        # Trim to keep prompt small
        if len(prior_text) > 1500:
            prior_text = prior_text[:1500] + '...'
        prior_context = f'\n\nPREVIOUS REFLECTION:\n{prior_text}'

    prompt = f"""Analyze this reasoning report and produce a reflection. You are a meta-cognitive layer for a personal knowledge graph.

REASONING REPORT:
{report[:6000]}
{prior_context}

Produce a reflection with these sections:
## Key Findings
- 3-5 bullet points on the most important discoveries

## What Changed Since Last Run
- 2-4 bullets on structural or thematic changes (if no prior reflection, note this is the first run)

## Blind Spots
- 2-4 areas the graph is missing or underrepresenting

Return ONLY the markdown content (no frontmatter). Be specific — reference note titles with [[wikilinks]]."""

    try:
        text = call_haiku(prompt, api_key, max_tokens=1024)
    except Exception as e:
        logger.error(f'Reflection generation failed: {e}')
        return None

    # Ask for importance score
    importance_prompt = f"""Rate the importance of these reasoning findings on a scale of 1-10.
1 = routine, no surprises. 10 = paradigm-shifting insight.

{text[:2000]}

Return ONLY a single integer."""

    importance = 5
    try:
        imp_text = call_haiku(importance_prompt, api_key, max_tokens=16)
        # Extract first integer found
        for word in imp_text.strip().split():
            word_clean = word.strip('.,;:!?')
            if word_clean.isdigit():
                importance = max(1, min(10, int(word_clean)))
                break
    except Exception:
        pass  # Default importance 5 is fine

    # Write the note
    frontmatter = f"""---
title: "Reasoning Reflection — {today}"
tags: [reflection, reasoning]
type: reflection
created: {today}
source: reasoning-engine
run_number: {run_number}
importance: {importance}
---"""

    full_content = f"{frontmatter}\n\n# Reasoning Reflection — {today}\n\n{text}\n"

    npath = _notes_path()
    filename = f'reasoning-reflection-{today}.md'
    filepath = os.path.join(npath, filename)

    # If a reflection already exists for today, append run number
    if os.path.exists(filepath):
        filename = f'reasoning-reflection-{today}-run{run_number}.md'
        filepath = os.path.join(npath, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)

    logger.info(f'Reflection written: {filename} (importance={importance})')
    return filepath


# ── Living Summary Update ────────────────────────────────────────────────

def update_living_summary(report: str, api_key: str, call_haiku) -> bool:
    """Update the living summary via structured merge. Returns True on success."""
    current = load_living_summary()

    prompt = f"""Given the current living summary and new reasoning findings, propose updates.
Return ONLY a JSON object with the fields that should change.
Do not include unchanged fields. Use the exact field names from the schema.

Schema fields:
- thematic_map: {{macro_themes: list[str] (max 6), emerging_themes: list[str] (max 4), declining_themes: list[str] (max 4)}}
- active_hypotheses: list[str] (max 8)
- known_blind_spots: list[str] (max 8)
- graph_evolution: {{growth_rate: str, structural_changes: list[str] (max 5), stability_note: str}}
- meta_observations: list[str] (max 8)

Current summary:
{current.model_dump_json(indent=2)}

New findings:
{report[:4000]}

Return ONLY valid JSON. No markdown, no explanation."""

    try:
        diff_text = call_haiku(prompt, api_key, max_tokens=1500)
    except Exception as e:
        logger.error(f'Living summary update failed (API call): {e}')
        return False

    # Extract JSON from response (handle markdown code blocks)
    diff_text = diff_text.strip()
    if diff_text.startswith('```'):
        lines = diff_text.split('\n')
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith('```')]
        diff_text = '\n'.join(lines)

    try:
        parsed_diff = json.loads(diff_text)
    except json.JSONDecodeError as e:
        logger.error(f'Living summary update failed (invalid JSON): {e}')
        return False

    # Deep merge: update nested models properly
    try:
        current_data = current.model_dump()

        for key, value in parsed_diff.items():
            if key in current_data:
                if isinstance(current_data[key], dict) and isinstance(value, dict):
                    current_data[key].update(value)
                else:
                    current_data[key] = value

        # Bump version and timestamp
        current_data['version'] = current.version + 1
        current_data['updated_at'] = datetime.datetime.now().isoformat()

        # Validate through Pydantic — enforces types and max lengths
        updated = LivingSummary.model_validate(current_data)
        save_living_summary(updated)
        logger.info(f'Living summary updated to v{updated.version}')
        return True

    except Exception as e:
        logger.error(f'Living summary update failed (validation): {e}')
        return False


# ── Rule Extraction ──────────────────────────────────────────────────────

def extract_rules(report: str, run_number: int, api_key: str,
                  call_haiku) -> bool:
    """Extract new rules and validate existing ones. Returns True on success."""
    store = load_rules()

    existing_rules_text = ''
    if store.rules:
        existing_rules_text = '\n'.join(
            f'- [{r.id}] (confidence={r.confidence:.1f}): {r.rule}'
            for r in store.rules
        )

    prompt = f"""Analyze this reasoning report and extract structural heuristics about the knowledge graph.

REPORT:
{report[:4000]}

EXISTING RULES:
{existing_rules_text or '(none yet)'}

For each finding, determine:
1. NEW RULES: Patterns that could be reusable heuristics (e.g., "Notes tagged X within Y days of each other represent Z")
2. VALIDATED RULES: Which existing rules were confirmed by this run's findings?
3. WEAKENED RULES: Which existing rules were contradicted?

Return a JSON object:
{{
  "new_rules": [
    {{"rule": "description of the heuristic", "confidence": 0.6}}
  ],
  "validated": ["rule-001", "rule-003"],
  "weakened": ["rule-002"]
}}

Return ONLY valid JSON. If no updates, return {{"new_rules": [], "validated": [], "weakened": []}}"""

    try:
        result_text = call_haiku(prompt, api_key, max_tokens=1024)
    except Exception as e:
        logger.error(f'Rule extraction failed (API call): {e}')
        return False

    # Extract JSON
    result_text = result_text.strip()
    if result_text.startswith('```'):
        lines = result_text.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        result_text = '\n'.join(lines)

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError as e:
        logger.error(f'Rule extraction failed (invalid JSON): {e}')
        return False

    try:
        # Add new rules
        next_id = len(store.rules) + 1
        for new_rule in result.get('new_rules', []):
            if not new_rule.get('rule'):
                continue
            store.rules.append(ReasoningRule(
                id=f'rule-{next_id:03d}',
                rule=new_rule['rule'],
                confidence=min(1.0, max(0.0, float(new_rule.get('confidence', 0.5)))),
                discovered_at_run=run_number,
                last_validated_run=run_number,
            ))
            next_id += 1

        # Update validated rules
        for rule_id in result.get('validated', []):
            for rule in store.rules:
                if rule.id == rule_id:
                    rule.last_validated_run = run_number
                    rule.times_applied += 1
                    rule.times_useful += 1
                    # Boost confidence slightly
                    rule.confidence = min(1.0, rule.confidence + 0.05)

        # Update weakened rules
        for rule_id in result.get('weakened', []):
            for rule in store.rules:
                if rule.id == rule_id:
                    rule.times_applied += 1
                    # Reduce confidence
                    rule.confidence = max(0.0, rule.confidence - 0.1)

        save_rules(store)
        logger.info(f'Rules updated: {len(result.get("new_rules", []))} new, '
                    f'{len(result.get("validated", []))} validated, '
                    f'{len(result.get("weakened", []))} weakened')
        return True

    except Exception as e:
        logger.error(f'Rule extraction failed (processing): {e}')
        return False


# ── Strategy Evaluation ──────────────────────────────────────────────────

def evaluate_strategies(report: str, run_number: int, api_key: str,
                        call_haiku) -> bool:
    """Evaluate and discover strategies. Returns True on success."""
    store = load_strategies()

    existing_strategies_text = ''
    if store.strategies:
        existing_strategies_text = '\n'.join(
            f'- [{s.id}] {s.name}: {s.description} (applied={s.times_applied}, useful={s.times_useful})'
            for s in store.strategies
        )

    prompt = f"""Analyze this reasoning report and identify analytical strategies that were used or could be used.

REPORT:
{report[:4000]}

EXISTING STRATEGIES:
{existing_strategies_text or '(none yet)'}

Determine:
1. NEW STRATEGIES: Named analytical approaches that proved productive in this run
2. APPLIED STRATEGIES: Which existing strategies were used? Were they useful (yes/no)?

Return a JSON object:
{{
  "new_strategies": [
    {{
      "name": "Strategy Name",
      "description": "What it does",
      "applicable_when": "When to use it"
    }}
  ],
  "applied": [
    {{"id": "strategy-001", "useful": true}}
  ]
}}

Return ONLY valid JSON. If no updates, return {{"new_strategies": [], "applied": []}}"""

    try:
        result_text = call_haiku(prompt, api_key, max_tokens=1024)
    except Exception as e:
        logger.error(f'Strategy evaluation failed (API call): {e}')
        return False

    result_text = result_text.strip()
    if result_text.startswith('```'):
        lines = result_text.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        result_text = '\n'.join(lines)

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError as e:
        logger.error(f'Strategy evaluation failed (invalid JSON): {e}')
        return False

    try:
        # Add new strategies
        next_id = len(store.strategies) + 1
        for new_strat in result.get('new_strategies', []):
            if not new_strat.get('name'):
                continue
            store.strategies.append(ReasoningStrategy(
                id=f'strategy-{next_id:03d}',
                name=new_strat['name'],
                description=new_strat.get('description', ''),
                applicable_when=new_strat.get('applicable_when', ''),
                discovered_at_run=run_number,
            ))
            next_id += 1

        # Update applied strategies
        for applied in result.get('applied', []):
            strat_id = applied.get('id', '')
            for strat in store.strategies:
                if strat.id == strat_id:
                    strat.times_applied += 1
                    if applied.get('useful', False):
                        strat.times_useful += 1

        save_strategies(store)
        logger.info(f'Strategies updated: {len(result.get("new_strategies", []))} new, '
                    f'{len(result.get("applied", []))} evaluated')
        return True

    except Exception as e:
        logger.error(f'Strategy evaluation failed (processing): {e}')
        return False


# ── Pre-Run Memory Loading ───────────────────────────────────────────────

def build_memory_context(previous_report: Optional[str] = None,
                         changed_note_titles: Optional[list[str]] = None,
                         section_name: Optional[str] = None) -> str:
    """Build context from memory system for injection into Haiku prompts.

    Replaces _build_prior_context with memory-aware version.
    Loads: living summary + last 3 reflections + active rules + strategies.
    Then appends prior report section + changed notes (original behavior).
    """
    parts = []

    # 1. Living summary — always loaded
    summary = load_living_summary()
    if summary.version > 1 or summary.thematic_map.macro_themes:
        summary_text = _format_living_summary(summary)
        parts.append(f'KNOWLEDGE GRAPH MEMORY:\n{summary_text}')

    # 2. Last 3 reflections (importance-weighted)
    reflections = _load_recent_reflections(limit=3)
    if reflections:
        ref_parts = []
        for r in reflections:
            # Extract body only (skip frontmatter)
            body = r['content']
            if '---' in body:
                chunks = body.split('---', 2)
                if len(chunks) >= 3:
                    body = chunks[2].strip()
            # Trim
            if len(body) > 800:
                body = body[:800] + '...'
            ref_parts.append(f'[Run importance={r["importance"]}] {body}')
        parts.append('RECENT REFLECTIONS:\n' + '\n\n'.join(ref_parts))

    # 3. Active rules (confidence > 0.5)
    rules = load_rules()
    active_rules = [r for r in rules.rules if r.confidence > 0.5]
    if active_rules:
        rules_text = '\n'.join(
            f'- {r.rule} (confidence={r.confidence:.1f}, validated run {r.last_validated_run})'
            for r in active_rules[:10]  # Cap at 10 to control tokens
        )
        parts.append(f'ACTIVE REASONING RULES:\n{rules_text}')

    # 4. Strategies
    strategies = load_strategies()
    if strategies.strategies:
        strat_text = '\n'.join(
            f'- {s.name}: {s.description} (useful {s.times_useful}/{s.times_applied} times)'
            for s in strategies.strategies[:8]
        )
        parts.append(f'REASONING STRATEGIES:\n{strat_text}')

    # 5. Prior report section (original behavior)
    if previous_report and section_name:
        import re
        pattern = rf'^## \d*b?\.?\s*{re.escape(section_name)}'
        lines = previous_report.split('\n')
        capture = False
        section_lines = []
        for line in lines:
            if re.match(pattern, line):
                capture = True
                continue
            elif capture and line.startswith('## '):
                break
            elif capture:
                section_lines.append(line)
        if section_lines:
            section_text = '\n'.join(section_lines).strip()
            if len(section_text) > 800:
                section_text = section_text[:800] + '...'
            parts.append(f'PREVIOUS RUN — {section_name}:\n{section_text}')

    # 6. Changed notes
    if changed_note_titles:
        if len(changed_note_titles) <= 20:
            parts.append(f'NOTES CHANGED SINCE LAST RUN ({len(changed_note_titles)}): '
                         + ', '.join(changed_note_titles))
        else:
            parts.append(f'NOTES CHANGED SINCE LAST RUN: {len(changed_note_titles)} notes '
                         f'(including: {", ".join(changed_note_titles[:15])}...)')

    if not parts:
        return ''

    return ('\n\n--- MEMORY CONTEXT ---\n'
            + '\n\n'.join(parts)
            + '\n\nIMPORTANT: Build on prior memory. Focus on what is NEW or CHANGED. '
            'Do not repeat prior findings unless they have meaningfully evolved.\n'
            '--- END MEMORY CONTEXT ---\n\n')


def _format_living_summary(summary: LivingSummary) -> str:
    """Format the living summary for prompt injection."""
    parts = []
    tm = summary.thematic_map
    if tm.macro_themes:
        parts.append(f'Macro themes: {", ".join(tm.macro_themes)}')
    if tm.emerging_themes:
        parts.append(f'Emerging: {", ".join(tm.emerging_themes)}')
    if tm.declining_themes:
        parts.append(f'Declining: {", ".join(tm.declining_themes)}')
    if summary.active_hypotheses:
        parts.append(f'Hypotheses: {"; ".join(summary.active_hypotheses)}')
    if summary.known_blind_spots:
        parts.append(f'Blind spots: {"; ".join(summary.known_blind_spots)}')
    ge = summary.graph_evolution
    if ge.growth_rate:
        parts.append(f'Growth: {ge.growth_rate}')
    if ge.stability_note:
        parts.append(f'Stability: {ge.stability_note}')
    if summary.meta_observations:
        parts.append(f'Meta: {"; ".join(summary.meta_observations)}')
    return '\n'.join(parts)


# ── Post-Run Orchestrator ────────────────────────────────────────────────

def post_run_memory(report: str, api_key: str, call_haiku,
                    progress_cb=None) -> dict:
    """Run all four post-run memory steps. Each is independent — partial failure OK.

    Returns dict with status of each step.
    """
    def _progress(msg):
        if progress_cb:
            progress_cb('memory', msg)

    run_number = _get_run_number()
    results = {
        'run_number': run_number,
        'reflection': None,
        'summary': False,
        'rules': False,
        'strategies': False,
    }

    # 1. Generate reflection node
    _progress(f'Generating reflection (run #{run_number})...')
    try:
        ref_path = generate_reflection(report, run_number, api_key, call_haiku)
        results['reflection'] = ref_path
        if ref_path:
            _progress(f'Reflection written: {os.path.basename(ref_path)}')
        else:
            _progress('Reflection generation failed (logged)')
    except Exception as e:
        logger.error(f'Reflection generation crashed: {e}')
        _progress(f'Reflection generation crashed: {e}')

    # 2. Update living summary
    _progress('Updating living summary...')
    try:
        results['summary'] = update_living_summary(report, api_key, call_haiku)
        if results['summary']:
            _progress('Living summary updated')
        else:
            _progress('Living summary update failed (logged)')
    except Exception as e:
        logger.error(f'Living summary update crashed: {e}')
        _progress(f'Living summary update crashed: {e}')

    # 3. Extract rules
    _progress('Extracting reasoning rules...')
    try:
        results['rules'] = extract_rules(report, run_number, api_key, call_haiku)
        if results['rules']:
            _progress('Rules updated')
        else:
            _progress('Rule extraction failed (logged)')
    except Exception as e:
        logger.error(f'Rule extraction crashed: {e}')
        _progress(f'Rule extraction crashed: {e}')

    # 4. Evaluate strategies
    _progress('Evaluating reasoning strategies...')
    try:
        results['strategies'] = evaluate_strategies(report, run_number, api_key, call_haiku)
        if results['strategies']:
            _progress('Strategies updated')
        else:
            _progress('Strategy evaluation failed (logged)')
    except Exception as e:
        logger.error(f'Strategy evaluation crashed: {e}')
        _progress(f'Strategy evaluation crashed: {e}')

    successes = sum([
        results['reflection'] is not None,
        results['summary'],
        results['rules'],
        results['strategies'],
    ])
    _progress(f'Memory system: {successes}/4 steps succeeded (run #{run_number})')

    return results
