"""Strictly stage teacher instructions as data; semantic review remains required."""
import json
from pathlib import Path
import re

from .baseline import file_hash
from .consumption import canonical


def strict_json(text):
    def reject_constant(_):
        raise ValueError('nonfinite JSON')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def validate_candidates(payload, context):
    if type(payload) is not dict or set(payload) != {'schema', 'parent_prompt_sha256', 'diagnosis', 'candidates'}:
        raise ValueError('exact teacher schema required')
    if (payload['schema'] != 'jev-teacher-prompts-v1'
            or payload['parent_prompt_sha256'] != context['parent_prompt_sha256']):
        raise ValueError('teacher schema or parent prompt changed')
    evidence = {r['evidence_id'] for r in context['representatives']}

    def refs(row, allowed):
        ids = row['evidence_ids']
        if (type(ids) is not list or not ids or any(type(x) is not str for x in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= allowed):
            raise ValueError('unknown or duplicate teacher evidence IDs')

    def nonempty(row, keys):
        if any(type(row[k]) is not str or not row[k].strip() for k in keys):
            raise ValueError('nonempty teacher text required')

    if type(payload['diagnosis']) is not list or not payload['diagnosis']:
        raise ValueError('diagnosis required')
    for row in payload['diagnosis']:
        if type(row) is not dict or set(row) != {'mechanism', 'evidence_ids', 'uncertainty'}:
            raise ValueError('exact diagnosis fields required')
        nonempty(row, ('mechanism', 'uncertainty'))
        refs(row, {r['id'] for r in context['all_group_summary']})
    candidates = payload['candidates']
    if type(candidates) is not list or not 1 <= len(candidates) <= 4:
        raise ValueError('one to four candidates required')
    ids, instructions = set(), {context['parent_instruction'].strip()}
    for row in candidates:
        if type(row) is not dict or set(row) != {'id', 'instruction', 'mechanism', 'evidence_ids', 'counterevidence', 'falsification'}:
            raise ValueError('exact candidate fields required')
        nonempty(row, ('id', 'instruction', 'mechanism', 'counterevidence', 'falsification'))
        refs(row, evidence)
        if not re.fullmatch(r'p[1-4]', row['id']) or row['id'] in ids:
            raise ValueError('unique bounded candidate ID required')
        instruction = row['instruction']
        if len(instruction) > 4000 or instruction.strip() in instructions:
            raise ValueError('unique bounded replacement instruction required')
        # Structural screening cannot prove absence of semantic memorization.
        # Explicit review below is mandatory before any candidate evaluation.
        all_ids = {r['id'] for r in context['all_group_summary']}
        if any(uid in instruction for uid in all_ids):
            raise ValueError('sample identity leaked into instruction')
        ids.add(row['id']); instructions.add(instruction.strip())
    return candidates


def load_proposal(request_root):
    """Verify completed wire content even when a proposal fails structural screening."""
    root = Path(request_root)
    cfg = json.loads((root / 'executor-config.private.json').read_text())
    messages_path = Path(cfg['messages']['path'])
    if file_hash(messages_path) != cfg['messages']['sha256']:
        raise ValueError('teacher messages changed')
    messages = json.loads(messages_path.read_text())
    context = strict_json(messages[1]['content'])
    if context['split'] != 'train' or context['selection_or_test_feedback'] is not False:
        raise ValueError('TRAIN-only teacher evidence required')
    response_root = Path(cfg['output_root'])
    result = json.loads((response_root / 'result.json').read_text())
    if (result['state'] != 'response_received' or result['http_status'] != 200
            or result['finish_reasons'] != ['stop']):
        raise ValueError('complete nontruncated teacher response required')
    if file_hash(response_root / 'response.sse') != result['response_sha256']:
        raise ValueError('teacher wire response changed')
    response = json.loads((response_root / 'response.json').read_text())
    # Bind visible final content to persisted wire bytes without inspecting or
    # promoting private reasoning content.
    chunks, finishes = [], []
    done = False
    pending = []
    for line in (response_root / 'response.sse').read_text().splitlines():
        if line.startswith('data:'):
            pending.append(line[5:].lstrip())
        elif not line and pending:
            data = '\n'.join(pending); pending = []
            if data == '[DONE]':
                done = True
                break
            event = strict_json(data)
            for choice in event.get('choices', []):
                if choice['index'] != 0: raise ValueError('one completion required')
                delta = choice.get('delta', {})
                if delta.get('content') is not None: chunks.append(delta['content'])
                if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
    content = ''.join(chunks)
    if (not done or len(response['choices']) != 1 or finishes != ['stop']
            or response['choices'][0]['finish_reason'] != 'stop'
            or response['choices'][0]['message']['content'] != content):
        raise ValueError('teacher final content custody changed')
    payload = strict_json(content)
    return cfg, context, result, payload


def stage_candidates(request_root, output):
    root, output = Path(request_root), Path(output)
    cfg, context, result, payload = load_proposal(root)
    candidates = validate_candidates(payload, context)
    # Registered narrower per-request budgets override the generic four-prompt
    # schema. Check before creating output, not after staging executable inputs.
    if (root / 'prepared.json').exists():
        prepared = json.loads((root / 'prepared.json').read_text())
        limit = prepared.get('max_candidates', 4)
        characters = prepared.get('max_instruction_characters', 4000)
        if (type(limit) is not int or not 1 <= limit <= 4
                or type(characters) is not int or not 1 <= characters <= 4000):
            raise ValueError('invalid registered candidate limits')
        if len(candidates) > limit or any(len(r['instruction']) > characters for r in candidates):
            raise ValueError('teacher exceeded registered candidate budget')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'proposal.json').write_bytes(canonical(payload))
    receipt = {'schema': 'jev-teacher-staging-v1', 'semantic_review': 'pending',
               'candidate_accepted': False, 'parent_prompt_sha256': context['parent_prompt_sha256'],
               'messages_sha256': cfg['messages']['sha256'],
               'response_sha256': result['response_sha256'], 'usage': result.get('usage'),
               'candidates': []}
    for row in candidates:
        path = output / (row['id'] + '.txt')
        path.write_text(row['instruction'])
        receipt['candidates'].append({'id': row['id'], 'instruction_sha256': file_hash(path)})
    (output / 'staged.json').write_bytes(canonical(receipt))
    return receipt
