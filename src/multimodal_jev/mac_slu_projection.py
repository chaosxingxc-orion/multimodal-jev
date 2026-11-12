"""Lossless MAC-SLU frame conversion and input-only finite decision schemas.

Source absence is not negative supervision without an explicit completeness
review. Frame multiplicity/slot values remain in targets, outside Span inputs.
"""
from copy import deepcopy
import ast
import hashlib
import json

from .span_contract import SpanQuestion, SpanRequest
from .task_schema import DecisionSchema


def convert_annotation(row):
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('source id required')
    semantics = row.get('semantics')
    if not isinstance(semantics, dict):
        raise ValueError('raw semantics mapping required, not normalized list')
    frames, unresolved = [], []
    if not semantics:
        unresolved.append(dict(kind='empty_semantics', instance=None))
    for instance, domains in semantics.items():
        if not isinstance(instance, str) or not isinstance(domains, dict):
            raise ValueError('named instance mapping required')
        if not domains:
            unresolved.append(dict(kind='empty_instance', instance=instance))
        for domain, entries in domains.items():
            if not isinstance(domain, str) or not domain or not isinstance(entries, list):
                raise ValueError('domain and slot list required')
            if any(not isinstance(s, dict) or not {'name', 'value'} <= set(s)
                   or not isinstance(s['name'], str) or not s['name'] for s in entries):
                raise ValueError('explicit named source values required')
            labels = [s['value'] for s in entries if s['name'] == 'intent']
            if len(labels) != 1 or not isinstance(labels[0], str) or not labels[0]:
                raise ValueError('exactly one explicit intent per domain frame required')
            frames.append(dict(instance=instance, domain=domain, intent=labels[0],
                               slots=deepcopy([s for s in entries if s['name'] != 'intent']),
                               source_entries=deepcopy(entries)))
    result = dict(id=row['id'], frames=frames, unresolved=unresolved,
                  instance_order=list(semantics), annotation_complete=False,
                  supervision='explicit_positive_only_pending_completeness_review')
    if reconstruct_annotation(result) != semantics:
        raise ValueError('lossless annotation reconstruction failed')
    return result


def reconstruct_annotation(target):
    result = {key: {} for key in target['instance_order']}
    for frame in target['frames']:
        instance, domain = frame['instance'], frame['domain']
        if instance not in result or domain in result[instance]:
            raise ValueError('duplicated or unknown frame identity')
        result[instance][domain] = deepcopy(frame['source_entries'])
    return result


def official_frames(target):
    """Require representability, preserving frame order and duplicate intents.

    This is format conversion only, not completeness/semantic certification.
    """
    if target['unresolved']:
        raise ValueError('unresolved source containers cannot become official empty gold')
    result = []
    for frame in target['frames']:
        slots = frame['slots']
        if len({s['name'] for s in slots}) != len(slots):
            raise ValueError('official slot dict would discard repeated roles')
        result.append(dict(domain=frame['domain'], intent=frame['intent'],
                           slots={s['name']: deepcopy(s['value']) for s in slots}))
    return result


def declared_bank(source_text, training_targets):
    """Fixed public menu plus declared TRAIN-only additions; no test gold access."""
    constants = {}
    for node in ast.parse(source_text).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {'DOMAIN_INTENT_LIST', 'SLOT_LIST'}:
                    constants[name.id] = node.value.value
    intents, roles = set(), {}
    domain = None
    for line in constants['DOMAIN_INTENT_LIST'].splitlines():
        if line.startswith('- '): domain = line[2:]
        elif line.startswith('    - ') and domain: intents.add((domain, line[6:]))
    for line in constants['SLOT_LIST'].splitlines():
        if not line.startswith('- '): continue
        key, description = line[2:].split(':', 1)
        domain, role = key.split('-', 1)
        if (domain, role) in roles: raise ValueError('duplicate public role definition')
        roles[domain, role] = description.strip()
    if not intents or not roles: raise ValueError('nonempty public definitions required')
    original = set(intents)
    for target in training_targets:
        for frame in target['frames']:
            intents.add((frame['domain'], frame['intent']))
            for slot in frame['slots']:
                if (frame['domain'], slot['name']) not in roles:
                    raise ValueError('TRAIN role lacks public definition')
    body = dict(schema='mac-slu-finite-menu-v1',
                intents=[dict(domain=d, intent=i) for d, i in sorted(intents)],
                roles=[dict(domain=d, role=r, description=v) for (d, r), v in sorted(roles.items())],
                train_only_intent_additions=[dict(domain=d, intent=i) for d, i in sorted(intents-original)],
                scope='Public menu + TRAIN-only additions; not unchanged official candidate condition.')
    body['bank_id'] = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return body


def questions_for_intent(bank, domain, intent):
    if dict(domain=domain, intent=intent) not in bank['intents']:
        raise ValueError('intent outside fixed bank')
    definitions = [('intent', f'音频中是否存在领域“{domain}”、意图“{intent}”的语义帧？')]
    for role in bank['roles']:
        if role['domain'] != domain: continue
        definitions.append(('role:' + role['role'],
            f'音频中是否存在领域“{domain}”、意图“{intent}”的语义帧，且该帧具有槽位角色“{role["role"]}”？'
            f'角色定义：{role["description"]} 不同意图的槽位不能混用。'))
    return tuple(SpanQuestion(key, DecisionSchema(id=key, decision_type='noul',
                 question=question, allow_unknown=False)) for key, question in definitions)


def input_request(audio, bank, domain, intent):
    # This function accepts no source annotation, query, split_sens or target.
    questions = questions_for_intent(bank, domain, intent)
    return SpanRequest(audio=audio, questions=questions, final_id=questions[-1].id,
        requirements=('直接听取音频，逐项判断给定定义。',
                      '同时输出全部布尔决策，保留意图与槽位角色的关联，不生成转写或解释。'))


def decision_targets(target, bank, domain, intent):
    frames = [f for f in target['frames'] if (f['domain'], f['intent']) == (domain, intent)]
    present_roles = {s['name'] for f in frames for s in f['slots']}
    result = []
    for question in questions_for_intent(bank, domain, intent):
        present = bool(frames) if question.id == 'intent' else question.id[5:] in present_roles
        result.append(dict(id=question.id, status='available' if present else 'unavailable',
                           value=True if present else None,
                           reason='explicit_source_positive' if present else 'absence_not_completeness_reviewed'))
    return result
