#!/usr/bin/env python3
"""Aggregate learning-only annotation support; does not approve semantic labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_definitions import definitions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def has_nan(value):
    if isinstance(value, dict):
        return any(has_nan(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nan(v) for v in value)
    return value == 'NAN'


def audit(slurp, sfc, partition, partition_receipt):
    inputs = {}
    manifest = json.loads((slurp / 'MANIFEST.json').read_text())
    for name in ('learning.requests.jsonl', 'learning.targets.jsonl'):
        inputs[str(slurp / name)] = digest(slurp / name)
        assert inputs[str(slurp / name)] == manifest['files'][name]
    requests = {r['id']: r for r in rows(slurp / 'learning.requests.jsonl')}
    sources = {}
    for target in rows(slurp / 'learning.targets.jsonl'):
        request = requests.pop(target['id'])
        assert request['role'] == 'learning' and request['official_split'] == 'train'
        assert target['request_sha256'] == hashlib.sha256(canonical(request)).hexdigest()
        assert target['annotation_status'] == 'available'
        assert target['optimization_allowed'] is False
        for uid in target['source_ids']:
            gold = target['native_target']
            assert uid not in sources or sources[uid] == gold
            sources[uid] = gold
    assert not requests
    type_counts, cardinalities = Counter(), Counter()
    signatures, type_intents = defaultdict(Counter), defaultdict(set)
    repeats = 0
    for gold in sources.values():
        intent = gold['scenario'] + '/' + gold['action']
        kinds = [entity['type'] for entity in gold['entities']]
        types = set(kinds)
        type_counts.update(types)
        cardinalities[len(types)] += 1
        signatures[intent][tuple(sorted(types))] += 1
        repeats += len(kinds) != len(types)
        for kind in types:
            type_intents[kind].add(intent)
    slurp_report = dict(
        source_utterances=len(sources), intent_pairs=len(signatures), types=len(type_counts),
        annotated_type_cardinality=dict(sorted(cardinalities.items())),
        repeated_same_type_sources=repeats,
        intents_with_multiple_annotated_type_sets=sum(len(v) > 1 for v in signatures.values()),
        types_in_multiple_intents=sum(len(v) > 1 for v in type_intents.values()),
        type_source_support=dict(sorted(type_counts.items())),
        # In-sample majority lookup is only a shortcut diagnostic, not a model baseline.
        oracle_intent_majority_typeset_resubstitution=dict(
            correct=sum(max(v.values()) for v in signatures.values()), total=len(sources)),
    )
    receipt = json.loads(partition_receipt.read_text())
    inputs[str(partition)] = digest(partition)
    assert inputs[str(partition)] == receipt['partition_sha256']
    part = json.loads(partition.read_text())
    learning = set(part['learning'])
    assert not learning.intersection(part['selection'])
    native_manifest = json.loads((sfc / 'MANIFEST.json').read_text())
    for suffix in ('requests', 'targets'):
        path = sfc / f'train_id.{suffix}.jsonl'
        inputs[str(path)] = digest(path)
        assert inputs[str(path)] == native_manifest['splits']['train_id'][suffix + '_sha256']
    requests = {r['id']: r for r in rows(sfc / 'train_id.requests.jsonl')
                if r['source_group'] in learning}
    counts = Counter()
    groups = defaultdict(dict)
    required_schemas = Counter()
    for target in rows(sfc / 'train_id.targets.jsonl'):
        # Whole TRAIN authenticated; selection annotations are not analyzed.
        if target['metadata']['source_group'] not in learning:
            continue
        request = requests.pop(target['id'])
        assert request['split'] == 'train_id'
        assert target['request_sha256'] == hashlib.sha256(canonical(request)).hexdigest()
        funcs = {f['name']: f for f in definitions(request['input'])['functions']}
        calls = target['supervision']['calls']
        counts['bundles'] += 1
        counts['calls'] += len(calls)
        counts['multi_call_bundles'] += len(calls) > 1
        counts['repeated_function_bundles'] += len(calls) > len({c['name'] for c in calls})
        counts['bundles_with_NAN'] += any(has_nan(c['arguments']) for c in calls)
        counts['two_audio_bundles'] += len(request['input']['audios']) == 2
        for call in calls:
            schema = funcs[call['name']]['parameters']
            required_schemas['explicit_required'] += 'required' in schema
            required_schemas['missing_required_keyword'] += 'required' not in schema
            args = call['arguments']
            required = schema.get('required', [])
            counts['required_argument_positions'] += len(required)
            counts['required_positions_absent'] += sum(k not in args for k in required)
            counts['required_positions_with_NAN'] += sum(k in args and has_nan(args[k]) for k in required)
            counts['calls_with_array_argument'] += any(isinstance(v, list) for v in args.values())
        group = target['metadata']['source_group']
        stage = target['metadata']['stage']
        assert stage not in groups[group]
        groups[group][stage] = calls
    assert not requests and set(groups) == learning
    transitions = Counter()
    for stages in groups.values():
        if 2 not in stages:
            assert set(stages) == {1}
            continue
        assert set(stages) == {1, 2}
        before, after = stages[1], stages[2]
        transitions['two_stage_groups'] += 1
        transitions['NAN_before'] += any(has_nan(c['arguments']) for c in before)
        transitions['NAN_after'] += any(has_nan(c['arguments']) for c in after)
        transitions['NAN_to_no_NAN'] += (any(has_nan(c['arguments']) for c in before)
                                         and not any(has_nan(c['arguments']) for c in after))
        # Multisets, preserving repeat calls but ignoring list ordering.
        transitions['call_multiset_changed'] += Counter(map(lambda c: json.dumps(c, sort_keys=True), before)) != Counter(map(lambda c: json.dumps(c, sort_keys=True), after))
        transitions['function_multiset_changed'] += Counter(c['name'] for c in before) != Counter(c['name'] for c in after)
        if (len({c['name'] for c in before}) == len(before)
                and len({c['name'] for c in after}) == len(after)
                and {c['name'] for c in before} == {c['name'] for c in after}):
            transitions['uniquely_function_aligned_groups'] += 1
            after_by_name = {c['name']: c['arguments'] for c in after}
            for call in before:
                old, new = call['arguments'], after_by_name[call['name']]
                transitions['argument_keysets_changed'] += set(old) != set(new)
                for key in old.keys() & new.keys():
                    if has_nan(old[key]):
                        transitions['NAN_arguments_resolved'] += not has_nan(new[key])
                    else:
                        transitions['previously_known_arguments_preserved'] += old[key] == new[key]
                        transitions['previously_known_arguments_changed'] += old[key] != new[key]
    assert counts['bundles'] == part['summary']['learning']['bundles']
    for path in (slurp / 'MANIFEST.json', sfc / 'MANIFEST.json', partition_receipt, Path(__file__)):
        inputs[str(path)] = digest(path)
    return dict(schema='decision-chain-learning-support-v1', input_sha256=inputs,
                slurp=slurp_report, sfc=dict(counts=counts, required_schemas=required_schemas,
                                            transitions=transitions, source_groups=len(groups)),
                optimization_ready=False, inference_run=False, test_opened=False,
                selection_scope='SFC whole TRAIN streamed for authentication and learning-group filtering; no selection annotation statistics or examples used. SLURP learning files only.',
                limitations=['Annotation structure, not audio semantic adjudication.',
                             'Function-aligned argument changes are annotation facts, not verified audio entailment or autonomous clarification policy.',
                             'Native presence absence and temporal labels require separate quality acceptance.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('slurp', 'sfc', 'partition', 'partition-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.slurp, args.sfc, args.partition, args.partition_receipt)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, ensure_ascii=False, indent=2))
