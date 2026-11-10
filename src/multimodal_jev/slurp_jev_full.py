"""Full SLURP typed bundles: label-free definitions and all entity occurrences."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .baseline import atomic_json, file_hash
from .consumption import canonical, validate_request
from .task_schema import Candidate, DecisionSchema


def make_bank(taxonomy):
    pairs = taxonomy['scenario_action_pairs']
    types = taxonomy['entity_types']
    if len({tuple(p) for p in pairs}) != len(pairs) or len(set(types)) != len(types):
        raise ValueError('unique taxonomy required')
    candidates = tuple(Candidate(f'i{i:02d}', f'scenario={s}; action={a}')
                       for i, (s, a) in enumerate(pairs))
    tasks = {}
    for view in ('native', 'permutation', 'rebind'):
        options = candidates
        if view == 'rebind':
            options = tuple(Candidate(candidates[(i+1) % len(candidates)].id, c.description)
                            for i, c in enumerate(candidates))
        key = 'intent.' + view
        tasks[key] = asdict(DecisionSchema(key, 'choice',
            'Which scenario and action pair describes the spoken request?', options, allow_unknown=False))
    for entity in types:
        definition = entity.replace('_', ' ')
        for polarity in ('present', 'absent'):
            key = f'entity.{entity}.{polarity}'
            question = (f'Does the utterance contain at least one {definition} entity?' if polarity == 'present'
                        else f'Does the utterance contain no {definition} entity?')
            tasks[key] = asdict(DecisionSchema(key, 'noul', question, allow_unknown=False))
        key = f'entity.{entity}.values'
        tasks[key] = asdict(DecisionSchema(key, 'open_slots',
            f'List every {definition} entity in spoken order using its lowercase spoken surface words. '
            'Preserve repeated occurrences; return an empty list when none is present.', allow_unknown=False))
    return {'schema': 'slurp-jev-task-bank-v1', 'scenario_action_pairs': pairs,
            'entity_types': types, 'tasks': tasks}


def validate_bundle(bundle, bank):
    if set(bundle) != {'id', 'split', 'source_group', 'audio_sha256', 'input'}:
        raise ValueError('label-free bundle required')
    if set(bundle['input']) != {'audio', 'task_bank_sha256'}:
        raise ValueError('audio and task bank reference only')
    if bundle['input']['task_bank_sha256'] != hashlib.sha256(canonical(bank)).hexdigest():
        raise ValueError('task bank changed')
    if bundle['split'] not in ('train', 'valid', 'test') or any(
            type(bundle[k]) is not str or not bundle[k] for k in ('id','source_group','audio_sha256')):
        raise ValueError('bundle identity required')
    if type(bundle['input']['audio']) is not str or not bundle['input']['audio']:
        raise ValueError('audio required')


def expand_bundle(bundle, bank):
    """Resolve with input alone; never use gold to choose a query or candidate."""
    validate_bundle(bundle, bank)
    for key, definition in bank['tasks'].items():
        schema = dict(definition)
        if key == 'intent.permutation':
            schema['candidates'] = sorted(definition['candidates'], key=lambda c:
                hashlib.sha256((bundle['source_group']+'|'+c['id']).encode()).digest())
        yield validate_request({'id': bundle['id']+'::'+key, 'split': bundle['split'],
            'source_group': bundle['source_group'], 'audio_sha256': bundle['audio_sha256'],
            'input': {'audio': bundle['input']['audio'], 'schema': schema}})


def values_from_target(target, bank):
    index = bank['scenario_action_pairs'].index([target['scenario'], target['action']])
    values = {'intent.native': f'i{index:02d}', 'intent.permutation': f'i{index:02d}',
              'intent.rebind': f'i{(index+1) % len(bank["scenario_action_pairs"]):02d}'}
    if any(e['type'] not in bank['entity_types'] for e in target['entities']):
        raise ValueError('unknown entity type')
    for entity in bank['entity_types']:
        fillers = [e['filler'] for e in target['entities'] if e['type'] == entity]
        values[f'entity.{entity}.present'] = bool(fillers)
        values[f'entity.{entity}.absent'] = not fillers
        values[f'entity.{entity}.values'] = fillers
    if set(values) != set(bank['tasks']) or any(
            not DecisionSchema.from_dict(bank['tasks'][k]).accepts(v) for k,v in values.items()):
        raise ValueError('typed target mismatch')
    return values


def reconstruct(values, bank):
    """Native entity multiset reconstruction; no cross-type ordering claim."""
    if set(values) != set(bank['tasks']): raise ValueError('complete task population required')
    if any(not DecisionSchema.from_dict(bank['tasks'][k]).accepts(v) for k,v in values.items()):
        raise ValueError('invalid typed values')
    native = bank['tasks']['intent.native']['candidates']
    index = next(i for i,c in enumerate(native) if c['id'] == values['intent.native'])
    scenario, action = bank['scenario_action_pairs'][index]
    entities = [{'type': t, 'filler': v} for t in bank['entity_types'] for v in values[f'entity.{t}.values']]
    return {'scenario': scenario, 'action': action, 'entities': entities}


def build(native, destination, state):
    native, destination, state = Path(native), Path(destination), Path(state)
    manifest = json.loads((native/'MANIFEST.json').read_text())
    taxonomy = json.loads((native/'taxonomy.json').read_text())
    if file_hash(native/'taxonomy.json') != manifest['identity']['taxonomy_sha256']:
        raise ValueError('native taxonomy changed')
    bank = make_bank(taxonomy); bank_hash = hashlib.sha256(canonical(bank)).hexdigest()
    identity = {'schema': 'slurp-jev-full-bundle-v1', 'native_release': str(native),
        'native_manifest_sha256': file_hash(native/'MANIFEST.json'), 'task_bank_sha256': bank_hash,
        'builder_sha256': file_hash(__file__),
        'schema_sha256': file_hash(Path(__file__).with_name('task_schema.py')),
        'consumption_sha256': file_hash(Path(__file__).with_name('consumption.py'))}
    release_id = hashlib.sha256(canonical(identity)).hexdigest()
    output = destination/release_id
    destination.mkdir(parents=True, exist_ok=True); state.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError('immutable release exists')
    stage = Path(tempfile.mkdtemp(prefix='.building-', dir=destination))
    (stage/'task-bank.json').write_bytes(canonical(bank))
    splits = {}; groups = {}; audio_splits = {}; total = 0; start = time.monotonic()
    for split in ('train', 'valid', 'test'):
        pins = manifest['splits'][split]
        for kind in ('requests', 'targets'):
            if file_hash(native/f'{split}.{kind}.jsonl') != pins[kind+'_sha256']:
                raise ValueError('native release file changed')
        seen = set(); split_groups = set(); positives = repeated = occurrences = 0
        with (native/f'{split}.requests.jsonl').open() as ri, (native/f'{split}.targets.jsonl').open() as ti, \
             (stage/f'{split}.bundles.jsonl').open('wb') as ro, (stage/f'{split}.targets.jsonl').open('wb') as to:
            for request_line, target_line in zip(ri, ti, strict=True):
                request, target = json.loads(request_line), json.loads(target_line)
                uid = request['id']; group = request['source_group']; audio_hash = request['audio_sha256']
                if uid in seen or uid != target['id'] or request['split'] != split:
                    raise ValueError('native identity mismatch')
                if hashlib.sha256(canonical(request)).hexdigest() != target['request_sha256']:
                    raise ValueError('native target binding changed')
                if groups.get(group, split) != split or audio_splits.get(audio_hash, split) != split:
                    raise ValueError('source group or audio crosses splits')
                groups[group] = split; audio_splits[audio_hash] = split
                bundle = {k:request[k] for k in ('id','split','source_group','audio_sha256')}
                bundle['input'] = {'audio': request['input']['audio'], 'task_bank_sha256': bank_hash}
                validate_bundle(bundle, bank)
                values = values_from_target(target['target'], bank)
                reconstructed = reconstruct(values, bank)
                assert reconstructed['scenario'] == target['target']['scenario']
                assert reconstructed['action'] == target['target']['action']
                if Counter((e['type'],e['filler']) for e in reconstructed['entities']) != Counter(
                        (e['type'],e['filler']) for e in target['target']['entities']):
                    raise ValueError('native entity multiplicity lost')
                ro.write(canonical(bundle))
                to.write(canonical({'id':uid, 'values':values,
                    'bundle_sha256':hashlib.sha256(canonical(bundle)).hexdigest(),
                    'native_target_sha256':hashlib.sha256(canonical(target)).hexdigest(),
                    'native_target_id':uid}))
                lengths = [len(values[f'entity.{t}.values']) for t in bank['entity_types']]
                positives += sum(n>0 for n in lengths); repeated += sum(n>1 for n in lengths)
                occurrences += sum(lengths); seen.add(uid); split_groups.add(group); total += 1
                if total % 500 == 0:
                    atomic_json(state/'progress.json', {'phase':'export_full_jev','split':split,
                        'recordings':total,'heartbeat_unix':time.time(),'elapsed_seconds':time.monotonic()-start})
        if len(seen) != pins['recordings']: raise ValueError('native recording coverage changed')
        splits[split] = {'recordings':len(seen),'source_groups':len(split_groups),
            'decisions':len(seen)*len(bank['tasks']), 'positive_entity_queries':positives,
            'negative_entity_queries':len(seen)*len(bank['entity_types'])-positives,
            'multiple_occurrence_queries':repeated, 'entity_occurrences':occurrences,
            'bundles_sha256':file_hash(stage/f'{split}.bundles.jsonl'),
            'targets_sha256':file_hash(stage/f'{split}.targets.jsonl')}
    receipt = {'release_id':release_id,'identity':identity,'splits':splits,
        'decisions_per_recording':len(bank['tasks']), 'native_entity_multiset_parity':True,
        'elapsed_seconds':time.monotonic()-start,
        'scope':'full native source coverage, derived typed tasks; not inference or official SLU score'}
    (stage/'MANIFEST.json').write_bytes(canonical(receipt)); os.rename(stage,output)
    return output, receipt
