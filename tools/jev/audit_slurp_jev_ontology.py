#!/usr/bin/env python3
"""Lock matched intent/type identifiers from authenticated TRAIN inventory.

Reads learning labels for class support only. Does not read selection/test gold,
create reviewed negatives, approve training, or claim a model evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_benchmark import ontology
import hashlib


def run(receipt_path, omni_lock, output):
    receipt_path, omni_lock, output = map(Path,(receipt_path,omni_lock,output))
    receipt = json.loads(receipt_path.read_text())
    source = Path(receipt['release']); manifest_path = source/'MANIFEST.json'
    if file_hash(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('verified source manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'slurp-clean-source-bundles-v1':
        raise ValueError('clean source bundle release required')
    names = ('taxonomy-inventory.json','learning.requests.jsonl','learning.targets.jsonl')
    for name in names:
        if file_hash(source/name) != manifest['files'][name]:
            raise ValueError('source asset changed: '+name)
    reference = json.loads(omni_lock.read_text()); omni = Path(reference['source_root'])
    fold_path = omni/'docs/results/s-stages/FOLDS.json'
    if file_hash(fold_path) != reference['source_files'][str(fold_path.relative_to(omni))]:
        raise ValueError('omni fold source changed')
    # This historical receipt is used only for its intent identifier inventory;
    # its filler examples and old unclean TRAIN counts never become supervision.
    intent_path = omni/'studies/speech-aware-evidence-acquisition/docs/receipts/knowledge/slurp-intent-slot-schema-v1.json'
    intent_receipt = json.loads(intent_path.read_text())
    bank = ontology(json.loads((source/names[0]).read_text()),json.loads(fold_path.read_text()),
                    [r['intent'] for r in intent_receipt['intents']])
    requests = {}
    for line in (source/names[1]).read_text().splitlines():
        r = json.loads(line)
        if r['id'] in requests or r['role'] != 'learning' or r['official_split'] != 'train':
            raise ValueError('unique learning-only source required')
        requests[r['id']] = r
    intents = Counter(); type_recordings = Counter(); type_sources = {k:set() for k in bank['slot_types']}
    seen = set(); empty = 0
    for line in (source/names[2]).read_text().splitlines():
        t = json.loads(line); uid = t['id']
        if uid in seen or uid not in requests: raise ValueError('target roster mismatch')
        seen.add(uid); r = requests[uid]
        if hashlib.sha256(canonical(r)).hexdigest() != t['request_sha256']:
            raise ValueError('target not bound to request')
        target = t['native_target']; label = target['scenario']+'_'+target['action']
        if label not in {i['id'] for i in bank['intents']}: raise ValueError('unmapped learning intent')
        intents[label] += 1
        types = {e['type'] for e in target['entities']}
        if not types <= set(bank['slot_types']): raise ValueError('unmapped learning type')
        empty += not types
        type_recordings.update(types)
        for name in types: type_sources[name].add(r['source_group'])
    if seen != set(requests) or len(seen) != receipt['counts']['learning']:
        raise ValueError('incomplete learning roster')
    identity = {str(p):file_hash(p) for p in [receipt_path,omni_lock,manifest_path,fold_path,intent_path,
                *(source/n for n in names),Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/slurp_jev_benchmark.py']}
    report = dict(schema='slurp-jev-ontology-audit-v1',passed=True,source_files=identity,
        omni_git_head=reference['source_git_head'],learning_recordings=len(seen),
        intent_classes=len(bank['intents']),slot_types=len(bank['slot_types']),folds=len(bank['type_folds']),
        raw_aliases=bank['raw_alias_count'],ambiguous_aliases=bank['ambiguous_raw_aliases'],
        learning_intent_recordings=dict(sorted(intents.items())),
        learning_type_recordings={t:type_recordings[t] for t in bank['slot_types']},
        learning_type_source_groups={t:len(type_sources[t]) for t in bank['slot_types']},
        learning_empty_native_annotations=empty,
        selection_targets_opened=False,test_opened=False,audio_rehashed=False,
        optimization_ready=False,
        scope='Exact 60-intent/55-type mapping and learning native-annotation support only. No semantic approval, reviewed negative labels, common evaluation roster or model score.')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'ontology.json',bank)
    report['ontology_sha256']=file_hash(output/'ontology.json')
    atomic_json(output/'audit.json',report)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-receipt',required=True);p.add_argument('--omni-lock',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.source_receipt,a.omni_lock,a.output)
    print(json.dumps({k:r[k] for k in ('passed','learning_recordings','intent_classes','slot_types','folds','raw_aliases','ambiguous_aliases')},indent=2))
