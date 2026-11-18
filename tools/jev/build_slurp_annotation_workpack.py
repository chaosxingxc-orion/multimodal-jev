#!/usr/bin/env python3
"""Build the complete learning-only annotation workpack; never issue training labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, ensure_ascii=False)+'\n').encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def unique_index(items, key):
    result = {}
    for item in items:
        identity = str(item[key])
        if identity in result:
            raise ValueError('duplicate identity: '+identity)
        result[identity] = item
    return result


def source_bindings(population, targets):
    unique_index(population, 'id')
    output = defaultdict(list)
    for row in population:
        if row['role'] != 'learning' or row['official_split'] != 'train' or row['known_pending_source']:
            raise ValueError('only nonpending learning TRAIN permitted')
        target = targets[row['id']]
        if content_sha(target) != row['source_target_sha256']:
            raise ValueError('target identity changed')
        native = target['native_target']
        if row['intent'] != native['scenario']+'_'+native['action']:
            raise ValueError('native intent mismatch')
        ids = [str(s) for s in target['source_ids']]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('empty or duplicate source IDs')
        for sid in ids:
            output[sid].append(dict(recording_id=row['id'], source_group=row['source_group'],
                audio_sha256=row['audio_sha256'], source_target_sha256=row['source_target_sha256'],
                all_source_ids=ids))
    return output


def check_review(decision, family, metadata, intents, slots, policies):
    sid = str(decision['source_id'])
    if sid not in metadata or decision['optimization_allowed'] is not False:
        raise ValueError('review outside learning or unexpectedly admitted')
    if family == 'I_fit':
        name = decision['semantic_id']
        expected = content_sha(intents[name])
        metadata_hash = decision['metadata_sha256']
        if decision['exclusive_I_class_truth'] is not None:
            raise ValueError('definition fit is not exclusive I_class truth')
    elif family == 'S':
        name = decision['entity_type']
        expected = content_sha(slots[name])
        metadata_hash = decision['metadata_record_sha256']
        if decision['policy_sha256'] != content_sha(policies[name]):
            raise ValueError('stale slot policy')
    else:
        raise ValueError('unsupported review family')
    if decision['definition_sha256'] != expected or metadata_hash != content_sha(metadata[sid]):
        raise ValueError('stale review definition or source')
    return sid, name


def merge_reference(reviews, sid, family, definition, decision, report, duplicate_policy):
    """One assessed coordinate, with every agreeing reference assessment retained.

    This never votes, selects a winner, or upgrades admission. A changed value,
    identity or status requires an explicit adjudication rather than this merge.
    """
    entry=dict(family=family, definition_id=definition, decision=decision,
               decision_sha256=content_sha(decision), report=report)
    existing=next((e for e in reviews[sid] if
                   (e['family'],e['definition_id'])==(family,definition)),None)
    if existing is None:
        reviews[sid].append(entry)
        return True
    if duplicate_policy!='retain_agreeing_reference_assessments_v1':
        raise ValueError('duplicate source/definition review')
    if family!='S':raise ValueError('duplicate merge only defined for slot reference evidence')
    # Only the reference wording may differ; even unavailable -> true is not an
    # automatic refinement. Reject disagreements for separate adjudication.
    core=lambda d:{k:v for k,v in d.items() if k not in ('evidence','limitation')}
    if core(existing['decision'])!=core(decision):
        raise ValueError('inconsistent duplicate reference requires adjudication')
    if report==existing['report'] or any(a['report']==report for a in existing.get('additional_reference_assessments',[])):
        raise ValueError('same reference report included twice')
    existing.setdefault('additional_reference_assessments',[]).append(
        dict(decision=decision,decision_sha256=content_sha(decision),report=report))
    existing['merge_policy']=duplicate_policy
    existing['independent_review_inferred']=False
    return False


def validate_boundaries(document, slots, slot_sha, policy_sha):
    if (document['source_definition_sha256']!=slot_sha or
            document['source_policy_sha256']!=policy_sha):
        raise ValueError('boundary definition/policy mismatch')
    seen=set()
    for rule in document['rules']:
        types=rule['types']
        if len(types)<2 or len(set(types))!=len(types) or not set(types)<=set(slots):
            raise ValueError('invalid boundary type support')
        identity=tuple(sorted(types))
        if identity in seen:raise ValueError('duplicate boundary')
        seen.add(identity)


def main():
    parser = argparse.ArgumentParser()
    for name in ('release', 'evidence_config', 'intents', 'slots', 'policies', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--boundaries',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pins = {}

    def pin(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('dependency changed: '+str(path))
        if str(path) in pins and pins[str(path)] != actual:
            raise ValueError('inconsistent dependency')
        pins[str(path)] = actual

    pin(args.release/'MANIFEST.json', args.manifest_sha256)
    manifest = read(args.release/'MANIFEST.json')
    source = Path(manifest['source'])
    pin(source/'MANIFEST.json')
    source_manifest = read(source/'MANIFEST.json')
    pin(args.release/'learning.jsonl', manifest['files']['learning.jsonl'])
    pin(source/'learning.targets.jsonl', source_manifest['files']['learning.targets.jsonl'])
    metadata_path = source.parent/'metadata/train.jsonl'
    pin(metadata_path, source_manifest['identity']['input_hashes']['clean_metadata_sha256'])
    for path in (args.evidence_config, args.intents, args.slots, args.policies):
        pin(path)
    population = rows(args.release/'learning.jsonl')
    if len(population) != manifest['counts']['learning']['recordings']:
        raise ValueError('population count mismatch')
    targets = unique_index(rows(source/'learning.targets.jsonl'), 'id')
    bindings = source_bindings(population, targets)
    metadata = unique_index([r for r in rows(metadata_path) if str(r['slurp_id']) in bindings], 'slurp_id')
    if set(metadata) != set(bindings):
        raise ValueError('missing source metadata')
    intents = unique_index(read(args.intents)['cards'], 'semantic_id')
    slots = unique_index(read(args.slots)['types'], 'entity_type')
    policies = unique_index(read(args.policies)['policies'], 'entity_type')
    if set(policies) != set(slots) or read(args.policies)['source_definition_sha256'] != sha(args.slots):
        raise ValueError('slot policy/definition mismatch')
    if len(intents) != 60 or len(slots) != 55:
        raise ValueError('complete SLURP ontology required')
    boundary_document=None
    if args.boundaries is not None:
        pin(args.boundaries)
        boundary_document=read(args.boundaries)
        validate_boundaries(boundary_document,slots,sha(args.slots),sha(args.policies))
    evidence_config=read(args.evidence_config)
    duplicate_policy=evidence_config.get('duplicate_policy','reject')
    if duplicate_policy not in ('reject','retain_agreeing_reference_assessments_v1'):
        raise ValueError('unknown duplicate policy')
    reviews = defaultdict(list)
    coverage = {'I_fit': Counter(), 'S': Counter()}
    assessment_count=0
    for spec in evidence_config['sources']:
        path = Path(spec['report'])
        pin(path, spec['sha256'])
        report = read(path)
        for dependency, digest in report['pins'].items():
            pin(Path(dependency), digest)
        for filename, digest in report['files'].items():
            pin(path.parent/filename, digest)
        if spec['kind'] not in ('slot', 'intent_fit'):
            raise ValueError('unknown evidence kind')
        family = 'S' if spec['kind'] == 'slot' else 'I_fit'
        for d in rows(path.parent/'decisions.jsonl'):
            sid, definition = check_review(d, family, metadata, intents, slots, policies)
            added=merge_reference(reviews,sid,family,definition,d,str(path),duplicate_policy)
            coverage[family][definition] += int(added)
            assessment_count+=1
    # IDs are ordered independently of labels/answers; this workpack is never actor input.
    order = sorted(metadata, key=lambda sid: hashlib.sha256(('annotation-workpack-v1:'+sid).encode()).hexdigest())
    args.output.mkdir(parents=True)
    class_sources = Counter()
    with (args.output/'sources.jsonl').open('w') as stream:
        for sid in order:
            m = metadata[sid]
            native_intent = m['scenario']+'_'+m['action']
            if native_intent not in intents or not {e['type'] for e in m['entities']} <= slots.keys():
                raise ValueError('source outside ontology')
            for b in bindings[sid]:
                native = targets[b['recording_id']]['native_target']
                if (native['scenario'], native['action']) != (m['scenario'], m['action']):
                    raise ValueError('source/recording intent mismatch')
                if {e['type'] for e in native['entities']} != {e['type'] for e in m['entities']}:
                    raise ValueError('source/recording native type mismatch')
            class_sources[native_intent] += 1
            item = dict(source_id=sid, metadata_sha256=content_sha(m), reference=m['sentence'],
                native_intent=native_intent, native_annotations=m['entities'],
                recording_bindings=bindings[sid], existing_reference_evidence=reviews.get(sid, []),
                required_coverage='all definitions in ontology.json; unassessed is unavailable, never negative',
                semantic_annotation=None, exclusive_mapping_review=None,
                optimization_allowed=False, actor_input_allowed=False,
                evidence_scope='Reference annotation work item; no waveform listening or semantic admission.')
            stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False)+'\n')
    ontology = dict(intents=list(intents.values()), slots=list(slots.values()), policies=list(policies.values()),
        task_contract='configs/tasks/jev-task-design-v2.json',
        annotation_rules=['Assess complete utterance under frozen definitions before deriving predicate answers.',
            'Record individually assessed types or an explicit exhaustive coverage declaration; omission is not absence.',
            'Keep native labels, semantic proposals, exclusive mapping and admission separate.',
            'Existing I_fit evidence is neither I_class nor automatically I_present truth.',
            'Resolve policy conflicts before bulk annotation; independently review disagreements.',
            'All linked source IDs must support a recording label; do not propagate across source groups.',
            'Reference-only decisions do not certify audibility of all linked recordings.'])
    pin(Path(ontology['task_contract']))
    if boundary_document is not None:
        ontology['slot_boundary_document']=boundary_document
        ontology['slot_boundary_application']='Apply current explicit scope/role/overlap rules. Historical and evidence fields are provenance only.'
    (args.output/'ontology.json').write_text(json.dumps(ontology, ensure_ascii=False, indent=2)+'\n')
    per_definition = {family: {name: dict(total_source_pairs=len(metadata),
        reference_assessed=counts[name], without_reference=len(metadata)-counts[name], admitted=0)
        for name in sorted(intents if family == 'I_fit' else slots)} for family, counts in coverage.items()}
    report = dict(schema='slurp-full-annotation-workpack-v1', pins=pins, implementation_sha256=sha(__file__),
        learning_recordings=len(population), source_utterances=len(metadata),
        source_recording_bindings=sum(map(len, bindings.values())),
        reference_decisions=sum(map(len, reviews.values())),
        sources_with_reference=sum(bool(items) for items in reviews.values()),
        native_class_source_counts=dict(sorted(class_sources.items())), per_definition=per_definition,
        ontology_source_pair_counts=dict(I=len(metadata)*len(intents), S=len(metadata)*len(slots)),
        all_semantic_labels_admitted=0, training_dispatch_allowed=False,
        full_population_preserved=True, selection_and_test_read=False,
        files={name: sha(args.output/name) for name in ('sources.jsonl', 'ontology.json')},
        scope='Complete learning annotation input, not completed annotation or a training manifest.')
    if boundary_document is not None or duplicate_policy!='reject':
        report.update(reference_assessments=assessment_count,
            repeated_reference_assessments=assessment_count-report['reference_decisions'],
            duplicate_policy=duplicate_policy,
            shared_boundary_rules=len(boundary_document['rules']) if boundary_document else 0,
            full_slot_rules_supplied=boundary_document is not None)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('learning_recordings', 'source_utterances',
        'source_recording_bindings', 'reference_decisions', 'sources_with_reference', 'ontology_source_pair_counts')}))


if __name__ == '__main__':
    main()
