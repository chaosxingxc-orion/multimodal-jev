#!/usr/bin/env python3
"""Rescore accepted immutable native predictions on a pinned clean TEST subset."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.slurp_native import native_target
from multimodal_jev.slurp_score import load_official_metrics, prepare_predictions, score_native


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def clean_population(requests, targets, clean_metadata, excluded):
    """Prove a pure exclusion subset and unchanged native targets before scoring."""
    req={r['id']:r for r in requests}; gold={r['id']:r for r in targets}
    if len(req)!=len(requests) or len(gold)!=len(targets) or set(req)!=set(gold):
        raise ValueError('unique aligned original population required')
    if len(set(excluded))!=len(excluded) or not set(excluded)<=set(req):
        raise ValueError('unique known exclusions required')
    kept={}; source_ids=set()
    for row in clean_metadata:
        sid=str(row['slurp_id'])
        if sid in source_ids:raise ValueError('duplicate clean source')
        source_ids.add(sid)
        target=native_target(row)
        for recording in row['recordings']:
            uid='slurp:'+recording['file']
            if uid in kept or uid not in gold:raise ValueError('duplicate or unknown clean recording')
            original=gold[uid]
            if (original['target']!=target or str(original['metadata']['slurp_id'])!=sid
                    or original['metadata']['file']!=recording['file']):
                raise ValueError('clean target or source identity changed')
            kept[uid]=target
    if set(kept)!=set(req)-set(excluded):raise ValueError('clean population differs from fixed exclusions')
    return [r for r in requests if r['id'] in kept], [r for r in targets if r['id'] in kept]


def run(root, verified, clean, alignment_sha, output):
    if output.exists():raise FileExistsError(output)
    receipt=read(verified); acceptance=Path(receipt['acceptance_path'])
    if file_hash(acceptance)!=receipt['acceptance_sha256'] or Path(receipt['root'])!=root:
        raise ValueError('accepted run identity changed')
    pins={str(verified):file_hash(verified),str(acceptance):file_hash(acceptance)}
    for p in (root/'run/terminal.json',root/'evaluation-terminal.json'):
        if read(p)['exit_code']!=0:raise ValueError('successful prior run required')
        pins[str(p)]=file_hash(p)
    intent_path=acceptance.parent/'intent.json'; intent=read(intent_path)
    pins[str(intent_path)]=file_hash(intent_path)
    for name,key in [('identity.json','identity_sha256'),('predictions.jsonl','predictions_sha256')]:
        path=root/'run'/name
        if file_hash(path)!=intent[key]:raise ValueError('accepted input/output changed')
        pins[str(path)]=file_hash(path)
    identity=read(root/'run/identity.json'); cfg=identity['config']
    if cfg['route']!='audio-direct' or cfg['parameters_frozen'] is not True or cfg['split']!='test':
        raise ValueError('frozen direct audio TEST required')
    release=Path(cfg['requests']).parent; manifest=read(release/'MANIFEST.json')
    for name,expected in [('test.requests.jsonl',identity['requests_sha256']),
                          ('test.targets.jsonl',manifest['splits']['test']['targets_sha256']),
                          ('taxonomy.json',identity['taxonomy_sha256'])]:
        p=release/name
        if file_hash(p)!=expected:raise ValueError('native data changed')
        pins[str(p)]=expected
    if file_hash(root/'score/result.json')!=receipt['original_score_sha256']:
        raise ValueError('accepted original score changed')
    pins[str(root/'score/result.json')]=receipt['original_score_sha256']
    alignment=clean/'ALIGNMENT-RECEIPT.json'
    if file_hash(alignment)!=alignment_sha:raise ValueError('clean alignment changed')
    aligned=read(alignment); metadata=clean/'metadata/test.jsonl'
    if file_hash(metadata)!=aligned['counts']['test']['metadata_sha256']:
        raise ValueError('clean metadata changed')
    if aligned['source_hashes']['test']!=manifest['identity']['source_sha256']['test']:
        raise ValueError('clean and native source differ')
    excluded=clean/'test.native-excluded-ids.json'
    for p in (alignment,metadata,excluded,release/'MANIFEST.json'):
        pins[str(p)]=file_hash(p)
    requests=rows(release/'test.requests.jsonl');targets=rows(release/'test.targets.jsonl')
    predictions=rows(root/'run/predictions.jsonl');taxonomy=read(release/'taxonomy.json')
    # Validate the whole accepted population before selecting the clean subset.
    _,original_counts=prepare_predictions(requests,targets,predictions,taxonomy)
    clean_requests,clean_targets=clean_population(requests,targets,rows(metadata),read(excluded))
    if len(clean_requests)!=aligned['counts']['test']['retained_unique_recording_files']:
        raise ValueError('clean count differs')
    keep={r['id'] for r in clean_requests};predictions=[p for p in predictions if p['example_id'] in keep]
    repo=Path(read(root/'launch.json')['scorer_repository'])
    metric=load_official_metrics(repo)
    pins[str(repo/'FETCH.json')]=file_hash(repo/'FETCH.json')
    score,normalized=score_native(clean_requests,clean_targets,predictions,taxonomy,metric)
    score.update(schema='slurp-clean-native-rescore-v1',original_examples=len(requests),
        excluded_recordings=len(requests)-len(clean_requests),source_groups=len({r['source_group'] for r in clean_requests}),
        full_official_split=False,full_declared_clean_population=True,split='test',pins=pins,
        implementation_sha256=file_hash(__file__),scoring_implementation_sha256=file_hash(Path(__file__).resolve().parents[2]/'src/multimodal_jev/slurp_score.py'),
        inference_reused=True,new_inference=False,weight_updates=0,
        original_parse_counts=original_counts,
        scope='Frozen historical native composite predictions, clean-v7 exclusion-only TEST rescore. Not semantic-Q truth, new prompt results or an algorithm improvement.',
        optimization_allowed=False)
    output.mkdir(parents=True)
    atomic_json(output/'population.json',dict(ids=[r['id'] for r in clean_requests],excluded_ids=read(excluded)))
    (output/'official-predictions.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in normalized))
    score['population_sha256']=file_hash(output/'population.json')
    score['normalized_predictions_sha256']=file_hash(output/'official-predictions.jsonl')
    atomic_json(output/'result.json',score)
    atomic_json(output/'terminal.json',dict(exit_code=0,result_sha256=file_hash(output/'result.json')))
    print(json.dumps(dict(examples=score['examples'],malformed=score['malformed'],
        joint_intent=score['metrics']['joint_intent']['overall'],slu_f1=score['metrics']['slu_f1']['overall'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','verified','clean','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--alignment-sha256',required=True);a=p.parse_args()
    run(a.root,a.verified,a.clean,a.alignment_sha256,a.output)
