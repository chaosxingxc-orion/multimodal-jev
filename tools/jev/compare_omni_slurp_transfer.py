#!/usr/bin/env python3
"""Recompute full transfer results from all successful raw-output shards."""
import argparse
from collections import Counter,defaultdict
import hashlib
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_slurp_transfer_contract import select_transfer_shard
from multimodal_jev.slurp_jev_comparison import PairedAccumulator,require_full_roster
from multimodal_jev.slurp_jev_shards import load_shard,plan_shards
from multimodal_jev.slurp_jev_score import score_bundle,summarize
from multimodal_jev.slurp_score import score_native,load_official_metrics


def load(run,protocol_path,prompt,shard):
    protocol=json.loads(protocol_path.read_text());identity=json.loads((run/'identity.json').read_text())
    cfg=identity['config'];result=json.loads((run/'result.json').read_text())
    launch=json.loads((run.parent/'launch.json').read_text())
    terminal=json.loads((run.parent/'docker-terminal.json').read_text())
    if (terminal['container_id']!=launch['container_id'] or launch['image']!=protocol['image']
            or launch['physical_devices'] not in ([4,5],[6,7])
            or terminal['state']['Status']!='exited' or terminal['state']['ExitCode']!=0
            or json.loads((run/'terminal.json').read_text())['exit_code']!=0
            or json.loads((run.parent/'evaluation-terminal.json').read_text())['exit_code']!=0
            or result['status']!='passed' or result['weight_updates']!=0
            or result['model_files_unchanged'] is not True or result['audio_condition']!='actual'):
        raise ValueError('complete successful frozen transfer container required')
    if (cfg['shard_id']!=shard or cfg['prompt_id']!=prompt
            or cfg['transfer_protocol_sha256']!=file_hash(protocol_path)
            or identity['transfer_protocol_sha256']!=cfg['transfer_protocol_sha256']
            or identity['instruction_sha256']!=protocol['prompts'][prompt]
            or identity['requests_sha256']!=file_hash(cfg['requests'])
            or identity['asset_audit_sha256']!=protocol['asset_audit_sha256']
            or identity['versions']!=protocol['versions'] or identity['model_files']!=protocol['model_files']
            or identity['population_selector_sha256']!=file_hash(inspect.getfile(select_transfer_shard))
            or result['predictions_sha256']!=file_hash(run/'predictions.jsonl')):
        raise ValueError('transfer input or output identity changed')
    rows=select_transfer_shard(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    ids=[r['id'] for r in rows];found={p['example_id']:p for p in predictions}
    if (json.loads((run/'selection.json').read_text())['ids']!=ids
            or len(predictions)!=len(ids) or len(found)!=len(ids) or set(found)!=set(ids)
            or result['requests']!=len(ids) or result['predictions']!=len(ids)
            or any(p['split']!='test' or p['audio_condition']!='actual'
                   or type(p['replicate']) is not int or p['replicate']!=0 for p in predictions)):
        raise ValueError('complete unique transfer predictions required')
    if result['length_limited']!=sum(p['length_limited'] for p in predictions):
        raise ValueError('output truncation count changed')
    return identity,predictions,result


def unique_rows(path):
    rows={}
    for line in path.open():
        row=json.loads(line)
        if row['id'] in rows:raise ValueError('duplicate source or target ID')
        rows[row['id']]=row
    return rows


def compare(protocol_path,runs,repository):
    protocol=json.loads(protocol_path.read_text());plan_path=Path(protocol['shard_plan'])
    if file_hash(plan_path)!=protocol['shard_plan_sha256']:raise ValueError('shard plan changed')
    plan=json.loads(plan_path.read_text());require_full_roster(plan,protocol['prompts'],runs)
    if plan_shards(plan['release'],'test',plan['recordings_per_shard_target'])!=plan:
        raise ValueError('whole TEST plan must reproduce')
    # Check all terminal states before accessing TEST labels.
    for entries in runs.values():
        for run in entries.values():
            if json.loads((run.parent/'evaluation-terminal.json').read_text())['exit_code']!=0:
                raise ValueError('every shard must finish successfully')
    release=Path(plan['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    target_path=release/'test.targets.jsonl'
    if file_hash(target_path)!=manifest['splits']['test']['targets_sha256']:raise ValueError('Jev targets changed')
    targets=unique_rows(target_path)
    native=Path(manifest['identity']['native_release']);native_manifest=json.loads((native/'MANIFEST.json').read_text())
    if file_hash(native/'MANIFEST.json')!=manifest['identity']['native_manifest_sha256']:
        raise ValueError('native manifest changed')
    for kind in ('requests','targets'):
        if file_hash(native/f'test.{kind}.jsonl')!=native_manifest['splits']['test'][kind+'_sha256']:
            raise ValueError('native inputs changed')
    native_rows=unique_rows(native/'test.requests.jsonl');native_targets=unique_rows(native/'test.targets.jsonl')
    taxonomy_path=native/'taxonomy.json'
    if file_hash(taxonomy_path)!=native_manifest['identity']['taxonomy_sha256']:raise ValueError('taxonomy changed')
    taxonomy=json.loads(taxonomy_path.read_text())
    factory=load_official_metrics(repository)
    totals={k:Counter() for k in runs};native_predictions={k:[] for k in runs}
    paired={k:PairedAccumulator() for k in runs if k!='parent'}
    receipts={k:[] for k in runs};truncation=Counter();reference=None;seen=set()
    for shard in plan['shards']:
        sid=shard['id'];bundles,bank=load_shard(plan,sid);by_prompt={}
        for key in runs:
            identity,predictions,result=load(runs[key][sid],protocol_path,key,sid)
            signature={k:identity[k] for k in ('runner_sha256','source_sha256','versions','model_files','population_selector_sha256')}
            if reference is None:reference=signature
            elif signature!=reference:raise ValueError('transfer implementation differs across prompts or shards')
            grouped=defaultdict(list)
            for row in predictions:grouped[row['example_id'].rsplit('::',1)[0]].append(row)
            by_prompt[key]=grouped;truncation[key]+=result['length_limited']
            receipts[key].append(dict(shard_id=sid,result_sha256=file_hash(runs[key][sid]/'result.json')))
        for bundle in bundles:
            uid=bundle['id']
            if uid in seen:raise ValueError('duplicate aggregate recording')
            seen.add(uid);target=targets[uid]
            if hashlib.sha256(canonical(native_targets[uid])).hexdigest()!=target['native_target_sha256']:
                raise ValueError('native target binding changed')
            for key in runs:
                counts,reconstructed=score_bundle(bundle,target,bank,by_prompt[key][uid]);totals[key].update(counts)
                native_predictions[key].append(dict(example_id=uid,replicate=0,
                    request_sha256=hashlib.sha256(canonical(native_rows[uid])).hexdigest(),
                    raw_text=json.dumps(reconstructed) if reconstructed is not None else 'invalid task bundle'))
                if key!='parent':paired[key].add(bundle,target,bank,by_prompt['parent'][uid],by_prompt[key][uid])
    if seen!=set(native_rows) or seen!=set(targets) or len(seen)!=plan['recordings']:
        raise ValueError('full native TEST population required')
    reports={}
    for key in runs:
        official,_=score_native(list(native_rows.values()),list(native_targets.values()),native_predictions[key],taxonomy,factory)
        if totals[key]['decisions']!=plan['decisions']:raise ValueError('full decision coverage required')
        reports[key]=dict(jev=summarize(totals[key]),official_native_reconstruction=official,
            length_limited=truncation[key],shards=receipts[key])
    return dict(schema='jev-omni-full-slurp-transfer-comparison-v1',protocol_sha256=file_hash(protocol_path),
        full_official_split=True,recordings=len(seen),requests_per_prompt=plan['decisions'],
        prompts=reports,paired_to_parent={k:v.report() for k,v in paired.items()},
        targets_sha256=file_hash(target_path),scorer_manifest_sha256=file_hash(repository/'FETCH.json'),
        weight_updates=0,scope='Full frozen audio-direct transfer; official metrics recomputed globally, never averaged across shard F1s')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('protocol','runs','repository','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();mapping=json.loads(a.runs.read_text())
    result=compare(a.protocol,{k:{s:Path(r) for s,r in v.items()} for k,v in mapping.items()},a.repository)
    with a.output.open('xb') as handle:handle.write(canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('prompts','paired_to_parent')},indent=2))
