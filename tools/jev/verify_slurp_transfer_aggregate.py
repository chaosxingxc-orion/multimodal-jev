#!/usr/bin/env python3
"""Synthetic aggregate math fixture using the pinned official SLURP scorer.

This deliberately replaces the successful-run loader with synthetic predictions;
it verifies aggregate scoring, not container admission or model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_full import build,expand_bundle
from multimodal_jev.slurp_jev_shards import plan_shards,load_shard
from multimodal_jev.slurp_score import load_official_metrics,score_native
import compare_omni_slurp_transfer as comparison


def verify(output,repository):
    output.mkdir(parents=True,exist_ok=False);native=output/'native';native.mkdir()
    taxonomy=dict(scenario_action_pairs=[['email','send'],['music','play']],entity_types=['person','artist_name'])
    (native/'taxonomy.json').write_bytes(canonical(taxonomy))
    manifest=dict(identity={'taxonomy_sha256':file_hash(native/'taxonomy.json')},splits={})
    for split in ('train','valid','test'):
        requests=[];targets=[]
        for i in range(3):
            uid=f'{split}-{i}';request=dict(id=uid,split=split,source_group=uid,audio_sha256=uid,
                input={'audio':'/'+uid,'schema_sha256':manifest['identity']['taxonomy_sha256']})
            target=dict(id=uid,target=dict(scenario='email',action='send',entities=[dict(type='person',filler='amy')]*2),
                metadata={'file':uid+'.flac'},request_sha256=hashlib.sha256(canonical(request)).hexdigest())
            requests.append(request);targets.append(target)
        for kind,rows in [('requests',requests),('targets',targets)]:
            (native/f'{split}.{kind}.jsonl').write_bytes(b''.join(map(canonical,rows)))
        manifest['splits'][split]=dict(recordings=3,**{kind+'_sha256':file_hash(native/f'{split}.{kind}.jsonl') for kind in ('requests','targets')})
    atomic_json(native/'MANIFEST.json',manifest)
    release,_=build(native,output/'releases',output/'build-state')
    plan=plan_shards(release,'test',2);plan_path=output/'plan.json';atomic_json(plan_path,plan)
    protocol_path=output/'protocol.json';atomic_json(protocol_path,dict(shard_plan=str(plan_path),
        shard_plan_sha256=file_hash(plan_path),prompts={'parent':'synthetic-parent','p1':'synthetic-candidate'}))
    targets=comparison.unique_rows(release/'test.targets.jsonl');runs={k:{} for k in ('parent','p1')}
    memory={};parent_shard_f1=[];factory=load_official_metrics(repository)
    native_rows=comparison.unique_rows(native/'test.requests.jsonl');native_targets=comparison.unique_rows(native/'test.targets.jsonl')
    for shard in plan['shards']:
        bundles,bank=load_shard(plan,shard['id'])
        for key in runs:
            run=output/key/shard['id']/'run';run.mkdir(parents=True);runs[key][shard['id']]=run
            atomic_json(run.parent/'evaluation-terminal.json',{'exit_code':0})
            atomic_json(run/'result.json',{'synthetic_fixture':True})
            predictions=[];reconstructed=[]
            for bundle in bundles:
                uid=bundle['id'];values=dict(targets[uid]['values'])
                if key=='parent' and uid=='test-2':values['entity.person.values']=[]
                for row in expand_bundle(bundle,bank):
                    predictions.append(dict(example_id=row['id'],replicate=0,
                        request_sha256=hashlib.sha256(canonical(row)).hexdigest(),
                        raw_text=json.dumps({'value':values[row['id'].rsplit('::',1)[-1]]})))
                value=dict(native_targets[uid]['target'])
                if key=='parent' and uid=='test-2':value['entities']=[]
                reconstructed.append(dict(example_id=uid,replicate=0,
                    request_sha256=native_targets[uid]['request_sha256'],raw_text=json.dumps(value)))
            memory[(key,shard['id'])]=predictions
            if key=='parent':
                scored,_=score_native([native_rows[b['id']] for b in bundles],list(native_targets.values()),reconstructed,taxonomy,factory)
                parent_shard_f1.append(scored['metrics']['entity_exact']['overall'][2])
    original=comparison.load
    def synthetic_load(run,protocol,prompt,shard):
        return dict(runner_sha256='fixture',source_sha256={},versions={},model_files=[],population_selector_sha256='fixture'),memory[(prompt,shard)],{'length_limited':0}
    comparison.load=synthetic_load
    try:report=comparison.compare(protocol_path,runs,repository)
    finally:comparison.load=original
    parent=report['prompts']['parent']['official_native_reconstruction']['metrics']['entity_exact']['overall'][2]
    candidate=report['prompts']['p1']['official_native_reconstruction']['metrics']['entity_exact']['overall'][2]
    averaged=sum(parent_shard_f1)/len(parent_shard_f1)
    assert abs(parent-.8)<1e-12 and candidate==1. and averaged==.5
    assert report['paired_to_parent']['p1']['by_task']['entity.person.values']['fixed']==1
    assert report['paired_to_parent']['p1']['all_tasks_by_recording']['fixed']==1
    atomic_json(output/'aggregate.json',report)
    receipt=dict(synthetic_recordings=3,shards=2,parent_global_entity_f1=parent,
        candidate_global_entity_f1=candidate,incorrect_mean_of_shard_f1=averaged,
        aggregate_sha256=file_hash(output/'aggregate.json'),scorer_manifest_sha256=file_hash(repository/'FETCH.json'),
        scope='Synthetic aggregate math with official scorer; run loader replaced explicitly; no model, TEST data or container admission verification')
    atomic_json(output/'result.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repository',type=Path,required=True);a=p.parse_args()
    print(json.dumps(verify(a.output,a.repository),indent=2))
