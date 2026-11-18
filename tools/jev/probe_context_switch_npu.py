"""Actual audio resident A->B->A against independent fixed-policy HF outputs."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import Qwen3ASRProcessor
from multimodal_jev.audio_policy import load_audio
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import encode_context_request
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_request_identity import tensor_identity
from context_switch_rollout import SwitchingContextRollout


def run(c,out):
    torch.set_num_threads(2)
    for p,h in c['pins'].items():
        if file_hash(p)!=h:raise ValueError('source changed: '+p)
    policies=[]
    for name in ('a','b'):
        path=out.parent/name/'policy';manifest=json.loads((path/'MANIFEST.json').read_text())
        policies.append(dict(snapshot=str(path),policy_id=manifest['policy_id'],contract=manifest['contract'],
                             adapter=str(out.parent/name/'adapter')))
    assert policies[0]['policy_id']!=policies[1]['policy_id']
    root=Path(policies[0]['snapshot'])
    processor=Qwen3ASRProcessor.from_pretrained(root/'processor',local_files_only=True)
    task=json.loads(Path(c['tasks']).read_text())['intent']
    candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    inputs=[]
    for row in c['audio']:
        assert file_hash(row['audio'])==row['sha256']
        request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        x,p=encode_context_request(processor,request,load_audio(row['audio']),decision_token_id=151706,
                                   audio_token_id=c['audio_token_id'],max_tokens=4096)
        x={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in x.items()}
        inputs.append((row,x,p))
    refs=[json.loads((out.parent/name/'hf.json').read_text()) for name in ('a','b')]
    config=dict(model=c['model'],base_id=c['base_id'],physical_devices=c['physical_devices'],
                max_model_len=4096,max_decisions=60,tensor_parallel_size=1,**policies[0])
    if 'kv_cache_memory_bytes' in c:
        config['kv_cache_memory_bytes']=c['kv_cache_memory_bytes']
    engine=SwitchingContextRollout(config);records=[];switches=[]
    try:
        for stage,which in enumerate((0,1,0)):
            if stage:
                start=time.monotonic();p=policies[which]
                receipt=engine.switch(snapshot=p['snapshot'],policy_id=p['policy_id'],adapter=p['adapter'])
                switches.append(dict(stage=stage,seconds=time.monotonic()-start,receipt=receipt))
            for (asset,x,positions),ref in zip(inputs,refs[which],strict=True):
                assert ref['id']==asset['id'] and ref['inputs']=={k:tensor_identity(v) for k,v in x.items()}
                assert ref['positions']==positions.tolist()
                value=engine.score(x,positions,audio_token_id=c['audio_token_id'])
                assert value['request_identity']['inputs']==ref['inputs']
                assert value['request_identity']['positions']==ref['positions'][0]
                error=float((torch.tensor(value['prob'])-torch.tensor(ref['prob'])).abs().max())
                records.append(dict(stage=stage,id=asset['id'],probability_max_error=error,**value))
                atomic_json(out/'progress.json',dict(stage=stage,records=len(records)))
                if error!=0:raise ValueError('switched actual audio probability differs')
        assert any(a['prob']!=b['prob'] for a,b in zip(refs[0],refs[1],strict=True))
        atomic_json(out/'result.json',dict(passed=True,records=records,switches=switches,
            queries_per_policy=len(inputs),policy_change_observable=True,
            scope='Same-engine A/B/A audio parity; not continuous actor updates or task efficacy'))
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    run(json.loads(a.config.read_text()),a.output)
