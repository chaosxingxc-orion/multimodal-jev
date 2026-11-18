#!/usr/bin/env python3
"""Independent exact state and full128 probability audit of continuous recovery."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import file_hash,atomic_json


def run(config,output):
    import torch
    from multimodal_jev.simple_continuous_resume import authenticate
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.simple_resident_acceptance import compare_updated_batch
    from multimodal_jev.resident_service_admission import IMAGE
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('independent CPU audit required')
    files={}
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('audit dependency changed')
        files[path]=digest
    def read(path):
        path=Path(path);files[str(path)]=file_hash(path);return json.loads(path.read_text())
    root=Path(config['training_root']);train=read(root/'config.json')
    launch=read(root/'launch.json');observed=read(config['training_observed'])
    if (observed['container_id']!=launch['container_id'] or observed['image']!=IMAGE
            or launch['image']!=IMAGE or observed['state']['Status']!='exited'
            or observed['state']['ExitCode']!=0 or read(root/'terminal.json')['exit_code']!=0
            or launch['config_sha256']!=file_hash(root/'config.json')):
        raise ValueError('exact successful recovery process required')
    accepted=authenticate(train);files.update(accepted['files']);updates=accepted['completed_updates']
    source=Path(train['continuous_resume']['training_root'])/'run'
    sampling=read(source/f'inputs-{updates-1:04d}.json')['sampling']
    rollout=read(source/f'rollout-after-{updates:04d}.json')
    ranks=[];reports=[];states=[]
    for rank in range(4):
        out=root/'run'/f'rank-{rank}'
        if read(out/'terminal.json')['exit_code']!=0:raise ValueError('all recovery ranks complete required')
        report=read(out/'result.json');reports.append(report)
        if (report['passed'] is not True or report['rank']!=rank or report['updates']!=updates
                or report['policy_id']!=accepted['policy_id']
                or report['reference_policy_id']!=accepted['reference_policy_id']
                or report['schedule_sha256']!=train['schedule_sha256']
                or report['checkpoint_id']!=accepted['checkpoint_id']
                or len(report['execution_dtypes'])!=112
                or any(v!=['torch.bfloat16','torch.bfloat16'] for v in report['execution_dtypes'].values())):
            raise ValueError('complete BF16 recovered rank and lineage required')
        path=out/f'checkpoint-{updates:04d}.pt';files[str(path)]=file_hash(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        exact=compare_state(accepted['states'][rank],state,rank=rank,updates=updates,continuous=True)
        if report['exact_state']!=exact:raise ValueError('reported exact state differs')
        states.append(exact);ranks.append(read(out/'probabilities.json'))
    parity=compare_updated_batch(ranks,rollout,sampling,policy_id=accepted['policy_id'],updates=updates)
    if any(report['parity']!=parity for report in reports):raise ValueError('reported recovery parity differs')
    return dict(passed=True,updates=updates,ranks=4,requests=128,policy_id=accepted['policy_id'],
        exact_master_adam=True,full_candidate_parity=True,state_checks=states,parity=parity,files=files,
        scope='Independent fresh-process continuous DP4 FP32/Adam and full128 recorded-vLLM probability recovery; no optimizer update or task efficacy.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
