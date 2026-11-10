#!/usr/bin/env python3
"""Real verl numerical check with explicitly synthetic SFC sampling records."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_reward_batch import compose


def run(fixture_path,fixture_hash,out):
    if file_hash(fixture_path)!=fixture_hash:raise ValueError('registered synthetic fixture changed')
    fixture=json.loads(fixture_path.read_text())
    if fixture['scope']!='synthetic numerical fixture; no actor samples':raise ValueError('synthetic scope required')
    args=fixture['arguments'];args['prompt_evidence']={(r['example_id'],r['decision_id']):r['evidence'] for r in args['prompt_evidence']}
    class Tokenizer:
        def __len__(self):return 1000
        def decode(self,ids,skip_special_tokens):
            return fixture['token_text'][str(ids[0])] if ids[-1]==999 else 'INVALID'
    args['tokenizer']=Tokenizer()
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    torch.set_num_threads(2)
    reports={};advantages={}
    lengths=[len(r['token_ids']) for r in args['records']];width=max(lengths)+1
    mask=torch.zeros((len(lengths),width))
    for i,n in enumerate(lengths):mask[i,:n]=1
    for mode in ('independent','function-conjunction'):
        batch=compose(**{**args,'mode':mode})
        rewards=torch.zeros_like(mask)
        for i,n in enumerate(lengths):rewards[i,n-1]=batch['rewards'][i]
        for normalized in (False,True):
            adv,_=compute_grpo_outcome_advantage(rewards,mask,np.array(batch['uid'],dtype=object),
                norm_adv_by_std_in_grpo=normalized)
            if not torch.isfinite(adv).all() or not torch.equal(adv[mask==0],torch.zeros_like(adv[mask==0])):
                raise ValueError('finite advantages with zero padding required')
            for uid in set(batch['uid']):
                indices=[i for i,u in enumerate(batch['uid']) if u==uid]
                if abs(float(adv[indices,0].sum()))>1e-5:raise ValueError('same-prompt centered advantages required')
            key=mode+('/standard' if normalized else '/center');advantages[key]=adv
            reports[key]=dict(nonzero_response_advantages=int((adv[:,0]!=0).sum()),
                max_abs_advantage=float(adv.abs().max()),padding_max_abs=0.)
    target=fixture['affected_call_example'];indices=[i for i,r in enumerate(args['records'])
        if r['example_id']==target and r['decision_id']=='f00.calls']
    if [args['records'][i]['replicate'] for i in indices]!=[0,1]:raise ValueError('registered fixture replicate order changed')
    independent=advantages['independent/center'][indices,0]
    related=advantages['function-conjunction/center'][indices,0]
    if not torch.equal(independent,torch.zeros(2)) or not torch.allclose(related,torch.tensor([.1,-.1]),atol=1e-6):
        raise ValueError('correct endpoint must gain contrast only from its aligned peer')
    # All-equal groups must have zero advantage, even when their reward is bad.
    equal=torch.zeros_like(mask)
    for i,n in enumerate(lengths):equal[i,n-1]=-1.
    zero,_=compute_grpo_outcome_advantage(equal,mask,np.array(batch['uid'],dtype=object),norm_adv_by_std_in_grpo=True)
    if torch.count_nonzero(zero):raise ValueError('equal-reward group must not invent advantage')
    result=dict(schema='sfc-jev-verl-reward-numerical-probe-v1',fixture_sha256=fixture_hash,
        versions={k:importlib.metadata.version(k) for k in ('torch','verl','numpy')},
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/probe_sfc_reward_advantages.py','src/multimodal_jev/sfc_reward_batch.py','src/multimodal_jev/sfc_jev_reward.py')},
        reports=reports,affected_call_center_advantages=related.tolist(),equal_reward_advantages_zero=True,
        responses=len(lengths),prompt_groups=batch['prompt_groups'],weight_updates=0,model_loaded=False,
        scope='Real pinned verl core advantage function; synthetic SFC inputs and sampled-token fixtures. '
              'No real tokenizer, audio processor, actor, policy gradient, distributed worker or RL efficacy claim.')
    atomic_json(out/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--fixture-hash',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(a.fixture,a.fixture_hash,a.output),indent=2))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
