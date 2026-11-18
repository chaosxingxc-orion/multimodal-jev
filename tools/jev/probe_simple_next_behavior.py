#!/usr/bin/env python3
"""Replay accepted updated vLLM behavior on the second full scheduled audio batch.

CPU transport only; neither executes an update nor accepts a continuous trainer.
"""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config):
    import torch
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.simple_next_behavior import next_batch
    from multimodal_jev.span_hybrid_actor import make_batch
    from multimodal_jev.span_policy import joint_logprobs
    from verify_slurp_simple_parity import verify
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only transport required')
    torch.set_num_threads(4)
    files={}
    def read(p):
        files[str(p)]=file_hash(p);return json.loads(Path(p).read_text())
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('transport dependency changed: '+p)
        files[p]=h
    old,new=[Path(config[k]) for k in ('reference_gate','behavior_gate')]
    reference,behavior=[verify(p) for p in (old,new)]
    for r in (reference,behavior):files.update(r['files'])
    if (reference['updates']!=2 or behavior['updates']!=1
            or reference['requests']!=256 or behavior['requests']!=256
            or behavior['sampled_resident_parity'] is None):
        raise ValueError('accepted original and updated full256 gates required')
    infer=read(new/'config.json');prior=read(old/'config.json')
    for k in ('base_id','model','inputs','population','runtime_profile','max_model_len'):
        if infer[k]!=prior[k]:raise ValueError('policy pair input/runtime differs')
    policy=infer['base_id']+':'+infer['adapter_id'];fixed=prior['base_id']+':'+prior['adapter_id']
    state=read(infer['checkpoint_evidence'])
    if state['reference_policy_id']!=fixed:raise ValueError('immutable reference lineage differs')
    train=read(Path(behavior['training_root'])/'config.json')
    schedule=SimpleRLSchedule(Path(train['epoch_file']).parent,
        manifest_sha256=train['schedule_sha256'],release_sha256=train['release_sha256'])
    files.update(schedule.files);slots,cursor=schedule.at(1)
    population,task,_=load(train['release'],train['release_sha256'],'learning',purpose='optimize')
    population_check=schedule.validate_population(population)
    # The already pinned task processor is sufficient; no base weights needed.
    from transformers import Qwen3ASRProcessor
    processor=Qwen3ASRProcessor.from_pretrained(train['model_path'],local_files_only=True)
    items,receipt=next_batch({r['id']:r for r in population},task,slots,processor,
        behavior_result=read(new/'run/vllm/result.json'),reference_result=read(old/'run/vllm/result.json'),
        policy_id=policy,reference_policy_id=fixed,completed_updates=1,seed=train['seed'])
    reports=[]
    for algorithm in ('grpo_v1','rloo_v1'):
        batch=make_batch(items,stage='task',coefficient=cursor['coefficient'],reduction='real_request_mean',
            algorithm=algorithm,kl_coefficient=.01,reference_policy_id=fixed)
        if len(batch)!=128 or any(len(c)!=32 for c in batch.split(32)):
            raise ValueError('complete second-batch DP4 transport required')
        counts=[]
        for item,payload in zip(items,batch.non_tensor_batch['multi_modal_inputs'],strict=True):
            reward=(item['actions'][:,:,0]==item['targets'][:,0,None]).float()
            centered=reward-reward.mean(-1,keepdim=True)
            expected=centered/(reward.std(-1,keepdim=True)+1e-6) if algorithm=='grpo_v1' else centered*16/15
            torch.testing.assert_close(expected,payload['sampled_advantages'],atol=1e-6,rtol=1e-6)
            if not torch.equal(payload['old_joint'],joint_logprobs(item['behavior'],item['actions'])):
                raise ValueError('updated sampled action log probabilities differ')
            counts.append(int(reward.sum()))
        reports.append(dict(algorithm=algorithm,correct_samples=sum(counts),
            all_wrong_groups=sum(n==0 for n in counts),mixed_groups=sum(0<n<16 for n in counts),
            all_correct_groups=sum(n==16 for n in counts)))
    return dict(passed=True,receipt=receipt,reports=reports,files=files,population_check=population_check,cursor=cursor,
        scope='Actual updated vLLM artifact and fixed reference replay into next scheduled native DataProto; no second NPU update or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text())))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
