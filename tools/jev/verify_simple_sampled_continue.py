#!/usr/bin/env python3
"""Independent CPU audit of the second scheduled audio RL update."""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    from safetensors.torch import load_file
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.simple_sampled_checkpoint import reconstruct
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_hybrid_objective import sft_coefficient
    from multimodal_jev.simple_next_behavior import next_batch
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from verify_simple_sampled_restore import run as verify_recovery
    from verify_slurp_simple_parity import verify
    recovery=verify_recovery(config,output,expected_mode='restore_updated_rl_then_second_update')
    files=dict(recovery['files']);root=Path(config['training_root'])
    def read(p):
        files[str(p)]=file_hash(p);return json.loads(Path(p).read_text())
    train=read(root/'config.json');plan=train['sampled_probe']
    gate=verify(plan['behavior_gate']);files.update(gate['files'])
    original=verify(plan['reference_gate']);files.update(original['files'])
    fixed=train['base_id']+':'+original['adapter_id'];behavior=train['base_id']+':'+gate['adapter_id']
    previous=read(Path(gate['training_root'])/'config.json')
    if (previous['sampled_probe']['behavior_gate']!=plan['reference_gate'] or plan['total_rl_updates']!=634
            or original['updates']!=2 or behavior==fixed):raise ValueError('original reference and budget required')
    adapter=root/'run/rl-adapter-step0002';adapter_id,assets=checkpoint_identity(adapter)
    files.update({str(adapter/k):h for k,h in assets.items()})
    commit=read(root/'run/rl-checkpoint-0002-committed.json')
    if (commit['adapter_id']!=adapter_id or commit['world_size']!=4
            or set(commit['files'])!={f'rl-rank-{r}/checkpoint-0002.pt' for r in range(4)}):
        raise ValueError('complete second checkpoint commit required')
    states=[];resident=[];coefficient=sft_coefficient(1,634)
    for rank in range(4):
        out=root/'run'/f'rl-rank-{rank}';path=out/'checkpoint-0002.pt'
        if file_hash(path)!=commit['files'][str(path.relative_to(root/'run'))]:
            raise ValueError('second shard changed')
        files[str(path)]=file_hash(path);state=torch.load(path,map_location='cpu',weights_only=True)
        if state['behavior_policy_id']!=behavior:raise ValueError('second behavior changed')
        states.append(state);r=read(out/'result.json');restored=read(out/'restored.json')
        expected=dict(passed=True,rank=rank,updates=2,algorithm=plan['algorithm'],base_id=train['base_id'],
            adapter_id=adapter_id,reference_policy_id=fixed,behavior_policy_id=behavior,
            coefficient=coefficient,phase='awaiting_acceptance')
        if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('second update report differs')
        if (len(r['execution_dtypes'])!=112
                or any(v!=['torch.bfloat16']*2 for v in r['execution_dtypes'].values())):
            raise ValueError('BF16 second update required')
        norms=r['metrics']['actor/grad_norm']
        if len(norms)!=1 or not 0<float(norms[0])<float('inf'):raise ValueError('invalid gradient')
        admission=dict(global_rows=128,global_real_requests=128,stage='task',coefficient=coefficient,
            reduction='real_request_mean',algorithm=plan['algorithm'],group_size=16)
        if any(r['admission'].get(k)!=v for k,v in admission.items()):raise ValueError('wrong second batch admission')
        if (restored['rank']!=rank or restored['world_size']!=4 or restored['previous_updates']!=2
                or not all(restored[k] is True for k in ('exact_masters','exact_adam','exact_resident_probabilities'))
                or restored['requests']!=32 or len(restored['records'])!=32
                or restored['records']!=read(out/'after.json')['records']):raise ValueError('second reload differs')
        resident.extend(restored['records'])
    source=Path(gate['training_root'])/'run/rl-rank-0/checkpoint-0001.pt'
    names=torch.load(source,map_location='cpu',weights_only=True)['names']
    tensors=reconstruct(states,load_file(str(adapter/'adapter_model.safetensors')),algorithm=plan['algorithm'],
        adapter_id=adapter_id,reference_policy_id=fixed,lr=train['lr'],expected_names=names,completed_updates=2)
    if checkpoint_identity(train['model_path'])[0]!=train['base_id']:raise ValueError('base identity changed')
    processor,base=load_runtime(train['model_path'],'cpu','bfloat16')
    loaded=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
    actual={adapter_key(n):p.detach().cpu() for n,p in loaded.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    if actual.keys()!=tensors.keys() or any(actual[k].dtype!=torch.bfloat16 or not torch.equal(actual[k],v) for k,v in tensors.items()):
        raise ValueError('actual second PEFT reload differs')
    population,task,_=load(train['release'],train['release_sha256'],'learning',purpose='optimize')
    schedule=SimpleRLSchedule(Path(train['epoch_file']).parent,
        manifest_sha256=train['schedule_sha256'],release_sha256=train['release_sha256'])
    files.update(schedule.files);population_check=schedule.validate_population(population)
    slots,cursor=schedule.at(1)
    if cursor['coefficient']!=coefficient:raise ValueError('second CE cursor differs')
    items,receipt=next_batch({r['id']:r for r in population},task,slots,processor,
        behavior_result=read(Path(plan['behavior_gate'])/'run/vllm/result.json'),
        reference_result=read(Path(plan['reference_gate'])/'run/vllm/result.json'),policy_id=behavior,
        reference_policy_id=fixed,completed_updates=1,seed=train['seed'])
    if receipt!=read(root/'run/rl-inputs.json') or [r['request_id'] for r in resident]!=[i['request_id'] for i in items]:
        raise ValueError('independent second batch replay differs')
    for r in resident:
        v=torch.tensor(r['candidate_logprobs'])
        if v.shape!=(1,60) or not torch.isfinite(v).all() or not torch.allclose(v.logsumexp(-1),torch.zeros(1),atol=1e-6,rtol=0):
            raise ValueError('second resident distribution invalid')
    return dict(passed=True,training_root=str(root),updates=2,algorithm=plan['algorithm'],adapter=str(adapter),
        adapter_id=adapter_id,reference_policy_id=fixed,behavior_policy_id=behavior,ranks=4,tensors=112,
        exact_master_export=True,exact_peft_reload=True,durable_commit_verified=True,supervision_records_verified=128,
        exact_pre_update_recovery=True,files=files,population_check=population_check,
        scope='Second scheduled new-behavior RL update with preserved Adam and fixed reference audited independently; updated backend parity and full training remain separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
