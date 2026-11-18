#!/usr/bin/env python3
"""Independent fixed-image CPU audit of a completed one-update NPU RL probe."""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    from safetensors.torch import load_file
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.policy_identity import checkpoint_identity,compare_adapter_artifacts
    from multimodal_jev.simple_sampled_checkpoint import reconstruct
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.simple_audio_batch import compile_batch
    from multimodal_jev.simple_sampled_rollout import actor_batch
    from verify_slurp_simple_parity import verify
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only audit required')
    torch.set_num_threads(4)
    files={}
    def read(p):
        p=Path(p);files[str(p)]=file_hash(p);return json.loads(p.read_text())
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('audit dependency changed: '+p)
        files[p]=h
    root=Path(config['training_root']);training=read(root/'config.json')
    launch=read(root/'launch.json');observed=read(config['training_observed'])
    if (launch['config_sha256']!=file_hash(root/'config.json')
            or observed['container_id']!=launch['container_id'] or observed['image']!=launch['image']
            or observed['state']['Status']!='exited' or observed['state']['ExitCode']!=0
            or read(root/'terminal.json')['exit_code']!=0):
        raise ValueError('exact successful probe terminal required')
    for p,h in training['pins'].items():
        if file_hash(p)!=h:raise ValueError('training dependency changed: '+p)
        files[p]=h
    plan=training['sampled_probe'];gate=verify(plan['behavior_gate']);files.update(gate['files'])
    reference_root=Path(gate['training_root'])
    original_config=read(reference_root/'config.json')
    from multimodal_jev.slurp_instruction_resume import FIELDS
    if any(training[k]!=original_config[k] for k in FIELDS):
        raise ValueError('accepted fixture factors changed')
    if (training.get('task_mode')!='direct_final' or training['updates']!=2
            or training['global_batch']!=128 or training.get('microbatch_per_rank',1)!=1
            or gate['updates']!=2 or gate['requests']!=256):
        raise ValueError('fixed two-step CE fixture and complete native128 RL batch required')
    evidence=compare_adapter_artifacts(reference_root/'run/adapter-step0002',root/'run/adapter-step0002')
    if evidence['reference_id']!=gate['adapter_id']:raise ValueError('reference artifact changed')
    policy=training['base_id']+':'+gate['adapter_id'];adapter=root/'run/rl-adapter-step0001'
    adapter_id,assets=checkpoint_identity(adapter)
    files.update({str(adapter/k):h for k,h in assets.items()})
    for key in ('reference_files','candidate_files'):
        parent=reference_root/'run/adapter-step0002' if key=='reference_files' else root/'run/adapter-step0002'
        files.update({str(parent/k):h for k,h in evidence[key].items()})
    commit=read(root/'run/rl-checkpoint-0001-committed.json')
    expected={f'rl-rank-{r}/checkpoint-0001.pt' for r in range(4)}
    if commit['adapter_id']!=adapter_id or commit['world_size']!=4 or set(commit['files'])!=expected:
        raise ValueError('complete committed RL state required')
    states=[];reference_states=[];reported_ids=[]
    for rank in range(4):
        rel=f'rl-rank-{rank}/checkpoint-0001.pt';p=root/'run'/rel
        if file_hash(p)!=commit['files'][rel]:raise ValueError('committed RL shard changed')
        files[str(p)]=file_hash(p);states.append(torch.load(p,map_location='cpu',weights_only=True))
        original=reference_root/'run'/f'rank-{rank}/checkpoint-0002.pt'
        prefix=root/'run'/f'rank-{rank}/checkpoint-0002.pt'
        for p in (original,prefix):files[str(p)]=file_hash(p)
        ref=torch.load(original,map_location='cpu',weights_only=True)
        compare_state(ref,torch.load(prefix,map_location='cpu',weights_only=True),rank=rank,updates=2)
        reference_states.append(ref)
        out=root/'run'/f'rl-rank-{rank}'
        result=read(out/'result.json');restored=read(out/'restored.json')
        if (read(root/'run'/f'rank-{rank}/terminal.json')['exit_code']!=0
                or result['passed'] is not True or result['rank']!=rank or result['updates']!=1
                or result['algorithm']!=plan['algorithm'] or result['base_id']!=training['base_id']
                or result['adapter_id']!=adapter_id or result['reference_policy_id']!=policy
                or result['phase']!='awaiting_acceptance'):
            raise ValueError('explicit unaccepted one-update report required')
        norms=result['metrics']['actor/grad_norm']
        if len(norms)!=1 or not 0<float(norms[0])<float('inf'):raise ValueError('finite nonzero gradient required')
        admission=result['admission']
        expected_admission=dict(global_rows=128,global_real_requests=128,stage='task',coefficient=1.,
            reduction='real_request_mean',algorithm=plan['algorithm'],group_size=16)
        if any(admission.get(k)!=v for k,v in expected_admission.items()):
            raise ValueError('complete fixed sampled actor admission required')
        if (len(result['execution_dtypes'])!=112
                or any(v!=['torch.bfloat16','torch.bfloat16'] for v in result['execution_dtypes'].values())):
            raise ValueError('all BF16 execution hooks required')
        if (restored['rank']!=rank or restored['world_size']!=4 or restored['previous_updates']!=1
                or restored['exact_masters'] is not True or restored['exact_adam'] is not True
                or restored['exact_resident_probabilities'] is not True or restored['requests']!=32
                or restored['records']!=read(out/'after.json')['records']):
            raise ValueError('complete exact resident reload evidence required')
        for row in restored['records']:
            logp=torch.tensor(row['candidate_logprobs'])
            if (logp.shape!=(1,60) or not torch.isfinite(logp).all()
                    or not torch.allclose(logp.logsumexp(-1),torch.zeros(1),atol=1e-6,rtol=0)):
                raise ValueError('finite normalized full60 probabilities required')
            reported_ids.append(row['request_id'])
    tensors=reconstruct(states,load_file(str(adapter/'adapter_model.safetensors')),
        algorithm=plan['algorithm'],adapter_id=adapter_id,reference_policy_id=policy,
        lr=training['lr'],expected_names=reference_states[0]['names'])
    if checkpoint_identity(training['model_path'])[0]!=training['base_id']:raise ValueError('base changed')
    processor,base=load_runtime(training['model_path'],'cpu','bfloat16')
    loaded=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
    actual={adapter_key(n):p.detach().cpu() for n,p in loaded.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    if actual.keys()!=tensors.keys() or any(actual[k].dtype!=torch.bfloat16 or not torch.equal(actual[k],v) for k,v in tensors.items()):
        raise ValueError('actual PEFT reload differs')
    rows,task,_=load(training['release'],training['release_sha256'],'learning',purpose='optimize')
    epoch=read(training['epoch_file'])
    from multimodal_jev.slurp_training import content_hash
    if content_hash(epoch['batches'])!=epoch['report']['batches_sha256']:
        raise ValueError('complete epoch schedule changed')
    entries,renderer,compiled=compile_batch({r['id']:r for r in rows},task,epoch['batches'][0],processor,
        policy_id=policy,global_batch=128)
    behavior=read(Path(plan['behavior_gate'])/'run/vllm/result.json')
    wanted={r[4] for r in renderer}
    behavior=dict(behavior,records=[r for r in behavior['records'] if r['request_id'] in wanted])
    items,sampling=actor_batch(entries,behavior,behavior,policy_id=policy,reference_policy_id=policy,
        seed=training['seed'],update=0)
    if read(root/'run/rl-inputs.json')!=dict(compiled=compiled,sampling=sampling):
        raise ValueError('independent scheduled audio and sampling replay differ')
    if reported_ids!=[item['request_id'] for item in items]:raise ValueError('resident request coverage differs')
    # Recompute signal coverage from independently compiled targets and sampled
    # actions. Constant groups remain in the denominator; CE is not RL signal.
    counts=[int((item['actions'][0,:,0]==item['targets'][0,0]).sum()) for item in items]
    signal=dict(requests=len(items),samples=16*len(items),correct_samples=sum(counts),
        all_wrong_groups=sum(n==0 for n in counts),all_correct_groups=sum(n==16 for n in counts),
        mixed_reward_groups=sum(0<n<16 for n in counts),
        mixed_group_request_weight=sum(item['request_weight'] for item,n in zip(items,counts,strict=True) if 0<n<16),
        real_request_weight=sum(item['request_weight'] for item in items),
        scope='Actual complete G16 sampling on the fixed learning engineering batch; no selection efficacy or isolated RL gradient contribution.')
    return dict(passed=True,training_root=str(root),updates=1,algorithm=plan['algorithm'],
        adapter=str(adapter),adapter_id=adapter_id,reference_policy_id=policy,
        exact_master_export=True,exact_peft_reload=True,durable_commit_verified=True,
        ranks=4,tensors=112,supervision_records_verified=128,sampling_signal=signal,files=files,
        scope='Independent committed four-rank FP32/Adam layout, exact BF16 reconstruction and actual CPU PEFT reload; scheduled audio/sampling replay and recorded resident recovery checked. Fresh HF/vLLM and fresh-process recovery remain separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
