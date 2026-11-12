#!/usr/bin/env python3
"""Two real audio updates in one FSDP actor; accepted historical behaviors.

Replay only. No resident vLLM, new learning arm or complete trainer claim.
"""
import argparse
import os
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read
from train_intent_finite_npu import run


def validate(plan):
    from multimodal_jev.intent_resume import validate_lineage
    from multimodal_jev.intent_objective import identity
    from queue_span_dp4_resume import check_pins
    if plan['schema']!='intent-resident-two-update-replay-v1' or len(plan['references'])!=2:
        raise ValueError('fixed two-update replay plan required')
    check_pins(plan['pins']);configs=[]
    for step,entry in enumerate(plan['references']):
        root=Path(entry['training']);suite=Path(entry['suite'])
        for p in (root/'config.json',root/'launch.json',suite/'verified.json',suite/'ACCEPTED.json'):
            if plan['pins'].get(str(p))!=file_hash(p):raise ValueError('pinned accepted references required')
        c=read(root/'config.json');v=read(suite/'verified.json')
        if (c['arm']!='D' or c['step']!=step or c['global_requests']!=8 or c['seed']!=317
                or c['lr']!=1e-4 or identity(c)!=identity({'arm':'D'})
                or v['root']!=str(root) or v['updates']!=step+1 or v['arm']!='D'):
            raise ValueError('original D first/second updates required')
        check_pins(v['files'])
        for path,h in v['files'].items():
            if plan['pins'].get(path)!=h:raise ValueError('complete accepted state must be pinned')
        configs.append(c)
    validate_lineage(*configs)
    return configs


def resident_second(plan,configs,**ctx):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from safetensors.torch import load_file,save_file
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.intent_actor_inputs import prepare
    from multimodal_jev.intent_objective import apply
    from multimodal_jev.span_finite_actor import make_global_batch,admit_batch
    from multimodal_jev.span_training_session import SpanTrainingSession
    from queue_intent_finite_npu import verify_final,verify_gate
    rank=ctx['rank'];root=ctx['root'];first_out=ctx['out'];model=ctx['model'];opt=ctx['optimizer'];actor=ctx['actor']
    out=root/f'resident-rank-{rank}';out.mkdir(exist_ok=False)
    refs=[Path(e['training']) for e in plan['references']];suites=[Path(e['suite']) for e in plan['references']]
    def gather(value):
        peers=[None]*4;dist.all_gather_object(peers,value);return peers
    def collective(name,operation):
        result=None;error=None
        try:result=operation()
        except Exception as exc:error=repr(exc)
        errors=gather(error)
        if any(errors):raise ValueError(name+': '+str(errors))
        return result
    def state(updates):
        return dict(world_size=4,rank=rank,updates=updates,
            names=[n for n,p in model.named_parameters() if p.requires_grad],
            parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
            optimizer=opt.state_dict())
    def verify_first():
        a=torch.load(refs[0]/'run'/f'rank-{rank}'/'optimizer-local.pt',map_location='cpu',weights_only=True)
        audit=compare_state(a,state(1),rank=rank,updates=1)
        for name in ('before.json','after.json'):
            if read(first_out/name)!=read(refs[0]/'run'/f'rank-{rank}'/name):raise ValueError('first update probabilities differ')
        if ctx['adapter_id']!=checkpoint_identity(refs[0]/'run/adapter')[0]:raise ValueError('first export differs')
        return audit
    atomic_json(out/'first-exact.json',collective('first_update',verify_first))
    def verify_reference():
        if rank!=0:return None
        for suite in suites:verify_final(suite)
        return verify_gate(suites[0],configs[0]['pilot'],configs[0]['manifest_sha256'])
    collective('accepted_reference',verify_reference)
    # Model/optimizer objects and FP32 values stay resident. No restore/load here.
    actor.config.jev_policy_id=ctx['base_id']+':'+ctx['adapter_id']
    session=SpanTrainingSession(actor,policy_id=actor.config.jev_policy_id,completed_updates=1,gather=gather)
    identities=([id(p) for p in model.parameters()],id(opt),id(actor));prepared=None
    def build():
        nonlocal prepared
        if rank==0:
            items,selection=prepare(configs[1],ctx['processor'],session.policy_id,1)
            items,objective=apply(configs[1],items);prepared=list(make_global_batch(items).split(2))
            atomic_json(root/'resident-selection.json',selection);atomic_json(root/'resident-objective.json',objective)
    collective('prepare_second',build);incoming=[None];dist.scatter_object_list(incoming,prepared,src=0);batch=incoming[0]
    admission=admit_batch(batch,actor.config);old=batch.batch['old_log_probs'].clone()
    computed=actor.compute_log_prob(batch)['log_probs'].detach().cpu()
    before=dict(old=old.tolist(),actor=computed.tolist(),admission=admission)
    atomic_json(out/'before.json',before)
    def check_before():
        if before!=read(refs[1]/'run'/f'rank-{rank}'/'before.json'):raise ValueError('resident second pre-update policy differs')
    collective('second_before',check_before)
    metrics=session.update(batch)
    if not torch.equal(old,batch.batch['old_log_probs']):raise ValueError('behavior replaced')
    current=state(2);torch.save(current,out/'optimizer-local.pt')
    def verify_second():
        if identities!=([id(p) for p in model.parameters()],id(opt),id(actor)):raise ValueError('resident objects replaced')
        return compare_state(torch.load(refs[1]/'run'/f'rank-{rank}'/'optimizer-local.pt',map_location='cpu',weights_only=True),current,rank=rank,updates=2)
    atomic_json(out/'second-exact.json',collective('second_state',verify_second))
    with FSDP.summon_full_params(model,writeback=False,rank0_only=False):
        tensors={n.replace('_fsdp_wrapped_module.','').replace('.default.','.'):
            p.detach().to(device='cpu',dtype=torch.bfloat16).clone()
            for n,p in ctx['adapted'].named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    expected=load_file(str(refs[1]/'run/adapter/adapter_model.safetensors'))
    def check_export():
        if tensors.keys()!=expected.keys() or any(not torch.equal(p,expected[n]) for n,p in tensors.items()):raise ValueError('second BF16 export differs')
    collective('second_export',check_export)
    if rank==0:
        import shutil
        dest=root/'resident-adapter';dest.mkdir();save_file(tensors,str(dest/'adapter_model.safetensors'))
        (dest/'adapter_model.safetensors').chmod(0o644)
        shutil.copyfile(refs[1]/'run/adapter/adapter_config.json',dest/'adapter_config.json')
    dist.barrier();pid=ctx['base_id']+':'+checkpoint_identity(root/'resident-adapter')[0];session.publish(pid)
    # Remains awaiting independent real backend gate, not accepted automatically.
    atomic_json(out/'result.json',dict(passed=True,rank=rank,updates=2,policy_id=pid,phase=session.phase,
        resident_objects_preserved=True,metrics=metrics,
        scope='Two resident actor updates replay accepted historical behaviors exactly. New export awaits independent gate; no resident rollout or full trainer proof.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank:raise ValueError('single node DP4 required')
    out=a.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        plan=read(a.config);configs=validate(plan)
        run(configs[0],a.output,out,rank,after_update=lambda **ctx:resident_second(plan,configs,**ctx))
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
