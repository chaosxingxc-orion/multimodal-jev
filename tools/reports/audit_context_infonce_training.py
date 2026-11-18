"""Independent collected-artifact audit for pure InfoNCE training windows.

No training dispatch, no RL/rollout assumptions, no NPU replay or efficacy claim.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.context_training_checkpoint import restore
from multimodal_jev.context_optimizer_audit import audit_adam_shards
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
from multimodal_jev.context_warmup import learning_rate
from multimodal_jev.local_config import load_config
from tools.reports.context_master_export_audit import audit as audit_masters


def read(path):
    return json.loads(Path(path).read_text())


def expected_contract(c):
    plan = derive(c['batch_config'])
    result = {k: c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
                               'release_sha256','schedule_sha256','global_batch','microbatch','stage')}
    result.update(optimizer=dict(lr=c['lr_schedule']['peak_lr'],betas=[.9,.999],eps=1e-8,weight_decay=0.),
                  objective=c['instruction_objective'],lr_schedule=c['lr_schedule'],grad_clip=1.,
                  world_size=plan['dp_replicas'],batch_config=c['batch_config'],
                  parent_schedule_sha256=c['parent_schedule_sha256'],
                  schedule_release_sha256=c['schedule_release_sha256'])
    if c.get('experiment_kind')=='infonce_tail_admission':
        from tools.reports.context_infonce_tail import contract
        result['tail_admission']=contract(c)
    return result


def restore_shard(state, c, *, rank, step):
    plan = derive(c['batch_config'])
    named = [(n,torch.nn.Parameter(torch.zeros_like(state['parameters'][n]))) for n in state['names']]
    scope = validate_trainable_scope(named)
    opt = torch.optim.AdamW([p for _,p in named],
        lr=learning_rate(c['lr_schedule'],step-1,plan['total_updates']),
        betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    recovery = restore(named,opt,state,rank=rank,world_size=plan['dp_replicas'],
                       completed_updates=step,contract=expected_contract(c))
    return dict(recovery=recovery,scope=scope,optimizer_audit=audit_adam_shards(named,opt,step))


def check_batch(batch, planned, population, task, *, policy, width, world):
    from multimodal_jev.context_candidate_task import compile_candidates,bind_candidate_targets
    groups = batch['rank_indices']
    if (batch['policy_id']!=policy or len(batch['records'])!=width
            or len(groups)!=world or any(len(g)!=width//world for g in groups)
            or sorted(i for g in groups for i in g)!=list(range(width))):
        raise ValueError('complete unique DP slot permutation required')
    candidates=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]
    for i,(slot,record) in enumerate(zip(planned['slots'],batch['records'],strict=True)):
        source=population[slot['id']]
        request=compile_candidates(source['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        target=bind_candidate_targets(request,semantics='exclusive_event',exclusive_event=source['intent'])
        uid=f'{source["id"]}::epoch{planned["epoch"]}-batch{planned["index"]}-slot{i}'
        if (record['id']!=slot['id'] or record['request_id']!=uid
                or record['weight']!=slot['weight'] or record['target']!=target):
            raise ValueError('TRAIN target, order or zero padding differs')
    return planned['real_queries']


def checkpoint_steps(c):
    start,stop,interval=c['start_update'],c['stop_update'],c['checkpoint_interval']
    if (any(type(v) is not int for v in (start,stop,interval))
            or not 0<=start<stop or interval<1):
        raise ValueError('valid checkpoint window required')
    steps=[s for s in range(start+1,stop+1) if s%interval==0 or s==stop]
    if c.get('experiment_kind')=='formal_infonce_source':
        plan=derive(c['batch_config'])
        if (start!=0 or stop!=plan['total_updates'] or c.get('resume_root')
                or c.get('engineering_prefix') is not False
                or c.get('training_dispatch_allowed') is not True
                or c.get('evaluation_steps')!=[0,*steps] or c.get('reference_endpoint')!=stop):
            raise ValueError('complete formal budget and matching registered checkpoints required')
    return steps


def check_artifact_inventory(run, c, world):
    run=Path(run);steps=checkpoint_steps(c)
    patterns={
        'checkpoint-*-committed.json':{f'checkpoint-{s:04d}-committed.json' for s in steps},
        'execution-*.pt':{f'execution-{s:04d}.pt' for s in steps},
        'policy-*':{f'policy-{s:04d}' for s in steps},
        'batch-*.json':{f'batch-{s:04d}.json' for s in range(c['start_update'],c['stop_update'])},
    }
    for pattern,expected in patterns.items():
        if {p.name for p in run.glob(pattern)}!=expected:
            raise ValueError('missing or unexpected checkpoint/update artifact: '+pattern)
    for rank in range(world):
        directory=run/f'rank-{rank}'
        if ({p.name for p in directory.glob('checkpoint-*.pt')}!=
                {f'checkpoint-{s:04d}.pt' for s in steps}
                or {p.name for p in directory.glob('update-*.json')}!=
                {f'update-{s:04d}.json' for s in range(c['start_update']+1,c['stop_update']+1)}):
            raise ValueError('missing or unexpected rank checkpoint/update artifact')
    return set(steps)


def check_update(record, c, *, step, real, checkpoint, scope):
    plan=derive(c['batch_config'])
    if (record['update']!=step or record['real_queries']!=real
            or record['checkpoint'] is not checkpoint
            or record['learning_rate']!=learning_rate(c['lr_schedule'],step-1,plan['total_updates'])
            or record['input_service']['cache_hits']!=plan['global_batch']):
        raise ValueError('actual LR, denominator, cache or checkpoint cadence differs')
    for key in ('actor_seconds','total_seconds','preparation_seconds'):
        if type(record[key]) not in (int,float) or not math.isfinite(record[key]) or record[key]<=0:
            raise ValueError('positive measured durations required')
    audit=record['optimizer_audit']
    if scope['base_model_saved'] is not False or len(audit)!=scope['tensors']:
        raise ValueError('LoRA/head optimizer scope differs')
    counts=dict(lora=0,head=0)
    for name,state in audit.items():
        key='.'.join(p for p in name.split('.') if p not in ('_fsdp_wrapped_module','module'))
        kind='head' if key.startswith('decision_head.') else 'lora' if (
            key.startswith('backbone.') and ('.lora_A.' in key or '.lora_B.' in key)) else None
        if kind is None or type(state['numel']) is not int or state['numel']<0:
            raise ValueError('unexpected optimizer parameter')
        if state!=dict(numel=0,state='unowned_empty') and state!=dict(numel=state['numel'],state='initialized',step=step):
            raise ValueError('optimizer update cursor differs')
        counts[kind]+=state['numel']
    if counts!=scope['local_numel']:
        raise ValueError('optimizer parameter ownership differs')
    return record['total_seconds']


def audit_outputs(package, collection, *, initial_policy):
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.context_replica_schedule import load_schedule
    from multimodal_jev.slurp_portable_release import load
    from tools.reports.audit_context_replica_training import check_result,check_schedule
    package,collection=Path(package),Path(collection)
    c=read(package/'config.json');run=collection/'run';plan=derive(c['batch_config'])
    world=plan['dp_replicas'];start=c['start_update'];stop=c['stop_update']
    if (c['stage']!='instruction' or c['instruction_objective']['kind']!='candidate_infonce'
            or c['instruction_objective']['probability_coefficient']!=0.
            or not 1<=world<=8 or c['batch_config']['inference_instances']!=0):
        raise ValueError('pure InfoNCE declared DP required')
    check_result(read(run/'result.json'),start,stop,plan,c['engineering_prefix'])
    if read(collection/'supervision/terminal.json')['exit_code']!=0:
        raise ValueError('supervised trainer did not exit normally')
    rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
    population={r['id']:r for r in rows}
    batches,_=load_schedule(package/'source/schedule',c['schedule_sha256'],
        release_sha256=c['release_sha256'],population=population,batch_config=c['batch_config'],
        parent_root=package/'source/parent-schedule',parent_manifest_sha256=c['parent_schedule_sha256'],
        parent_release_sha256=c['schedule_release_sha256'])
    check_schedule(batches,plan)
    tail=None
    if c.get('experiment_kind')=='infonce_tail_admission':
        from tools.reports.context_infonce_tail import contract,select
        tail=contract(c);batches=select(batches,c,population.keys())
    initial=read_execution_bundle(initial_policy,expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    checkpoints=check_artifact_inventory(run,c,world)
    prior=initial['tensors'];summaries=[]
    for index in range(start,stop):
        step=index+1;planned=batches[index]
        real=check_batch(read(run/f'batch-{index:04d}.json'),planned,population,task,
            policy=c['source_policy_id']+':instruction-update'+str(index),width=plan['global_batch'],world=world)
        records=[read(run/f'rank-{rank}/update-{step:04d}.json') for rank in range(world)]
        scopes=[read(run/f'rank-{rank}/checkpoint-scope.json') for rank in range(world)]
        times=[check_update(record,c,step=step,real=real,checkpoint=step in checkpoints,scope=scope)
               for record,scope in zip(records,scopes,strict=True)]
        if step not in checkpoints:
            summaries.append(dict(step=step,real_queries=real,checkpoint=False,
                optimizer_evidence='runtime_record_only_no_independent_tensor_restore',
                slowest_rank_seconds=max(times)))
            continue
        commit=read(run/f'checkpoint-{step:04d}-committed.json')
        shards={f'rank-{rank}/checkpoint-{step:04d}.pt':file_hash(run/f'rank-{rank}/checkpoint-{step:04d}.pt')
                for rank in range(world)}
        if (commit['updates']!=step or commit['contract']!=expected_contract(c)
                or commit['rank_shards']!=shards
                or commit['execution_file']!=f'execution-{step:04d}.pt'
                or file_hash(run/commit['execution_file'])!=commit['execution_sha256']
                or commit['joint_policy_directory']!=f'policy-{step:04d}'
                or file_hash(run/f'policy-{step:04d}/MANIFEST.json')!=commit['joint_policy_manifest_sha256']):
            raise ValueError('committed identity, objective or complete shards differ')
        bundle=read_execution_bundle(run/f'policy-{step:04d}',expected_policy_id=commit['joint_policy']['policy_id'],
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        execution=torch.load(run/commit['execution_file'],map_location='cpu',weights_only=True)
        if (set(bundle['tensors'])!=set(execution)|{'decision_token_row'}
                or set(bundle['tensors'])!=set(initial['tensors'])
                or any(not torch.equal(v,bundle['tensors'][n]) for n,v in execution.items())
                or not torch.equal(bundle['tensors']['decision_token_row'],initial['tensors']['decision_token_row'])):
            raise ValueError('published policy or frozen marker differs from export')
        states=[torch.load(run/name,map_location='cpu',weights_only=True) for name in shards]
        master=audit_masters(states,execution,world_size=world)
        restored=[]
        for rank,state in enumerate(states):
            recovery=restore_shard(state,c,rank=rank,step=step)
            if (records[rank]['optimizer_audit']!=recovery['optimizer_audit']
                    or scopes[rank]!=recovery['scope']):
                raise ValueError('actual optimizer state or scope differs')
            restored.append(recovery['recovery'])
        changes={kind:sum(not torch.equal(v,prior[n]) for n,v in bundle['tensors'].items() if n.startswith(prefix))
                 for kind,prefix in [('lora','backbone.'),('head','decision_head.')]}
        prior=bundle['tensors']
        summaries.append(dict(step=step,real_queries=real,checkpoint=True,master_export=master,cpu_restored=restored,
            changed_execution_tensors=changes,slowest_rank_seconds=max(times)))
    if any(sum(s.get('changed_execution_tensors',{}).get(kind,0) for s in summaries)==0 for kind in ('lora','head')):
        raise ValueError('both LoRA and head must change')
    for rank in range(world):
        result=read(run/f'rank-{rank}/result.json')
        if (result['passed'] is not True or result['start_update']!=start or result['stop_update']!=stop
                or len(result['executions'])!=113 or set(result['executions'].values())!={'bfloat16'}):
            raise ValueError('all-rank BF16 completion required')
    return dict(schema='context-infonce-training-audit-v1',status='finite_infonce_artifacts_audited',
        world_size=world,updates=summaries,checkpoint_steps=sorted(checkpoints),
        real_exposures=sum(s['real_queries'] for s in summaries),tail_admission=tail,fresh_process_npu_resume_verified=False,
        npu_gradient_equivalence_verified=False,actual_length_group_recomputation_verified=False,
        training_dispatch_allowed=False,scope='Collected source/targets/LR/FP32 Adam/BF16 export; no efficacy or NPU replay claim')


def audit_collected(c, *, validator=None, command_builder=None):
    from tools.reports.context_instruction_operations import validate,command
    from tools.reports.context_replica_training_operations import IMAGE
    from tools.reports.context_dual_probe_operations import check_runtime
    operations,_=load_config(c['operations_config'],c['operations_local'])
    validator=validate if validator is None else validator
    command_builder=command if command_builder is None else command_builder
    manifest,config,_=validator(operations)
    root=Path(operations['operations_root']);tag=operations['tag']
    launch=read(root/(tag+'-launch.json'));observed=read(root/(tag+'-inspect.json'))
    check_runtime(dict(argv=command_builder(operations,manifest,config),image=IMAGE),launch,observed)
    collection=root/(tag+'-collected');transport=read(root/(tag+'-collected.transport.json'))
    if (transport['status']!='transport_complete_independent_audit_required'
            or transport['container_id']!=launch['container_id'] or transport['image']!=IMAGE
            or transport['state']!=observed['State'] or transport['remote_root']!=manifest['remote_root']
            or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))
            or file_hash(collection.with_suffix('.tar'))!=transport['archive_sha256']):
        raise ValueError('successful authenticated transport required')
    actual={str(p.relative_to(collection)):file_hash(p) for p in collection.rglob('*') if p.is_file()}
    if actual!=transport['files'] or any(p.is_symlink() for p in collection.rglob('*')):
        raise ValueError('collected bytes changed')
    if any(actual.get(n)!=manifest['files'][n] for n in ('config.json','run.py')):
        raise ValueError('executed configuration differs from sealed package')
    result=audit_outputs(operations['package'],collection,initial_policy=c['initial_policy'])
    result.update(container_id=launch['container_id'],manifest_sha256=operations['manifest_sha256'],
                  transport_sha256=file_hash(root/(tag+'-collected.transport.json')))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);torch.set_num_threads(1)
    result=audit_collected(c);result['provenance']=provenance
    with Path(c['output']).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','world_size','container_id','training_dispatch_allowed')}))
