"""Independent finite dual-credit collection audit; no execution or efficacy claim."""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.local_config import load_config
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
from tools.jev.context_rl_evaluation_plan import bound,sha
from tools.jev.audit_context_rl_training import update_targets,verify_checkpoint
from tools.reports.context_dual_group_audit import audit as audit_groups
from tools.reports.prepare_context_dual_throughput_pair import write


def expected_contract(c):
    result={k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
        'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds')}
    result.update(dual_objective=dict(algorithm=c['algorithm'],arm=c['dual_arm'],ranking_coefficient=.1,
        probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id']),
        dual_runtime_identity=c['dual_runtime_identity'],optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        objective=dict(group_size=16,reward='component_binary_correctness',kl=.01,auxiliary='constant_pair0.1_nll0.3'),
        grad_clip=1.,world_size=4)
    return result


def audit(c):
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.context_training_schedule import load_schedule
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    from tools.reports.context_dual_training_operations import validate,command,IMAGE
    from tools.reports.context_dual_probe_operations import check_runtime
    operations,_=load_config(c['operations_config'],c['operations_local'])
    manifest,config,service=validate(operations)
    start=config['start_update'];parent=None
    if start:
        parent_config,_=load_config(c['parent_audit_config'],c['parent_audit_local'])
        parent_operations,_=load_config(parent_config['operations_config'],parent_config['operations_local'])
        if validate(parent_operations)[1]['start_update']!=0:
            raise ValueError('replay parent must be initial prefix, not another replay')
        parent=audit(parent_config)
        binding=manifest['replay']
        if (parent['schema']!='context-dual-prefix-audit-v1' or parent['container_id']!=binding['parent_container'] or
                parent['manifest_sha256']!=binding['parent_manifest_sha256']):
            raise ValueError('replay requires independently audited original prefix')
    root=Path(c['collection']);transport=json.loads(Path(c['transport']).read_text())
    launch=json.loads(Path(c['launch']).read_text());inspected=json.loads(Path(c['inspect']).read_text())
    check_runtime(dict(argv=command(operations,manifest,config),image=IMAGE),launch,inspected)
    if (transport['status']!='transport_complete_independent_audit_required' or
            any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches')) or
            transport['state']!=inspected['State'] or transport['container_id']!=launch['container_id'] or
            transport['image']!=IMAGE or transport['remote_root']!=manifest['remote_root']):
        raise ValueError('successful unchanged actual runtime transport required')
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('collection symlink forbidden')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(transport['files']):raise ValueError('complete collected population required')
    for name in actual:bound(root,transport,name)
    if sha(Path(c['collection']).with_suffix('.tar'))!=transport['archive_sha256']:
        raise ValueError('transport archive changed')
    def read(name):return json.loads(bound(root,transport,name).read_text())
    if read('config.json')!=config or read('service-config.json')!=service or read('PREPARED.json')!=manifest:
        raise ValueError('collected configuration differs from sealed package')
    for name,key in (('config.json','config_sha256'),('service-config.json','service_config_sha256'),('PREPARED.json','manifest_sha256')):
        if transport['files'][name]!=launch[key]:raise ValueError('launch binding differs')
    terminal=read('supervision/terminal.json')
    if (terminal['passed'] is not True or terminal['actor_started'] is not True or terminal['forced_cleanup'] or
            terminal['actor_exit_code']!=0 or terminal['service_exit_code']!=0 or terminal['child_exit_codes']!=[0,0] or
            read('service/terminal.json')['exit_code']!=0):
        raise ValueError('normal completion of both supervised processes required')
    if read('run/result.json')!=dict(passed=True,start_update=start,stop_update=2,schedule_updates=634,
                                     learning_population=40513,engineering_prefix=True):
        raise ValueError('complete finite two-step result required')
    initial=read_execution_bundle(c['initial_policy'],expected_policy_id=config['source_policy_id'],
        expected_contract=config['source_contract'],expected_base_id=config['base_id'])
    rows,task,_=load(c['release'],config['release_sha256'],'learning',purpose='optimize')
    population={r['id']:r for r in rows};candidates=[v['id'] for v in task['schema']['candidates']]
    batches,_=load_schedule(c['schedule'],config['schedule_sha256'],release_sha256=config['release_sha256'],population=population)
    if len(population)!=40513 or len(batches)!=634 or len(candidates)!=60:
        raise ValueError('fixed full clean SLURP population and schedule required')
    ready=read('service/ready.json');current=ready['policy_id']
    expected_initial=manifest['replay']['initial_behavior_policy_id'] if start else config['source_policy_id']
    if (current!=expected_initial or ready['contract']!=config['source_contract'] or
            ready['max_batch_queries']!=service['policy']['max_batch_queries'] or
            ready['max_batch_tokens']!=16384 or ready['physical_devices']!=[7] or
            ready['transport_protocol']!='bulk_dual_logits_candidate_v1'):
        raise ValueError('actual service startup differs')
    contract=expected_contract(config);summaries=[];prior=initial['tensors'];changes=[]
    if start:
        parent_root=Path(parent_config['collection'])
        prior=read_execution_bundle(parent_root/'run/policy-0001',expected_policy_id=current,
            expected_contract=config['source_contract'],expected_base_id=config['base_id'])['tensors']
        if launch['container_id']==parent['container_id']:raise ValueError('fresh container required for replay')
        for rank in range(4):
            if read(f'run/rank-{rank}/restored.json')!=dict(exact_masters=True,exact_adam=True,completed_updates=1):
                raise ValueError('all-rank exact restoration required')
    for index in range(start,2):
        step=index+1;receipt=read(f'run/rollout-{index:04d}.json');batch=read(f'run/batch-{index:04d}.json')
        if receipt['runtime_identity']!=config['dual_runtime_identity']:
            raise ValueError('recorded rollout runtime differs')
        targets=update_targets(batch,receipt,batches[index],population,task,update=index,
            policy_id=current,reference_id=config['source_policy_id'])
        groups=torch.load(bound(root,transport,f'run/groups-{index:04d}.pt'),map_location='cpu',weights_only=True)
        expected_order=[batch['records'][i]['request_id'] for rank in batches[index]['rank_indices'] for i in rank]
        if [g['request_id'] for g in groups]!=expected_order:raise ValueError('actual DP schedule grouping differs')
        summary=audit_groups(groups,receipt['queries'],targets,candidates=candidates,algorithm=config['algorithm'],
            arm=config['dual_arm'],update=index,policy_id=current,reference_policy_id=config['source_policy_id'])
        commit=read(f'run/checkpoint-{step:04d}-committed.json')
        shards={f'rank-{rank}/checkpoint-{step:04d}.pt':transport['files'][f'run/rank-{rank}/checkpoint-{step:04d}.pt'] for rank in range(4)}
        if (commit['updates']!=step or commit['contract']!=contract or commit['rank_shards']!=shards or
                commit['execution_file']!=f'execution-{step:04d}.pt' or
                commit['execution_sha256']!=transport['files'][f'run/execution-{step:04d}.pt'] or
                commit['joint_policy_directory']!=f'policy-{step:04d}' or
                commit['joint_policy_manifest_sha256']!=transport['files'][f'run/policy-{step:04d}/MANIFEST.json']):
            raise ValueError('committed recovery/publication identity differs')
        current=commit['joint_policy']['policy_id']
        bundle=read_execution_bundle(root/f'run/policy-{step:04d}',expected_policy_id=current,
            expected_contract=config['source_contract'],expected_base_id=config['base_id'])
        verify_native_adapter(root/f'run/adapter-{step:04d}',bundle)
        verify_checkpoint(root/'run',step,contract,bundle)
        from tools.reports.context_master_export_audit import audit as audit_master_export
        master_export=audit_master_export([
            torch.load(bound(root,transport,f'run/rank-{rank}/checkpoint-{step:04d}.pt'),map_location='cpu',weights_only=True)
            for rank in range(4)],
            torch.load(bound(root,transport,f'run/execution-{step:04d}.pt'),map_location='cpu',weights_only=True))
        if not torch.equal(bundle['tensors']['decision_token_row'],initial['tensors']['decision_token_row']):
            raise ValueError('frozen marker changed')
        changed={kind:sum(not torch.equal(v,prior[k]) for k,v in bundle['tensors'].items() if k.startswith(prefix))
                 for kind,prefix in (('lora','backbone.'),('head','decision_head.'))}
        changes.append(changed);prior=bundle['tensors']
        activation=read(f'run/activation-{step:04d}.json')
        if activation['error'] is not None or activation['policy_id']!=current:
            raise ValueError('published policy not activated')
        times=[]
        for rank in range(4):
            record=read(f'run/rank-{rank}/update-{step:04d}.json');parity=record['behavior_parity_history']
            if (record['update']!=step or record['real_queries']!=summary['real_queries'] or
                    record['next_policy_id']!=current or record['rollout']!=receipt or record['checkpoint'] is not True or
                    len(parity)!=4 or any(len(g)!=4 or any(p!=dict(passed=True,error=None) for p in g) for g in parity)):
                raise ValueError('complete actual DP4 parity/update required')
            state=torch.load(bound(root,transport,f'run/rank-{rank}/checkpoint-{step:04d}.pt'),map_location='cpu',weights_only=True)
            scope=validate_trainable_scope([(n,torch.nn.Parameter(v)) for n,v in state['parameters'].items()])
            if read(f'run/rank-{rank}/checkpoint-scope.json')!=scope:
                raise ValueError('saved tensor whitelist/scope differs')
            times.append(record['total_seconds'])
        if any(type(v) not in (int,float) or not 0<v<float('inf') for v in times):raise ValueError('invalid full-step timer')
        summary.update(slowest_rank_step_seconds=max(times),changed_execution_tensors=changed,
                       master_export=master_export);summaries.append(summary)
    if any(sum(row[k] for row in changes)==0 for k in ('lora','head')):
        raise ValueError('finite prefix must exercise nonzero LoRA and head changes')
    for rank in range(4):
        result=read(f'run/rank-{rank}/result.json')
        if (result['passed'] is not True or result['start_update']!=start or result['stop_update']!=2 or
                len(result['executions'])!=113 or set(result['executions'].values())!={'bfloat16'}):
            raise ValueError('rank completion/BF16 execution differs')
    if start:
        from tools.reports.compare_context_dual_throughput_pair import equal
        for name in ('groups-0001.pt','execution-0002.pt',*[f'rank-{r}/checkpoint-0002.pt' for r in range(4)]):
            original=bound(parent_root,parent,'run/'+name);replayed=bound(root,transport,'run/'+name)
            if not equal(torch.load(original,map_location='cpu',weights_only=True),
                         torch.load(replayed,map_location='cpu',weights_only=True)):
                raise ValueError('fresh-process replay tensor/state differs: '+name)
        original_batch=json.loads(bound(parent_root,parent,'run/batch-0001.json').read_text())
        if original_batch!=read('run/batch-0001.json'):raise ValueError('replayed schedule inputs differ')
        original_commit=json.loads(bound(parent_root,parent,'run/checkpoint-0002-committed.json').read_text())
        if current!=original_commit['joint_policy']['policy_id']:raise ValueError('replayed policy differs')
    return dict(passed=True,schema='context-dual-replay-audit-v1' if start else 'context-dual-prefix-audit-v1',container_id=launch['container_id'],
        image=IMAGE,state=transport['state'],files=transport['files'],source_policy_id=config['source_policy_id'],
        manifest_sha256=launch['manifest_sha256'],updates=summaries,
        fresh_process_replay_verified=bool(start),training_dispatch_allowed=False,
        scope='Recorded finite DP4 population/dual credits/runtime/checkpoint/export audit; replay and efficacy separate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);torch.set_num_threads(1)
    result=audit(c);result['config_provenance']=provenance;write(c['output'],result)
    print(json.dumps({k:result[k] for k in ('passed','container_id','training_dispatch_allowed')}))
