"""Independent audit of the versioned arithmetic's bounded NPU execution.

Reuse existing population, reward, estimator, checkpoint and export auditors.
This adds arithmetic/source identity and retains the bounded status explicitly.
"""
import argparse
import json
from pathlib import Path
import torch

from tools.jev.context_rl_evaluation_plan import bound, sha
from tools.jev.audit_context_rl_training import update_targets, verify_checkpoint
from tools.jev.context_rl_group_audit import audit as audit_groups
from tools.jev.audit_context_checkpoint_lifecycle import audit as audit_lifecycle
from tools.reports.context_probability_policy_v2 import PROFILE


def validate_config(c):
    expected=dict(stage='task',engineering_prefix=True,training_dispatch_allowed=False,
                  stop_update=2,global_batch=128,microbatch=8,physical_devices=[4,5,6,7],
                  runtime_profile='qwen3-asr-context-bf16-packed-row-v1',binary_policy_arithmetic=PROFILE)
    if (any(type(c.get(k)) is not type(v) or c[k] != v for k,v in expected.items())
            or c.get('start_update') not in (0,1) or c.get('checkpoint_interval') not in (1,64)
            or c.get('algorithm') not in ('grpo_v1','rloo_v1')
            or c.get('behavior_bounds') != dict(probability_max_abs=2e-7,binary_logprob_max_abs=2e-6)):
        raise ValueError('bounded frozen arithmetic candidate required')


def audit(collection, transport, launch, package, initial_policy, schedule):
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.context_training_schedule import load_schedule
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    root,package=Path(collection),Path(package)
    state=transport['state']
    if (state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or transport['container_id']!=launch['container_id'] or transport['image']!=launch['image']
            or launch['image']!='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
            or transport['status']!='transport_complete_independent_audit_required'
            or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))):
        raise ValueError('successful unchanged fixed-image transport required')
    for name in transport['files']:bound(root,transport,name)
    def read(name):return json.loads(bound(root,transport,name).read_text())
    m=read('PREPARED.json');c=read('config.json');validate_config(c)
    if sha(package/'PREPARED.json')!=launch['manifest_sha256']:
        raise ValueError('executed candidate snapshot differs')
    for name,digest in m['files'].items():
        if sha(package/name)!=digest:raise ValueError('candidate file changed')
    from tools.reports.prepare_context_arithmetic_candidate import transform_actor, transform_trainer
    original=Path('.local/decision-objective-research/context-baseline-sequence-v4/grpo_v1-package/source')
    for rel,transform in [('src/multimodal_jev/context_native_actor.py',transform_actor),
                          ('tools/jev/train_context_online_candidate.py',transform_trainer)]:
        if (package/'source'/rel).read_text()!=transform((original/rel).read_text()):
            raise ValueError('unregistered actor/trainer mutation')
    if sha(package/'source/tools/jev/context_probability_policy_v2.py')!=sha('tools/reports/context_probability_policy_v2.py'):
        raise ValueError('arithmetic module differs')
    terminal=read('supervision/terminal.json')
    if (not terminal['passed'] or not terminal['actor_started'] or terminal['forced_cleanup']
            or terminal['actor_exit_code']!=0 or terminal['service_exit_code']!=0
            or terminal['child_exit_codes']!=[0,0] or read('service/terminal.json')['exit_code']!=0):
        raise ValueError('both actual processes must exit normally')
    expected_result=dict(passed=True,start_update=c['start_update'],stop_update=2,
                         schedule_updates=634,learning_population=40513,engineering_prefix=True)
    if read('run/result.json')!=expected_result:raise ValueError('incomplete bounded execution')
    initial=read_execution_bundle(initial_policy,expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
    population={r['id']:r for r in rows}
    batches,_=load_schedule(schedule,c['schedule_sha256'],release_sha256=c['release_sha256'],population=population)
    if len(population)!=40513 or len(batches)!=634:raise ValueError('complete learning schedule required')
    lifecycle=audit_lifecycle(root/'run',files={p[4:]:h for p,h in transport['files'].items() if p.startswith('run/')},
        start_update=c['start_update'],stop_update=2,checkpoint_interval=c['checkpoint_interval'])
    current=read('service/ready.json')['policy_id']
    if c['start_update']==0 and current!=c['source_policy_id']:raise ValueError('full Stage1 source mismatch')
    summaries=[];records=[]
    for index in range(c['start_update'],2):
        receipt=read(f'run/rollout-{index:04d}.json')
        targets=update_targets(read(f'run/batch-{index:04d}.json'),receipt,batches[index],population,task,
            update=index,policy_id=current,reference_id=c['source_policy_id'])
        groups=torch.load(bound(root,transport,f'run/groups-{index:04d}.pt'),map_location='cpu',weights_only=True)
        summary=audit_groups(groups,receipt['queries'],targets,algorithm=c['algorithm'],update=index,
            policy_id=current,reference_policy_id=c['source_policy_id'])
        step=index+1;current=lifecycle['policy_ids'][index-c['start_update']]
        bundle=read_execution_bundle(root/f'run/policy-{step:04d}',expected_policy_id=current,
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        if not torch.equal(bundle['tensors']['decision_token_row'],initial['tensors']['decision_token_row']):
            raise ValueError('frozen marker changed')
        verify_native_adapter(root/f'run/adapter-{step:04d}',bundle)
        for rank in range(4):
            record=read(f'run/rank-{rank}/update-{step:04d}.json');parity=record['behavior_parity_history']
            if (record['real_queries']!=summary['real_queries'] or len(parity)!=4
                    or any(len(group)!=4 for group in parity)
                    or any(not p['passed'] or p['threshold_flips'] for group in parity for p in group)):
                raise ValueError('actual four-rank/microbatch parity failed')
            records.append(dict(rank=rank,update=step,actor_seconds=record['actor_seconds'],
                total_seconds=record['total_seconds'],peak_allocated=record['peak_allocated'],
                metrics=record['metrics'],behavior_parity_history=parity))
        if step in lifecycle['recovery_steps']:
            commit=read(f'run/checkpoint-{step:04d}-committed.json');contract=commit['contract']
            expected={k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
                'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds',
                'binary_policy_arithmetic')}
            expected.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
                objective=dict(group_size=16,reward='complete_candidate_set_f1',kl=.01,auxiliary='cosine_0.3_to_0.1_first_80pct'),
                grad_clip=1.,world_size=4)
            if contract!=expected:raise ValueError('numeric checkpoint contract mismatch')
            verify_checkpoint(root/'run',step,contract,bundle)
        summaries.append(summary)
    for rank in range(4):
        r=read(f'run/rank-{rank}/result.json')
        if (not r['passed'] or r['start_update']!=c['start_update'] or r['stop_update']!=2
                or len(r['executions'])!=113 or set(r['executions'].values())!={'bfloat16'}):
            raise ValueError('rank completion/BF16 execution differs')
    return dict(passed=True,actor_container=launch['container_id'],service_container=launch['container_id'],
        image=launch['image'],state=state,files=transport['files'],start_update=c['start_update'],stop_update=2,
        binary_policy_arithmetic=PROFILE,source_policy_id=c['source_policy_id'],algorithm=c['algorithm'],
        updates=summaries,rank_records=records,supervision=terminal,
        scope='Actual bounded DP4/LoRA/head backward, optimizer, engine activation, complete batch and checkpoint audit. No full RL curve or efficacy claim; replay equality separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('collection','transport','launch','package','initial-policy','schedule','output'):
        p.add_argument('--'+name,required=True)
    a=vars(p.parse_args());out=Path(a.pop('output'))
    for key in ('transport','launch'):a[key]=json.loads(Path(a[key]).read_text())
    torch.set_num_threads(1)
    result=audit(**a)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','actor_container','start_update','stop_update','algorithm')}))
