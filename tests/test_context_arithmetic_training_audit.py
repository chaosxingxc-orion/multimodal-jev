"""Exercise the full634 orchestration; tensor/schedule subaudits are substitutes.

Their actual implementations are covered separately. This fixture proves only
control flow and cross-step binding, never successful training or NPU behavior.
"""
import json
from pathlib import Path

import pytest
import torch

from tools.reports import audit_context_arithmetic_training as module
from multimodal_jev import slurp_official_training, context_training_schedule, context_policy_artifacts


@pytest.fixture
def full(tmp_path, monkeypatch):
    root=tmp_path/'collection';root.mkdir();data={};calls={'groups':[], 'policies':[], 'checkpoints':[]}
    image='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
    c=dict(stage='task',algorithm='grpo_v1',engineering_prefix=False,training_dispatch_allowed=True,
           start_update=0,stop_update=634,checkpoint_interval=64,global_batch=128,microbatch=8,
           physical_devices=[4,5,6,7],release_sha256='release',release='/fixture',base_id='base',
           source_policy_id='initial',source_contract={},runtime_profile='profile',schedule_sha256='schedule',behavior_bounds={})
    c['binary_policy_arithmetic']=module.PROFILE
    data['config.json']=c
    data['supervision/terminal.json']=dict(passed=True,actor_started=True,forced_cleanup=False,
                                         actor_exit_code=0,service_exit_code=0,child_exit_codes=[0,0])
    data['service/terminal.json']={'exit_code':0};data['service/ready.json']={'policy_id':'initial'}
    data['run/result.json']=dict(passed=True,start_update=0,stop_update=634,schedule_updates=634,
                                learning_population=40513,engineering_prefix=False)
    saves=[*range(64,634,64),634]
    contract={k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
        'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds','binary_policy_arithmetic')}
    contract.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        objective=dict(group_size=16,reward='complete_candidate_set_f1',kl=.01,auxiliary='cosine_0.3_to_0.1_first_80pct'),grad_clip=1.,world_size=4)
    for index in range(634):
        step=index+1;previous='initial' if index==0 else f'policy-{index}'
        data[f'run/batch-{index:04d}.json']={'policy_id':previous,'index':index}
        data[f'run/rollout-{index:04d}.json']=dict(behavior_policy_id=previous,fixed_reference_policy_id='initial',update=index,queries=[])
        for rank in range(4):
            data[f'run/rank-{rank}/update-{step:04d}.json']=dict(real_queries=65 if step in (317,634) else 128,
                behavior_parity_history=[[dict(passed=True,threshold_flips=0) for _ in range(4)] for _ in range(4)],
                total_seconds=17.+rank/10)
        if step in saves:data[f'run/checkpoint-{step:04d}-committed.json']={'contract':contract}
    for rank in range(4):
        data[f'run/rank-{rank}/result.json']=dict(passed=True,start_update=0,stop_update=634,
                                                executions={str(i):'bfloat16' for i in range(113)})
    # In-memory authenticated transport substitute isolates orchestration from
    # thousands of disk files. Real bound/hash validation has separate tests.
    class Artifact:
        def __init__(self,name):self.name=name
        def read_text(self):return json.dumps(data[self.name])
    monkeypatch.setattr(module,'bound',lambda root,proof,name:Artifact(name))
    monkeypatch.setattr(module.torch,'load',lambda path,**kw: path.name)
    monkeypatch.setattr(slurp_official_training,'load',lambda *a,**k:([dict(id=str(i)) for i in range(40513)],{},{}))
    monkeypatch.setattr(context_training_schedule,'load_schedule',lambda *a,**k:([dict(index=i) for i in range(634)],{}))
    monkeypatch.setattr(module,'audit_lifecycle',lambda *a,**k:dict(policy_ids=[f'policy-{i}' for i in range(1,635)],recovery_steps=saves))
    def targets(batch,receipt,planned,population,task,**kw):
        assert batch['index']==planned['index']==kw['update']
        if batch['policy_id']!=kw['policy_id'] or receipt['fixed_reference_policy_id']!=kw['reference_id']:
            raise ValueError('cross-step policy or fixed reference differs')
        return {'index':kw['update']}
    monkeypatch.setattr(module,'update_targets',targets)
    def groups(saved,receipts,targets,**kw):
        assert saved==f"run/groups-{kw['update']:04d}.pt"
        assert targets['index']==kw['update'];calls['groups'].append(kw)
        return dict(real_queries=65 if kw['update'] in (316,633) else 128)
    monkeypatch.setattr(module,'audit_groups',groups)
    def policy(path,**kw):
        calls['policies'].append(kw['expected_policy_id'])
        return {'tensors':{'decision_token_row':torch.zeros(1)}}
    monkeypatch.setattr(context_policy_artifacts,'read_execution_bundle',policy)
    monkeypatch.setattr(context_policy_artifacts,'verify_native_adapter',lambda *a:None)
    monkeypatch.setattr(module,'verify_checkpoint',lambda run,step,contract,bundle:calls['checkpoints'].append(step))
    transport=dict(status='transport_complete_independent_audit_required',container_id='container',image=image,
                   state=dict(Status='exited',ExitCode=0,OOMKilled=False),bad_pins=[],bad_package=[],
                   launch_binding_mismatches=[],files={'config.json':'hash'})
    args=dict(collection=root,transport=transport,launch=dict(container_id='container',image=image,config_sha256='hash'),
              experiment={'release_sha256':'release'},schedule='/schedule',initial_policy='/initial')
    return args,data,calls


@pytest.mark.parametrize('algorithm',['grpo_v1','rloo_v1'])
def test_complete634_control_flow(full,algorithm):
    args,data,calls=full;data['config.json']['algorithm']=algorithm
    data['run/checkpoint-0064-committed.json']['contract']['algorithm']=algorithm
    result=module.audit(**args)
    assert result['passed'] and len(result['updates'])==634
    assert sum(x['real_queries'] for x in result['updates'])==81026
    assert calls['checkpoints']==[*range(64,634,64),634]
    assert calls['policies']==['initial',*[f'policy-{i}' for i in range(1,635)]]
    assert all(x['algorithm']==algorithm and x['reference_policy_id']=='initial' for x in calls['groups'])
    assert result['updates'][-1]['slowest_rank_step_seconds']==17.3


@pytest.mark.parametrize('fault',['stale','reference','parity','population','time','contract','rank_dtype','incomplete'])
def test_late_failure_never_passes(full,fault):
    args,data,_=full
    if fault=='stale':data['run/batch-0633.json']['policy_id']='policy-1'
    elif fault=='reference':data['run/rollout-0633.json']['fixed_reference_policy_id']='moving'
    elif fault=='parity':data['run/rank-3/update-0634.json']['behavior_parity_history'][3][3]['passed']=False
    elif fault=='population':data['run/rank-3/update-0634.json']['real_queries']=128
    elif fault=='time':data['run/rank-3/update-0634.json']['total_seconds']=float('nan')
    elif fault=='contract':data['run/checkpoint-0634-committed.json']['contract']['objective']['kl']=0.
    elif fault=='rank_dtype':data['run/rank-3/result.json']['executions']['0']='float32'
    elif fault=='incomplete':data['run/result.json']['stop_update']=633
    with pytest.raises(ValueError):module.audit(**args)


@pytest.mark.parametrize('fault',['config','checkpoint'])
def test_arithmetic_contract_cannot_be_dropped(full,fault):
    args,data,_=full
    if fault=='config':data['config.json'].pop('binary_policy_arithmetic')
    else:data['run/checkpoint-0634-committed.json']['contract'].pop('binary_policy_arithmetic')
    with pytest.raises(ValueError):module.audit(**args)
