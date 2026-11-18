import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
from tools.reports.context_attribute_actor_candidate import load, generate, REFERENCE_ID
from tools.reports.build_context_attribute_candidate import targets
from tools.reports.context_probability_policy_v2 import binary_logprobs
from tools.reports.context_attribute_objective_candidate import prepare_groups


def setup(arm):
    root=Path.cwd()
    menu=json.loads((root/'.local/decision-objective-research/slurp-native-attribute-prob-candidate-v1/catalog.json').read_text())
    actor,receipt=load(menu,arm,source_root=root)
    active=receipt['decision_ids'];all_ids=[n['id'] for n in menu['nodes']];idx=[all_ids.index(k) for k in active]
    y=torch.tensor([targets(menu,menu['events'][0]['id'])['labels']],dtype=torch.bool)[:,idx]
    n=len(active);ids=torch.arange(n)[None]
    behavior=tuple(p.detach() for p in binary_logprobs(torch.linspace(-.7,.5,n)[None]))
    a=torch.randint(0,2,(1,16,n),generator=torch.Generator().manual_seed(12));a[:,0]=y.long();a[:,1]=0
    item=dict(request_id='request-a',policy_id='policy-a',inputs={'input_ids':ids},positions=ids,
        labels=y,observed=torch.ones_like(y),request_weight=1.,real_request=True,
        behavior=behavior,actions=a,reference_policy=behavior,reference_policy_id=REFERENCE_ID,
        attribute_contract_id=receipt['contract_id'],decision_ids=active)
    options=dict(stage='task',ranking_coefficient=.3,probability_coefficient=.3,
        algorithm='grpo_v1',kl_coefficient=.01,reference_policy_id=REFERENCE_ID,
        attribute_contract_id=receipt['contract_id'])
    return actor,receipt,menu,item,options


@pytest.mark.parametrize('arm',['direct','q_input','q_credit'])
def test_native_transport_matches_objective_and_has_isolated_namespace(arm):
    actor,r,m,item,opts=setup(arm);batch=actor.make_batch([item],**opts)
    row=batch.non_tensor_batch['multi_modal_inputs'][0]
    expected=prepare_groups(item['behavior'],item['actions'],item['labels'],item['observed'],m,
        arm=arm,request_ids=[item['request_id']],algorithm='grpo_v1')
    for k in ('rewards','advantages','old_joint'):assert torch.equal(row[k],expected[k])
    assert row['attribute_contract_id']==r['contract_id'] and row['decision_ids']==r['decision_ids']
    from multimodal_jev import context_native_actor as original
    assert actor.MODE!=original.MODE and actor._active is not original._active
    assert load(m,arm,source_root=Path.cwd())[0] is actor


@pytest.mark.parametrize('fault',['arm','node_order','reference','missing_contract','G8','catalog'])
def test_wrong_binding_is_rejected(fault):
    actor,r,m,item,opts=setup('q_credit')
    if fault=='arm':item['attribute_contract_id']=setup('q_input')[1]['contract_id']
    if fault=='node_order':item['decision_ids']=item['decision_ids'][::-1]
    if fault=='reference':opts['reference_policy_id']='different'
    if fault=='missing_contract':item.pop('attribute_contract_id')
    if fault=='G8':item['actions']=item['actions'][:,:8]
    if fault=='catalog':
        m['events'][0]['description']+=' changed'
        with pytest.raises(ValueError):generate(m,'q_credit',source_root=Path.cwd())
        return
    with pytest.raises(ValueError):actor.make_batch([item],**opts)


def test_actual_gloo_admission_rechecks_rewards_and_contract(tmp_path):
    actor,r,m,item,opts=setup('q_credit');batch=actor.make_batch([item],**opts)
    cfg=SimpleNamespace(ppo_mini_batch_size=1,ppo_micro_batch_size_per_gpu=1,
                        jev_policy_id='policy-a',jev_context_objective=opts)
    dist.init_process_group('gloo',init_method=(tmp_path/'rendezvous').as_uri(),rank=0,world_size=1)
    try:
        result=actor.admit(batch,cfg);assert result['global_real_requests']==1
        for fault in ('reward','advantage','contract','node_order','denominator','response_mask'):
            bad=copy.deepcopy(batch);row=bad.non_tensor_batch['multi_modal_inputs'][0]
            if fault=='reward':row['rewards'][0,0]+=.1
            if fault=='advantage':row['advantages'][0,0]+=.1
            if fault=='contract':row['attribute_contract_id']='invalid'
            if fault=='node_order':row['decision_ids']=row['decision_ids'][::-1]
            if fault=='denominator':row['global_real_requests']=2
            if fault=='response_mask':bad.batch['response_mask'][0,0]=0
            with pytest.raises(ValueError):actor.admit(bad,cfg)
    finally:dist.destroy_process_group()
