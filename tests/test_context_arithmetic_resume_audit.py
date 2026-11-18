import json

import pytest
import torch

from tools.reports.audit_context_arithmetic_resume import compare_step, terminal


def put(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    if name.endswith('.pt'):torch.save(value,path)
    else:path.write_text(json.dumps(value))


@pytest.fixture
def pair(tmp_path):
    left=tmp_path/'original';right=tmp_path/'resumed'
    for root,policy in ((left,'original-policy'),(right,'resumed-policy')):
        put(root,'batch-0064.json',dict(policy_id=policy,rank_indices=[[0]],
            records=[dict(request_id='query',weight=1,target=dict(labels=[True,False]))]))
        put(root,'groups-0064.pt',[dict(request_id='query',policy_id=policy,
            actions=torch.tensor([[[1,0]]]),advantages=torch.tensor([[.25]]))])
    return left,right


def test_semantic_comparison_allows_authenticated_serialization_identity_difference(pair):
    groups=compare_step(*pair,64)
    assert groups[0]['policy_id']=='resumed-policy'


@pytest.mark.parametrize('fault',['action','advantage','dtype','extra','request','weight','policy'])
def test_divergent_replay_rejected(pair,fault):
    left,right=pair
    if fault=='weight':
        p=right/'batch-0064.json';value=json.loads(p.read_text());value['records'][0]['weight']=0
        p.write_text(json.dumps(value))
    else:
        value=torch.load(right/'groups-0064.pt',weights_only=True)
        if fault=='action':value[0]['actions'][0,0,1]=1
        if fault=='advantage':value[0]['advantages'][0,0]=.5
        if fault=='dtype':value[0]['actions']=value[0]['actions'].int()
        if fault=='extra':value.append(value[0])
        if fault=='request':value[0]['request_id']='different'
        if fault=='policy':value[0]['policy_id']='unbound'
        torch.save(value,right/'groups-0064.pt')
    with pytest.raises(ValueError):compare_step(left,right,64)


@pytest.fixture
def completed(tmp_path):
    put(tmp_path,'supervision/terminal.json',dict(passed=True,actor_started=True,forced_cleanup=False,
        actor_exit_code=0,service_exit_code=0,child_exit_codes=[0,0]))
    put(tmp_path,'service/terminal.json',dict(exit_code=0))
    put(tmp_path,'run/result.json',dict(passed=True,start_update=64,stop_update=66,
        schedule_updates=634,learning_population=40513,engineering_prefix=True))
    for rank in range(4):
        put(tmp_path,f'run/rank-{rank}/restored.json',dict(exact_masters=True,exact_adam=True,completed_updates=64))
        put(tmp_path,f'run/rank-{rank}/result.json',dict(passed=True,start_update=64,stop_update=66,
            executions={str(i):'bfloat16' for i in range(113)}))
    return tmp_path


def test_complete_finite_segment(completed):
    terminal(completed,64,66)


@pytest.mark.parametrize('fault',['forced','child','cursor','restore','dtype'])
def test_incomplete_or_failed_terminal_rejected(completed,fault):
    name={'forced':'supervision/terminal.json','child':'supervision/terminal.json',
        'cursor':'run/result.json','restore':'run/rank-3/restored.json','dtype':'run/rank-3/result.json'}[fault]
    p=completed/name;value=json.loads(p.read_text())
    if fault=='forced':value['forced_cleanup']=True
    if fault=='child':value['child_exit_codes']=[0,1]
    if fault=='cursor':value['stop_update']=634
    if fault=='restore':value['exact_adam']=False
    if fault=='dtype':value['executions']['0']='float32'
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):terminal(completed,64,66)
