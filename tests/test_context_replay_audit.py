import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from context_replay_audit import require_exact,verify_update


def state():
    return {'parameters':{'w':torch.tensor([1.,2.])},'optimizer':{'state':{0:{
        'step':torch.tensor(2.),'exp_avg':torch.tensor([.1,.2]),
        'exp_avg_sq':torch.tensor([.01,.02])}}},'samples':torch.tensor([0,1]),'cursor':2}


def test_exact_nested_master_adam_and_sample_state():
    value=state();require_exact(value,copy.deepcopy(value))


@pytest.mark.parametrize('fault',['master','adam','dtype','samples','cursor','key'])
def test_replay_difference_is_rejected(fault):
    original=state();value=copy.deepcopy(original)
    if fault=='master':value['parameters']['w'][0]+=.01
    elif fault=='adam':value['optimizer']['state'][0]['exp_avg_sq'][0]+=.001
    elif fault=='dtype':value['parameters']['w']=value['parameters']['w'].double()
    elif fault=='samples':value['samples'][0]=1
    elif fault=='cursor':value['cursor']=3
    elif fault=='key':del value['optimizer']['state'][0]['step']
    with pytest.raises(ValueError):require_exact(original,value)


def test_update_artifacts_require_restore_and_prior_behavior(tmp_path):
    import json
    left,right=tmp_path/'continuous',tmp_path/'resumed'
    for root in (left,right):
        root.mkdir()
        def save_json(name,value):(root/name).write_text(json.dumps(value))
        save_json('batch-0001.json',{'policy_id':'step1','records':[{'id':'audio'}]})
        torch.save([{'actions':torch.tensor([0,1])}],root/'groups-0001.pt')
        save_json('checkpoint-0001-committed.json',{'joint_policy':{'policy_id':'step1'}})
        save_json('checkpoint-0002-committed.json',dict(contract={'algorithm':'grpo_v1'},
            updates=2,execution_file='execution-0002.pt',joint_policy_directory='policy-0002'))
        torch.save({'w':torch.ones(2,dtype=torch.bfloat16)},root/'execution-0002.pt')
        (root/'policy-0002').mkdir()
        torch.save({'w':torch.ones(2,dtype=torch.bfloat16),'marker':torch.zeros(2)},root/'policy-0002/execution.pt')
        for rank in range(4):
            (root/f'rank-{rank}').mkdir()
            torch.save(state(),root/f'rank-{rank}/checkpoint-0002.pt')
            save_json(f'rank-{rank}/restored.json',dict(exact_masters=True,exact_adam=True,completed_updates=1))
    assert verify_update(left,right)['rank_shards']==4
    saved=state();saved['optimizer']['state'][0]['exp_avg_sq'][1]+=.001
    torch.save(saved,right/'rank-3/checkpoint-0002.pt')
    with pytest.raises(ValueError,match='rank-3'):verify_update(left,right)
    torch.save(state(),right/'rank-3/checkpoint-0002.pt')
    (right/'rank-0/restored.json').write_text(json.dumps(dict(exact_masters=True,exact_adam=False,completed_updates=1)))
    with pytest.raises(ValueError,match='restore receipt'):verify_update(left,right)
