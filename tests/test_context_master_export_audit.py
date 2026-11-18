import copy
import json

import pytest
import torch

from tools.reports.context_master_export_audit import audit,inspect_committed,sha


def fixture():
    names=['_fsdp_wrapped_module.backbone.q.lora_A.default.weight','_fsdp_wrapped_module.decision_head.projection.weight']
    tensors=[torch.arange(6,dtype=torch.float32).reshape(2,3)/7,torch.arange(5,dtype=torch.float32)/9]
    states=[]
    splits=[[0,1,4,6,6],[0,0,0,2,5]]
    for rank in range(4):
        parameters={n:v.flatten()[splits[i][rank]:splits[i][rank+1]].clone() for i,(n,v) in enumerate(zip(names,tensors))}
        states.append(dict(rank=rank,world_size=4,names=names,parameters=parameters,contract={'stage':'task'},completed_updates=2))
    return states,{n.removeprefix('_fsdp_wrapped_module.'):v.bfloat16() for n,v in zip(names,tensors)}


def test_master_shards_reconstruct_execution_including_empty_owners():
    states,execution=fixture();result=audit(states,execution)
    assert result['exact_bf16_from_master_shards'] and result['numel']==11


@pytest.mark.parametrize('fault',['master','export','rank','cursor','name','dtype','coverage'])
def test_individually_valid_but_inconsistent_states_rejected(fault):
    states,execution=fixture();name=states[0]['names'][0]
    if fault=='master':states[1]['parameters'][name][0]+=.5
    if fault=='export':execution[next(iter(execution))][0,0]+=.5
    if fault=='rank':states[0]['rank']=1
    if fault=='cursor':states[2]['completed_updates']=3
    if fault=='name':states[3]['names']=list(reversed(states[3]['names']))
    if fault=='dtype':states[0]['parameters'][name]=states[0]['parameters'][name].half()
    if fault=='coverage':states[1]['parameters'][name]=torch.empty(0)
    with pytest.raises(ValueError):audit(states,execution)


def test_committed_cli_input_binds_shards_and_contract(tmp_path):
    states,execution=fixture();shards={}
    for rank,state in enumerate(states):
        p=tmp_path/f'rank-{rank}/checkpoint-0002.pt';p.parent.mkdir();torch.save(state,p)
        shards[str(p.relative_to(tmp_path))]=sha(p)
    p=tmp_path/'execution-0002.pt';torch.save(execution,p)
    commit=tmp_path/'checkpoint-0002-committed.json'
    commit.write_text(json.dumps(dict(updates=2,contract=states[0]['contract'],rank_shards=shards,
        execution_file=p.name,execution_sha256=sha(p))))
    c=dict(run_root=str(tmp_path),step=2,commit_sha256=sha(commit))
    assert inspect_committed(c)['exact_bf16_from_master_shards']
    with (tmp_path/'rank-0/checkpoint-0002.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='committed shard changed'):inspect_committed(c)


def test_dp2_requires_explicit_topology_and_checks_both_shards():
    states,execution=fixture()
    merged=[]
    for rank in range(2):
        s=copy.deepcopy(states[rank*2]);s.update(rank=rank,world_size=2)
        s['parameters']={n:torch.cat([states[i]['parameters'][n].flatten() for i in (rank*2,rank*2+1)]) for n in s['names']}
        merged.append(s)
    with pytest.raises(ValueError):audit(merged,execution)
    assert audit(merged,execution,world_size=2)['rank_shards']==2
    merged[1]['parameters'][merged[1]['names'][0]][0]+=.25
    with pytest.raises(ValueError,match='reproduce BF16'):audit(merged,execution,world_size=2)


def test_declared_dp6_width_reconstructs_execution():
    # Topology is data: six shards split the same originals, some empty per rank.
    names=['_fsdp_wrapped_module.backbone.q.lora_A.default.weight',
           '_fsdp_wrapped_module.decision_head.projection.weight']
    tensors=[torch.arange(6,dtype=torch.float32).reshape(2,3)/7,torch.arange(5,dtype=torch.float32)/9]
    splits=[[0,1,2,3,4,5,6],[0,0,1,2,3,4,5]]
    states=[]
    for rank in range(6):
        parameters={n:v.flatten()[splits[i][rank]:splits[i][rank+1]].clone() for i,(n,v) in enumerate(zip(names,tensors))}
        states.append(dict(rank=rank,world_size=6,names=names,parameters=parameters,
                           contract={'stage':'task'},completed_updates=2))
    execution={n.removeprefix('_fsdp_wrapped_module.'):v.bfloat16() for n,v in zip(names,tensors)}
    result=audit(states,execution,world_size=6)
    assert result['rank_shards']==6 and result['numel']==11
    with pytest.raises(ValueError):audit(states,execution)
