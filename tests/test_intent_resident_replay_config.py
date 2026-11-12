import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import probe_intent_resident_replay_npu as tool


def fixture(monkeypatch):
    configs=[dict(arm='D',step=i,global_requests=8,seed=317,lr=1e-4,model_path='/model',pilot='/pilot',manifest_sha256='hash',lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',runtime_profile='fixed',initial_adapter='/initial' if i==0 else '/t0/run/adapter',acceptance_root='/g0' if i==0 else '/s0') for i in range(2)]
    configs[1].update(previous_root='/t0',previous_suite='/s0')
    data={};pins={};references=[]
    for i,c in enumerate(configs):
        references.append(dict(training=f'/t{i}',suite=f'/s{i}'));data[f'/t{i}/config.json']=c
        data[f'/s{i}/verified.json']=dict(root=f'/t{i}',updates=i+1,arm='D',files={f'/state{i}':'hash'})
        for p in [f'/t{i}/config.json',f'/t{i}/launch.json',f'/s{i}/verified.json',f'/s{i}/ACCEPTED.json',f'/state{i}']:pins[p]='hash'
    monkeypatch.setattr(tool,'read',lambda p:data[str(p)]);monkeypatch.setattr(tool,'file_hash',lambda p:'hash')
    import queue_span_dp4_resume
    monkeypatch.setattr(queue_span_dp4_resume,'check_pins',lambda p:None)
    return dict(schema='intent-resident-two-update-replay-v1',references=references,pins=pins),configs,data


def test_fixed_two_step_lineage(monkeypatch):
    plan,configs,_=fixture(monkeypatch);assert tool.validate(plan)==configs


@pytest.mark.parametrize('bad',['lr','step','objective','history','statepin','partial'])
def test_replay_does_not_change_learning_factors(monkeypatch,bad):
    plan,configs,data=fixture(monkeypatch)
    if bad=='lr':configs[1]['lr']=2e-5
    elif bad=='step':configs[1]['step']=2
    elif bad=='objective':configs[1]['objective']={'name':'smoothed-utility-v1'}
    elif bad=='history':configs[1]['initial_adapter']='/other'
    elif bad=='statepin':plan['pins'].pop('/state1')
    else:plan['references'].pop()
    with pytest.raises(ValueError):tool.validate(plan)
