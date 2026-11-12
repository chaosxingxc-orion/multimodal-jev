import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_batch32_control as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path,monkeypatch):
    items=[{'id':str(i)} for i in range(89)]
    monkeypatch.setattr(module,'load_pilot',lambda *args:(items,{}))
    c=dict(pins={},steps=[],initial_adapter='/initial',initial_gate='/gate')
    protocol=tmp_path/'protocol.json';c['protocol']=str(protocol)
    atomic_json(protocol,dict(schema='intent-finite-batch32-control-v1',total_updates=3,global_requests_per_update=32,total_requests=96,
        schedule=[[r['id'] for r in module.schedule(items,317,i,32)] for i in range(3)]))
    c['pins'][str(protocol)]=file_hash(protocol)
    for index in range(3):
        train=tmp_path/f'train{index}';suite=tmp_path/f'suite{index}';train.mkdir();suite.mkdir()
        cfg=dict(arm='D',step=index,seed=317,lr=1e-4,global_requests=32,model_path='/base',pilot='/pilot',manifest_sha256='m',
            initial_adapter='/initial',acceptance_root='/gate',lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',runtime_profile='fixed')
        if index:
            prior=c['steps'][-1];cfg.update(previous_root=prior['training_root'],previous_suite=prior['suite_root'],
                initial_adapter=prior['training_root']+'/run/adapter',acceptance_root=prior['suite_root'])
        atomic_json(train/'config-template.json',cfg);atomic_json(suite/'config-template.json',dict(training_root=str(train)))
        for root in (train,suite):
            atomic_json(root/'source-manifest.json',{'file':'same'});atomic_json(root/'prepared.json',{})
            for name in ('config-template.json','source-manifest.json','prepared.json'):c['pins'][str(root/name)]=file_hash(root/name)
        c['steps'].append(dict(training_root=str(train),suite_root=str(suite)))
    return c


def test_three_contiguous_identical_source_updates(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);module.validate(c)


@pytest.mark.parametrize('defect',['source','optimizer','batch','schedule'])
def test_rehashed_but_confounded_control_rejected(tmp_path,monkeypatch,defect):
    c=fixture(tmp_path,monkeypatch)
    if defect=='schedule':
        p=Path(c['protocol']);r=module.read(p);r['schedule'][0].reverse()
    elif defect=='source':
        p=Path(c['steps'][1]['training_root'])/'source-manifest.json';r={'file':'changed'}
    else:
        p=Path(c['steps'][1]['training_root'])/'config-template.json';r=module.read(p)
        r['adam_betas' if defect=='optimizer' else 'global_requests']=[0.,.999] if defect=='optimizer' else 8
    atomic_json(p,r);c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):module.validate(c)


def test_waits600_then_stops_failed_predecessor(tmp_path,monkeypatch):
    root=tmp_path/'pre';root.mkdir();(root/'run').mkdir();atomic_json(root/'config.json',{})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    c=dict(predecessor_controller=str(root),pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')});waits=[]
    def sleep(seconds):waits.append(seconds);atomic_json(root/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_predecessor(c,tmp_path)
    assert waits==[600]
