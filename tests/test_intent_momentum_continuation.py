import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_momentum_continuation as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path, monkeypatch):
    items=[{'id': str(i)} for i in range(89)]
    monkeypatch.setattr(module,'load_pilot',lambda *a:(items,{}))
    cfg=dict(pins={},steps=[],initial_updates=1,total_updates=12)
    base=dict(step=0,arm='D',seed=317,lr=1e-4,global_requests=8,model_path='/base',pilot='/pilot',
        manifest_sha256='m',lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',runtime_profile='v2',
        initial_adapter='/initial',acceptance_root='/gate')
    ref=tmp_path/'reference';ref.mkdir();atomic_json(ref/'config-template.json',base)
    cfg['reference_training']=str(ref);cfg['pins'][str(ref/'config-template.json')]=file_hash(ref/'config-template.json')
    for i in range(12):
        train=tmp_path/f'train{i}';suite=tmp_path/f'suite{i}';train.mkdir();suite.mkdir()
        current=dict(base,step=i,adam_betas=[0.,.999])
        if i:
            current.update(previous_root=str(prior),previous_suite=str(prior_suite),
                initial_adapter=str(prior/'run/adapter'),acceptance_root=str(prior_suite))
        atomic_json(train/'config-template.json',current)
        atomic_json(suite/'config-template.json',dict(training_root=str(train)))
        for root in (train,suite):
            (root/'source').mkdir();(root/'source/file').write_text('frozen')
            atomic_json(root/'source-manifest.json',{'file':file_hash(root/'source/file')})
            atomic_json(root/'prepared.json',{})
            for name in ('config-template.json','prepared.json','source-manifest.json'):cfg['pins'][str(root/name)]=file_hash(root/name)
        if i:cfg['steps'].append(dict(training_root=str(train),suite_root=str(suite)))
        else:cfg.update(previous_training=str(train),previous_suite=str(suite))
        prior,prior_suite=train,suite
    protocol=tmp_path/'protocol.json';cfg['protocol']=str(protocol)
    atomic_json(protocol,dict(schema='intent-finite-d12-momentum-control-v1',adam_betas=[0.,.999],lr=1e-4,
        total_updates=12,checkpoints=list(range(13)),schedule=[[r['id'] for r in module.schedule(items,317,i,8)] for i in range(12)]))
    cfg['pins'][str(protocol)]=file_hash(protocol)
    return cfg


def test_complete_frozen_chain(tmp_path,monkeypatch):
    module.validate(fixture(tmp_path,monkeypatch))


@pytest.mark.parametrize('bad',['source','source_file','schedule','optimizer','budget','initial_pin'])
def test_rejects_confounded_or_incomplete_controls(tmp_path,monkeypatch,bad):
    c=fixture(tmp_path,monkeypatch)
    root=Path(c['steps'][1]['training_root'])
    if bad=='source_file':(root/'source/file').write_text('tampered')
    elif bad=='initial_pin':c['pins'].pop(str(Path(c['previous_training'])/'config-template.json'))
    elif bad=='budget':c['total_updates']=11
    else:
        p=Path(c['protocol']) if bad=='schedule' else root/('source-manifest.json' if bad=='source' else 'config-template.json')
        v=module.read(p)
        if bad=='source':v['file']='different'
        if bad=='schedule':v['schedule'][0].reverse()
        if bad=='optimizer':v['adam_betas']=[.9,.999]
        atomic_json(p,v);c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):module.validate(c)


def test_wait_interval_and_failed_predecessor(tmp_path,monkeypatch):
    root=tmp_path/'pre';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',{})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    c=dict(predecessor_controller=str(root),pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})
    waits=[]
    def sleep(s):waits.append(s);atomic_json(root/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_predecessor(c,tmp_path)
    assert waits==[600]


def test_actual_container_failure_prevents_acceptance(tmp_path,monkeypatch):
    train=tmp_path/'train';train.mkdir()
    atomic_json(train/'launch.json',dict(image='fixed',container_id='failed'))
    monkeypatch.setattr(module,'completed',lambda *a:False)
    monkeypatch.setattr(module,'verify_final',lambda *a:pytest.fail('unverified container reached acceptance'))
    with pytest.raises(ValueError,match='actual completed'):
        module.authenticate_pair(dict(training_root=str(train),suite_root=str(tmp_path/'suite')),'fixed',1)


def test_success_reauthenticates_all_predecessor_steps_and_first_momentum(tmp_path,monkeypatch):
    root=tmp_path/'pre';root.mkdir();(root/'run').mkdir()
    steps=[dict(training_root=f't{i}',suite_root=f's{i}') for i in range(1,4)]
    atomic_json(root/'config.json',dict(pins={},image='fixed',steps=steps))
    atomic_json(root/'launch.json',dict(pid=0,argv=[]))
    atomic_json(root/'run/terminal.json',dict(exit_code=0))
    atomic_json(root/'run/progress.json',dict(completed_updates=3,total_updates=3,accepted_suite='s3'))
    c=dict(predecessor_controller=str(root),image='fixed',previous_training='m1',previous_suite='ms1',
        pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})
    calls=[]
    monkeypatch.setattr(module,'validate_batch',lambda c:None)
    def authenticate(step,image,updates):
        calls.append((step['training_root'],image,updates));return {'accepted':step['suite_root']}
    monkeypatch.setattr(module,'authenticate_pair',authenticate)
    module.await_predecessor(c,tmp_path)
    assert calls==[('t1','fixed',1),('t2','fixed',2),('t3','fixed',3),('m1','fixed',1)]
    assert module.read(tmp_path/'predecessor-accepted.json')['momentum_first']=={'accepted':'ms1'}
