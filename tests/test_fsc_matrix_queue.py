import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_fsc_transfer_matrix as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    root=tmp_path/'prior';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',{})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),predecessor_suite='suite',pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})


def test_ten_minute_wait_failure_preserves_no_dispatch(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);waits=[]
    def sleep(n):
        waits.append(n);atomic_json(Path(cfg['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_training(cfg,tmp_path)
    assert waits==[600]


def test_missing_process_is_not_completion(tmp_path):
    cfg=fixture(tmp_path);p=Path(cfg['predecessor_controller'])/'launch.json'
    atomic_json(p,dict(pid=999999999,argv=['missing']));cfg['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError,match='missing without terminal'):module.await_training(cfg,tmp_path)


def test_incomplete_matrix_never_scores(tmp_path,monkeypatch):
    root=tmp_path/'matrix';root.mkdir();atomic_json(root/'protocol.json',dict(checkpoints=[dict(key='step0')],image='fixed'))
    monkeypatch.setattr(module,'score',lambda *a:pytest.fail('partial matrix scored'))
    with pytest.raises(ValueError,match='seven-policy'):module.run_matrix(dict(matrix_root=str(root),image='fixed'),tmp_path)


def test_failed_training_never_prepares_matrix(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);builder=tmp_path/'builder.json';atomic_json(builder,{})
    cfg['builder_config']=str(builder);cfg['pins'][str(builder)]=file_hash(builder)
    atomic_json(Path(cfg['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module,'prepare',lambda *a:pytest.fail('created target matrix despite failed source'))
    with pytest.raises(ValueError,match='predecessor failed'):module.run(cfg,tmp_path)


@pytest.mark.parametrize('fail_at',[None,'Q-step6'])
def test_serial_acceptance_before_complete_matrix_scoring(tmp_path,monkeypatch,fail_at):
    from contextlib import nullcontext
    root=tmp_path/'matrix';root.mkdir();out=tmp_path/'out';out.mkdir()
    keys=['step0']+[f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)]
    for name in ('task.json','population.json'):atomic_json(root/name,{})
    entries=[dict(key=k,root=str(root/k)) for k in keys]
    protocol=dict(image='image',task=str(root/'task.json'),population=str(root/'population.json'),checkpoints=entries)
    atomic_json(root/'protocol.json',protocol);digest=file_hash(root/'protocol.json')
    for e in entries:
        gate=Path(e['root']);gate.mkdir()
        atomic_json(gate/'config.json',dict(pins={str(root/'protocol.json'):digest}))
        atomic_json(gate/'source-manifest.json',{})
        atomic_json(gate/'prepared.json',dict(root=str(gate),image='image',physical_devices=[4],create_argv=[],
            config_sha256=file_hash(gate/'config.json'),source_manifest_sha256=file_hash(gate/'source-manifest.json')))
    events=[]
    monkeypatch.setattr(module,'wait_resources',lambda *a:None)
    monkeypatch.setattr(module,'locks',lambda *a:nullcontext())
    def launch(gate,*args):events.append(('launch',gate.name));return dict(container_id=gate.name)
    def wait(cid,*args):events.append(('exit',cid))
    def accept(e,*args):
        events.append(('accept',e['key']))
        if e['key']==fail_at:raise ValueError('backend failed')
        return dict(audit={}),{}
    def score(*args):
        assert events==[(phase,key) for key in keys for phase in ('launch','exit','accept')]
        events.append(('score','all'));return dict(done=True)
    monkeypatch.setattr(module,'launch',launch);monkeypatch.setattr(module,'wait_exact',wait)
    monkeypatch.setattr(module,'authenticate_gate',accept);monkeypatch.setattr(module,'score',score)
    cfg=dict(matrix_root=str(root),image='image',pins={},lock_directory=str(tmp_path))
    if fail_at:
        with pytest.raises(ValueError,match='backend failed'):module.run_matrix(cfg,out)
        assert events[-1]==('accept',fail_at) and not (out/'comparison.json').exists()
    else:
        module.run_matrix(cfg,out)
        assert module.read(out/'comparison.json')['done']
