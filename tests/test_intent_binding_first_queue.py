from contextlib import nullcontext
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_binding_first_step as module
from multimodal_jev.baseline import atomic_json,file_hash


def test_first_update_order_and_no_duplicate_dispatch(tmp_path,monkeypatch):
    root=tmp_path/'plan';output=tmp_path/'out';output.mkdir();events=[]
    for kind in ('training','suite'):
        p=root/kind;p.mkdir(parents=True);atomic_json(p/'prepared.json',dict(create_argv=[kind]))
    cfg=dict(plan_root=str(root),policy_id='policy',image='image',lock_directory=str(tmp_path),pins={})
    monkeypatch.setattr(module,'validate',lambda c:events.append('validate'))
    monkeypatch.setattr(module,'await_predecessor',lambda *a:events.append('predecessor'))
    monkeypatch.setattr(module,'wait_resources',lambda *a:events.append('resources'))
    monkeypatch.setattr(module,'locks',lambda *a:nullcontext())
    monkeypatch.setattr(module,'check_pins',lambda *a:None)
    monkeypatch.setattr(module,'freeze_training',lambda *a:events.append('freeze-training') or {})
    monkeypatch.setattr(module,'freeze_suite',lambda *a:events.append('freeze-suite'))
    def launch(path,plan,argv):events.append('launch-'+path.name);return dict(container_id=path.name)
    monkeypatch.setattr(module,'launch',launch)
    monkeypatch.setattr(module,'wait_exact',lambda cid,*a:events.append('wait-'+cid))
    monkeypatch.setattr(module,'verify_final',lambda *a:events.append('verify-final') or dict(passed=True))
    module.run(cfg,output)
    assert events==['validate','predecessor','resources','validate','freeze-training','launch-training',
        'wait-training','resources','freeze-suite','launch-suite','wait-suite','verify-final']
    atomic_json(root/'training/create-attempt.json',{})
    events.clear()
    with pytest.raises(ValueError,match='already claimed'):module.run(cfg,output)
    assert events==['validate']


def predecessor(tmp_path):
    root=tmp_path/'prior';(root/'run').mkdir(parents=True)
    cfg=dict(image='image',previous_training='/first',previous_suite='/first-suite',
        steps=[dict(training_root=f'/t{i}',suite_root=f'/s{i}') for i in range(2,13)])
    atomic_json(root/'config.json',cfg);atomic_json(root/'launch.json',dict(pid=123,argv=['exact']))
    current=dict(predecessor_controller=str(root),image='image',pins={str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})
    output=tmp_path/'out';output.mkdir()
    return root,current,output


def test_full_predecessor_authenticates_every_actual_pair(tmp_path,monkeypatch):
    root,cfg,out=predecessor(tmp_path);seen=[]
    atomic_json(root/'run/terminal.json',dict(exit_code=0))
    atomic_json(root/'run/progress.json',dict(completed_updates=12,total_updates=12,accepted_suite='/s12'))
    monkeypatch.setattr(module,'validate_momentum',lambda *a:None)
    monkeypatch.setattr(module,'authenticate_pair',lambda step,image,n:seen.append(n) or dict(update=n))
    module.await_predecessor(cfg,out)
    assert seen==list(range(1,13))


@pytest.mark.parametrize('defect',['failed','partial','pair_failed','missing','pid_reused'])
def test_bad_predecessor_never_reaches_dispatch(tmp_path,monkeypatch,defect):
    root,cfg,out=predecessor(tmp_path)
    monkeypatch.setattr(module,'validate_momentum',lambda *a:None)
    if defect in ('failed','partial','pair_failed'):
        atomic_json(root/'run/terminal.json',dict(exit_code=1 if defect=='failed' else 0))
        atomic_json(root/'run/progress.json',dict(completed_updates=11 if defect=='partial' else 12,total_updates=12,accepted_suite='/s12'))
        def pair(*a):raise ValueError('actual pair failed')
        monkeypatch.setattr(module,'authenticate_pair',pair)
    else:
        original=Path.read_bytes
        def read(path):
            if str(path)=='/proc/123/cmdline':
                if defect=='missing':raise FileNotFoundError()
                return b'wrong\0'
            return original(path)
        monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ValueError):module.await_predecessor(cfg,out)
    assert not (out/'predecessor-accepted.json').exists()


def test_live_wait_uses_ten_minutes_then_rechecks(tmp_path,monkeypatch):
    root,cfg,out=predecessor(tmp_path);delays=[]
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'exact\0' if str(p)=='/proc/123/cmdline' else original(p))
    def sleep(seconds):
        delays.append(seconds);atomic_json(root/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_predecessor(cfg,out)
    assert delays==[600]


def final_pair(tmp_path):
    train=tmp_path/'training';suite=tmp_path/'suite';files={}
    for path in (train,suite):
        path.mkdir();atomic_json(path/'source-manifest.json',{})
        atomic_json(path/'config.json',dict(pins={},binding_manifest_sha256='manifest',adapter_id='new'))
        atomic_json(path/'launch.json',dict(container_id=path.name,image='image',
            config_sha256=file_hash(path/'config.json'),source_manifest_sha256=file_hash(path/'source-manifest.json')))
    for rank in range(4):
        out=train/'run'/f'rank-{rank}';out.mkdir(parents=True)
        for name in ('optimizer-local.pt','result.json','terminal.json','selection.json','before.json','after.json'):
            (out/name).write_text('fixture');files[str(out/name)]=file_hash(out/name)
    for name in ('selection.json','supervision.json'):
        p=train/'run'/name;p.write_text('fixture');files[str(p)]=file_hash(p)
    verified=dict(passed=True,schema='intent-binding-update-verified-v1',updates=1,arm='D',root=str(train),suite=str(suite),
        source_uses=8,view_uses=32,full_gate_requests=712,binding_manifest_sha256='manifest',exact_export_reload=True,
        adapter_id='new',files=files)
    atomic_json(suite/'verified.json',verified)
    atomic_json(suite/'ACCEPTED.json',dict(passed=True,adapter_id='new',verified_sha256=file_hash(suite/'verified.json')))
    atomic_json(suite/'controller-terminal.json',dict(exit_code=0))
    return train,suite


def test_final_checks_actual_containers_and_all_rank_evidence(tmp_path,monkeypatch):
    train,suite=final_pair(tmp_path);seen=[]
    monkeypatch.setattr(module,'completed',lambda cid,image:seen.append(cid) or True)
    assert module.verify_final(suite,train,'image')['passed']
    assert seen==['training','suite']


@pytest.mark.parametrize('defect',['running','native','missing_rank','changed_shard','foreign_adapter','failed_phase'])
def test_final_rejects_false_acceptance(tmp_path,monkeypatch,defect):
    train,suite=final_pair(tmp_path)
    monkeypatch.setattr(module,'completed',lambda *a:defect!='running')
    if defect in ('native','missing_rank','foreign_adapter'):
        v=module.read(suite/'verified.json')
        if defect=='native':v['full_gate_requests']=178
        elif defect=='missing_rank':v['files'].pop(str(train/'run/rank-3/optimizer-local.pt'))
        else:v['adapter_id']='foreign'
        atomic_json(suite/'verified.json',v);a=module.read(suite/'ACCEPTED.json')
        a['verified_sha256']=file_hash(suite/'verified.json');atomic_json(suite/'ACCEPTED.json',a)
    elif defect=='changed_shard':(train/'run/rank-3/optimizer-local.pt').write_text('changed')
    elif defect=='failed_phase':atomic_json(suite/'controller-terminal.json',dict(exit_code=1))
    with pytest.raises(ValueError):module.verify_final(suite,train,'image')
