"""Queue orchestration tests; Docker/engine acceptance is explicitly synthetic."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_jev_shards import make_plan
from test_sfc_jev import load_tool
from test_sfc_jev_shards import fixture as input_fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture(tmp_path,monkeypatch):
    m=load_tool('run_sfc_jev_shard_queue')
    release,_=input_fixture(tmp_path);plan=make_plan(release,'test_id',12)
    path=tmp_path/'population.json';atomic_json(path,plan)
    cfg=dict(prompt_id='parent',shard_plan=str(path),shard_plan_sha256=file_hash(path),shard_id=plan['shards'][0]['id'],
        split='test_id',image='pinned-image',jev_probe=str(tmp_path/'probe'),context_audit=str(tmp_path/'audit'),
        context_acceptance=str(tmp_path/'acceptance.json'),asset_audit=str(tmp_path/'assets.json'),
        context_container_id='cpu-cid',context_result_sha256='context-result')
    atomic_json(Path(cfg['context_acceptance']),{'synthetic':True})
    cfg['context_acceptance_sha256']=file_hash(cfg['context_acceptance'])
    template=tmp_path/'template.json';atomic_json(template,cfg)
    scorer=tmp_path/'scorer.py';scorer.write_text('# synthetic official scorer')
    # Keep actual plan reproduction; replace only model/context prerequisites.
    monkeypatch.setattr(m,'select_inputs',lambda _:None)
    monkeypatch.setattr(m,'check_probe',lambda _:None)
    monkeypatch.setattr(m,'accept_probe',lambda _:None)
    monkeypatch.setattr(m,'verify_context',lambda _:dict(container_id='cpu-cid',result_sha256='context-result',all_inputs_fit=True))
    monkeypatch.setattr(m.importlib.metadata,'version',lambda _: 'fixture-version')
    data={'tools/jev/run_sfc_jev_shard_queue.py':b'# exported runner', 'src/example.py':b'# exported module'}
    import hashlib
    pins={k:hashlib.sha256(v).hexdigest() for k,v in data.items()}
    monkeypatch.setattr(m,'committed_snapshot',lambda *a:('c'*40,data,pins))
    root=tmp_path/'queue'
    intent=m.prepare(root,template,file_hash(template),scorer,file_hash(scorer),'c'*40)
    # Unit test drives real pins except Python import provenance, tested below.
    def source(root,intent):
        if m.code_files(root/'source')!=intent['source_files']:raise ValueError('queue source changed')
    monkeypatch.setattr(m,'check_source',source)
    monkeypatch.setattr(m,'SOURCE_FILES',tuple(data))
    return m,root,intent,m.read(root/'config-template.json'),plan


def test_complete_queue_orders_dispatch_acceptance_and_merge(tmp_path,monkeypatch):
    m,root,intent,cfg,plan=fixture(tmp_path,monkeypatch);events=[]
    def dispatch(child,prep):
        events.append(('dispatch',child.name));atomic_json(child/'launch.json',dict(container_id=child.name))
    monkeypatch.setattr(m,'dispatch_once',dispatch)
    monkeypatch.setattr(m,'accept_shard',lambda child:events.append(('accept',child.name)) or {'accepted':True})
    def merge(path,pin,roster,roster_pin,scorer,scorer_pin,out):
        assert file_hash(roster)==roster_pin
        assert [r['shard_id'] for r in m.read(roster)['runs']]==intent['shard_ids']
        assert all(m.read(root/s/'evaluation-terminal.json')['exit_code']==0 for s in intent['shard_ids'])
        assert all({k:v for k,v in m.read(root/s/'config.json').items() if k!='shard_id'}==
                   {k:v for k,v in cfg.items() if k!='shard_id'} for s in intent['shard_ids'])
        events.append(('merge',plan['decisions']))
    monkeypatch.setattr(m,'merge_score',merge)
    m.execute(root,file_hash(root/'queue-plan.json'))
    assert events==[('dispatch','shard-0000'),('accept','shard-0000'),
                    ('dispatch','shard-0001'),('accept','shard-0001'),('merge',24)]
    assert m.read(root/'progress.json')['phase']=='complete'
    with pytest.raises(FileExistsError):m.execute(root,file_hash(root/'queue-plan.json'))


@pytest.mark.parametrize('failure',['prepare','dispatch','accept'])
def test_failed_shard_never_dispatches_later_shards_or_merges(tmp_path,monkeypatch,failure):
    m,root,intent,_,_=fixture(tmp_path,monkeypatch);events=[]
    prepare=m.prepare_shard
    def fail():raise RuntimeError('synthetic '+failure+' failure')
    def prepared(*a):
        child,prep=prepare(*a)
        if child.name=='shard-0001' and failure=='prepare':fail()
        return child,prep
    def dispatch(child,prep):
        events.append(child.name)
        if child.name=='shard-0000' and failure=='dispatch':fail()
        atomic_json(child/'launch.json',{})
    def accept(child):
        if child.name=='shard-0000' and failure=='accept':fail()
        return {}
    monkeypatch.setattr(m,'prepare_shard',prepared);monkeypatch.setattr(m,'dispatch_once',dispatch)
    monkeypatch.setattr(m,'accept_shard',accept)
    monkeypatch.setattr(m,'merge_score',lambda *a:pytest.fail('must not score incomplete population'))
    with pytest.raises(RuntimeError,match='synthetic'):m.execute(root,file_hash(root/'queue-plan.json'))
    assert events==['shard-0000']
    failed=root/('shard-0001' if failure=='prepare' else 'shard-0000')
    assert m.read(failed/'evaluation-terminal.json')['exit_code']==1
    if failure=='prepare':assert len(m.read(root/'completed-runs.json')['runs'])==1
    else:assert not (root/'completed-runs.json').exists()
    with pytest.raises(FileExistsError):m.execute(root,file_hash(root/'queue-plan.json'))


@pytest.mark.parametrize('change',['source','extra_source','template','scorer','population','roster','context','probe'])
def test_drift_and_failed_prerequisites_prevent_any_dispatch(tmp_path,monkeypatch,change):
    m,root,intent,cfg,_=fixture(tmp_path,monkeypatch)
    if change=='source':
        p=root/'source/src/example.py';p.chmod(0o644);p.write_text('changed')
    if change=='extra_source':
        p=root/'source/src';p.chmod(0o755);(p/'injected.py').write_text('extra')
    if change=='template':atomic_json(root/'config-template.json',{})
    if change=='scorer':Path(intent['official_scorer']).write_text('changed')
    if change=='population':atomic_json(Path(cfg['shard_plan']),{})
    if change=='roster':
        intent['shard_ids'].pop();atomic_json(root/'queue-plan.json',intent)
    if change=='context':monkeypatch.setattr(m,'verify_context',lambda _:dict(container_id='wrong',result_sha256='context-result',all_inputs_fit=True))
    if change=='probe':
        def bad(_):raise ValueError('probe failed')
        monkeypatch.setattr(m,'accept_probe',bad)
    monkeypatch.setattr(m,'dispatch_once',lambda *a:pytest.fail('must not dispatch'))
    with pytest.raises(ValueError):m.execute(root,file_hash(root/'queue-plan.json'))
    assert not (root/'dispatch-intent.json').exists()


def test_preparation_keeps_partition_evidence_and_source_visible_readonly(tmp_path,monkeypatch):
    m,root,intent,cfg,plan=fixture(tmp_path,monkeypatch)
    part=tmp_path/'partition-evidence/partition.json';part.parent.mkdir();part.write_text('{}')
    plan['partition']={'path':str(part)};atomic_json(Path(cfg['shard_plan']),plan)
    child,prep=m.prepare_shard(root,intent,cfg,intent['shard_ids'][0])
    argv=prep['argv'];mounts=[argv[i+1] for i,a in enumerate(argv[:-1]) if a=='-v']
    assert len({p.split(':')[1] for p in mounts})==len(mounts)
    for p in (root/'source',part.parent,Path(cfg['context_audit']),Path(cfg['jev_probe'])):
        assert f'{p}:{p}:ro' in mounts
    devices=[argv[i+1] for i,a in enumerate(argv[:-1]) if a=='--device']
    assert '/dev/davinci6:/dev/davinci6' in devices and '/dev/davinci7:/dev/davinci7' in devices
    assert not any('/dev/davinci'+str(d)+':' in v for d in range(4) for v in devices)
    assert not (root/'source').is_symlink() and (child/'source').resolve()==root/'source'
    assert Path(cfg['context_acceptance']).parent==root
    assert file_hash(cfg['context_acceptance'])==cfg['context_acceptance_sha256']
    assert prep['wall_seconds']==86400 and prep['automatic_retry'] is False


def test_committed_snapshot_rejects_modified_and_untracked_python(tmp_path):
    m=load_tool('run_sfc_jev_shard_queue');repo=tmp_path/'repo';repo.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    git('init','-q');(repo/'src').mkdir();(repo/'tools').mkdir()
    p=repo/'src/a.py';p.write_text('original');(repo/'tools/b.py').write_text('runner')
    git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
    revision,files,pins=m.committed_snapshot(repo,'HEAD')
    assert revision==git('rev-parse','HEAD') and files['src/a.py']==b'original'
    p.write_text('changed')
    with pytest.raises(ValueError,match='executing source'):m.committed_snapshot(repo,'HEAD')
    p.write_text('original');(repo/'src/untracked.py').write_text('extra')
    with pytest.raises(ValueError,match='executing source'):m.committed_snapshot(repo,'HEAD')


def test_exported_runner_rejects_imports_from_worktree(tmp_path,monkeypatch):
    m=load_tool('run_sfc_jev_shard_queue');root=tmp_path/'queue';source=root/'source'
    runner=source/'tools/jev/run_sfc_jev_shard_queue.py';runner.parent.mkdir(parents=True);runner.write_text('# source')
    intent=dict(source=str(source),source_files=m.code_files(source))
    with pytest.raises(ValueError,match='exported queue runner'):m.check_source(root,intent)
    monkeypatch.setattr(m,'__file__',str(runner))
    with pytest.raises(ValueError,match='dependencies'):m.check_source(root,intent)


def test_exported_snapshot_loads_real_dependencies_in_fresh_process(tmp_path):
    import os
    import shutil
    repo=Path(__file__).parents[1];root=tmp_path/'queue';source=root/'source'
    for folder in ('src','tools'):
        shutil.copytree(repo/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
    script="""import sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'source/tools/jev'))
import run_sfc_jev_shard_queue as queue
queue.check_source(root,dict(source=str(root/'source'),source_files=queue.code_files(root/'source')))
print('exported imports accepted')
"""
    result=subprocess.run([sys.executable,'-B','-c',script,str(root)],cwd=tmp_path,
        env={**os.environ,'PYTHONPATH':str(source/'src')},capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='exported imports accepted'
