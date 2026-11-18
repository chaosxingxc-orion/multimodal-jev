import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools.reports import watch_context_successor_jobs as module

TAG='context-rl-rloo-arithmetic-v2'
CID='a'*64


def setup(tmp_path):
    launch=dict(container_id=CID,image=module.IMAGE,remote_root=module.REMOTE+TAG)
    (tmp_path/(TAG+'-launch.json')).write_text(json.dumps(launch))
    return launch


def forbidden(_):raise AssertionError('unexpected remote call')


def test_missing_launch_and_discovery_never_query_remote(tmp_path):
    assert module.observe(TAG,base=tmp_path,remote=forbidden,now=1000)['status']=='awaiting_original_launch_no_remote_call'
    setup(tmp_path)
    assert module.observe(TAG,base=tmp_path,remote=forbidden,now=1000)['next_due']==1900
    assert module.observe(TAG,base=tmp_path,remote=forbidden,now=1899)['status']=='not_due_no_remote_call'


@pytest.mark.parametrize('status,running,terminal',[('running',True,False),('created',False,False),('exited',False,True),('dead',False,True)])
def test_exact_handle_state_and_throttle(tmp_path,status,running,terminal):
    setup(tmp_path);module.observe(TAG,base=tmp_path,remote=forbidden,now=1000)
    calls=[]
    def remote(args):
        calls.append(args)
        assert args[2]==CID
        if args[1]=='inspect':
            return json.dumps([dict(Id=CID,Image=module.IMAGE,State=dict(Status=status,Running=running))])
        assert args[:2]==['docker','exec']
        return json.dumps(dict(ranks=[],commits=[]))
    r=module.observe(TAG,base=tmp_path,remote=remote,now=1900)
    assert r['status']=='observed_original_handle'
    assert len(calls)==(2 if running else 1)
    state=json.loads((tmp_path/'successor-observations-v1'/(TAG+'-state.json')).read_text())
    assert state['terminal']==terminal
    r=module.observe(TAG,base=tmp_path,remote=forbidden,now=2000)
    assert r['status']==('terminal_already_observed' if terminal else 'not_due_no_remote_call')


@pytest.mark.parametrize('fault',['timeout','identity'])
def test_failure_preserves_handle_and_defers(tmp_path,fault):
    setup(tmp_path);module.observe(TAG,base=tmp_path,remote=forbidden,now=1000)
    def remote(args):
        if fault=='timeout':raise subprocess.TimeoutExpired(args,55)
        return json.dumps([dict(Id='b'*64,Image=module.IMAGE,State={'Running':False,'Status':'exited'})])
    r=module.observe(TAG,base=tmp_path,remote=remote,now=1900)
    assert r['status']=='observation_failed_preserve_original_handle'
    assert module.observe(TAG,base=tmp_path,remote=forbidden,now=2000)['next_due']==2800


def test_bound_launch_cannot_be_replaced(tmp_path):
    launch=setup(tmp_path);module.observe(TAG,base=tmp_path,remote=forbidden,now=1000)
    launch['container_id']='b'*64
    (tmp_path/(TAG+'-launch.json')).write_text(json.dumps(launch))
    with pytest.raises(ValueError,match='bound launch changed'):
        module.observe(TAG,base=tmp_path,remote=forbidden,now=1900)


@pytest.mark.parametrize('kind,paths',[
    ('selection',['evaluation/selection/progress.json']),
    ('fsc',['evaluation/initial/inference/progress.json','evaluation/selected/inference/progress.json']),
    ('test',['evaluation/initial/progress.json','evaluation/selected/progress.json']),
    ('minds',['evaluation/initial/progress.json','evaluation/selected/progress.json'])])
def test_actual_progress_reader_does_not_read_scores(tmp_path,kind,paths):
    expected={}
    for n in paths:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True)
        expected[n]=dict(completed=64,total=9226);p.write_text(json.dumps(expected[n]))
        (p.parent/'result.json').write_text('invalid json should never be consumed')
    result=subprocess.check_output([sys.executable,'-c',module.progress_code(kind,str(tmp_path))],text=True)
    assert json.loads(result)==expected


def test_training_reader_and_scope_excludes_current_owners(tmp_path):
    assert len(module.TARGETS)==26
    assert 'context-rl-grpo-arithmetic-v2' not in module.TARGETS
    assert set(module.TARGETS.values())=={'training','selection','fsc','test','minds'}
    for rank in range(4):
        p=tmp_path/f'run/rank-{rank}/update-0064.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(total_seconds=10.,preparation_seconds=6.,actor_seconds=3.)))
    (tmp_path/'run/checkpoint-0064-committed.json').write_text('{}')
    value=json.loads(subprocess.check_output([sys.executable,'-c',module.progress_code('training',str(tmp_path))],text=True))
    assert len(value['ranks'])==4 and value['commits']==['checkpoint-0064-committed.json']
