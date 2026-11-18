import json
import subprocess
from pathlib import Path
import pytest
from tools.reports import observe_context_existing_jobs as module


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    base=Path('base');base.mkdir()
    monkeypatch.setattr(module,'BASE',base)
    monkeypatch.setattr(module,'TARGETS',{'grpo':('launch.json','history.json')})
    launch=dict(container_id='original',image='fixed',remote_root='/data/original')
    (base/'launch.json').write_text(json.dumps(launch))
    Path('history.json').write_text(json.dumps(dict(launch,observed_unix=1000,next_ordinary_observation_unix=1600)))
    return base


def test_legacy_600second_value_cannot_shorten_interval(setup):
    def forbidden(_):raise AssertionError('remote called early')
    assert module.observe('grpo',remote=forbidden,now=1899)['next_ordinary_observation_unix']==1900
    assert not list((setup/'ordinary-observations-v1').glob('*attempt.json'))


def test_running_handle_and_new_interval(setup):
    calls=[]
    def remote(args):
        calls.append(args)
        if args[1]=='inspect':return json.dumps([dict(Id='original',Image='fixed',State={'Running':True})])
        return json.dumps({'ranks':[],'commits':[]})
    result=module.observe('grpo',remote=remote,now=1900)
    assert result['status']=='observed_original_handle' and len(calls)==2
    assert result['next_ordinary_observation_unix']==2800
    assert module.observe('grpo',remote=remote,now=2000)['status']=='not_due_no_remote_call'
    assert len(calls)==2


@pytest.mark.parametrize('failure',['timeout','identity'])
def test_failed_attempt_preserves_handle_and_throttles_retry(setup,failure):
    def remote(args):
        if failure=='timeout':raise subprocess.TimeoutExpired(args,55)
        return json.dumps([dict(Id='other',Image='fixed',State={'Running':True})])
    result=module.observe('grpo',remote=remote,now=1900)
    assert result['status']=='observation_failed_preserve_original_handle'
    assert 'state' not in result
    assert module.observe('grpo',remote=remote,now=1950)['status']=='not_due_no_remote_call'


def test_terminal_handle_does_not_exec_or_restart(setup):
    calls=[]
    def remote(args):
        calls.append(args)
        return json.dumps([dict(Id='original',Image='fixed',State={'Running':False,'ExitCode':0})])
    result=module.observe('grpo',remote=remote,now=1900)
    assert len(calls)==1 and result['state']['ExitCode']==0 and 'progress' not in result


def test_legacy_missing_image_still_throttles_but_conflicting_image_rejected(setup):
    p=Path('history.json');d=json.loads(p.read_text());d.pop('image');p.write_text(json.dumps(d))
    assert module.observe('grpo',now=1800)['status']=='not_due_no_remote_call'
    d['image']='wrong';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='image identity'):module.observe('grpo',now=1800)
