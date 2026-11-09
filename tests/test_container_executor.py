import json
from types import SimpleNamespace as NS
import subprocess
import pytest
import multimodal_jev.container_executor as module


def fixture(tmp_path):
    root=tmp_path/'run';root.mkdir();source=root/'source';source.mkdir()
    spec={'host':'local','image':'sha256:'+'a'*64,'cpus':2,'memory_gib':4,'cpuset':'0-1'}
    runtime={'shared_root':str(tmp_path),**{role:dict(spec) for role in ('rollout','actor','export')}}
    executor=module.ContainerExecutor(root,source,runtime)
    output=root/'iteration'/'rollout';output.parent.mkdir()
    return executor,output


def test_timeout_observes_original_container_and_never_relaunches(tmp_path,monkeypatch):
    executor,output=fixture(tmp_path);calls=[]
    def call(host,argv):
        calls.append(argv)
        if argv[1]=='image':return NS(returncode=0,stdout='[]',stderr='')
        if argv[1]=='run':raise subprocess.TimeoutExpired(argv,30)
        intent=json.loads((output.parent/'rollout-launch-intent.json').read_text())
        return NS(returncode=0,stdout=json.dumps([{'Id':'id1','Name':'/'+intent['name'],
            'Image':'sha256:'+'a'*64,'State':{'Status':'exited','ExitCode':0,'OOMKilled':False}}]),stderr='')
    monkeypatch.setattr(module,'call',call)
    executor.execute('rollout',output,['tool'])
    executor.execute('rollout',output,['tool'])
    assert sum(argv[1]=='run' for argv in calls)==1
    assert json.loads((output.parent/'rollout-container.json').read_text())['container_id']=='id1'
    with pytest.raises(ValueError,match='command changed'):executor.execute('rollout',output,['other'])


def test_observation_error_is_not_a_failed_job_or_relaunch(tmp_path,monkeypatch):
    executor,output=fixture(tmp_path);calls=[];observations=[]
    def call(host,argv):
        calls.append(argv)
        if argv[1]=='image':return NS(returncode=0,stdout='[]',stderr='')
        if argv[1]=='run':return NS(returncode=0,stdout='id1',stderr='')
        observations.append(1)
        if len(observations)==1:return NS(returncode=1,stdout='',stderr='temporary transport error')
        intent=json.loads((output.parent/'rollout-launch-intent.json').read_text())
        return NS(returncode=0,stdout=json.dumps([{'Id':'id1','Name':'/'+intent['name'],
            'Image':'sha256:'+'a'*64,'State':{'Status':'exited','ExitCode':137,'OOMKilled':True}}]),stderr='')
    monkeypatch.setattr(module,'call',call);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='stage failed'):executor.execute('rollout',output,['tool'])
    assert len(observations)==2 and sum(argv[1]=='run' for argv in calls)==1


def test_preexisting_stage_without_launch_identity_is_not_started(tmp_path,monkeypatch):
    executor,output=fixture(tmp_path);output.mkdir()
    monkeypatch.setattr(module,'call',lambda *a:pytest.fail('must not launch'))
    with pytest.raises(RuntimeError,match='untracked'):executor.execute('actor',output,['tool'])


def test_wrong_image_or_name_is_rejected():
    with pytest.raises(ValueError,match='identity'):
        module.verify_container({'Name':'/a','Image':'other'},'a','expected')


def test_cli_parse_failure_stops_without_observation_loop(tmp_path,monkeypatch):
    executor,output=fixture(tmp_path);calls=[]
    def call(host,argv):
        calls.append(argv)
        if argv[1]=='image':return NS(returncode=0,stdout='[]',stderr='')
        return NS(returncode=125,stdout='',stderr='unknown flag: --unsupported')
    monkeypatch.setattr(module,'call',call)
    with pytest.raises(RuntimeError,match='before container creation'):
        executor.execute('rollout',output,['tool'])
    assert [argv[1] for argv in calls]==['image','run']


def test_disk_allocation_failure_is_retained_without_resubmission(tmp_path,monkeypatch):
    executor,output=fixture(tmp_path)
    def call(host,argv):
        if argv[1]=='image':return NS(returncode=0,stdout='[]',stderr='')
        return NS(returncode=125,stdout='',stderr='mkdir overlay: no space left on device')
    monkeypatch.setattr(module,'call',call)
    with pytest.raises(RuntimeError,match='allocation failed'):
        executor.execute('actor',output,['tool'])
    assert (output.parent/'rollout-launch-intent.json').exists()
