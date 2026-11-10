import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_audio2tool_capacity as queue


def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_live_sampling_is_wait_not_dispatch(tmp_path,monkeypatch):
    start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    save(tmp_path/'evaluation-launch.json',dict(pid=os.getpid(),process_start=start))
    save(tmp_path/'queue-plan.json',{})
    monkeypatch.setattr(queue,'file_hash',lambda p:queue.SAMPLING_PLAN_SHA if p.name=='queue-plan.json' else queue.SAMPLING_LAUNCH_SHA)
    monkeypatch.setattr(queue,'remote',lambda *a,**k:pytest.fail('pending sampler must not contact Docker'))
    assert queue.sampling_ready(tmp_path) is False
    save(tmp_path/'evaluation-launch.json',dict(pid=os.getpid(),process_start='wrong-start'))
    with pytest.raises(RuntimeError,match='missing/replaced'):queue.sampling_ready(tmp_path)
    save(tmp_path/'evaluation-terminal.json',dict(exit_code=1))
    with pytest.raises(RuntimeError,match='failed'):queue.sampling_ready(tmp_path)


@pytest.mark.parametrize('failure',[None,'create_timeout','start_timeout','bad_created_image'])
def test_exact_create_then_record_then_start_without_replay(tmp_path,monkeypatch,failure):
    cid='a'*64;prep=dict(argv=['docker','create','fixture'],image='image',source_commit='commit');calls=[]
    def remote(argv,**kw):
        calls.append(argv)
        if argv[1]=='create':
            assert (tmp_path/'dispatch-intent.json').exists()
            if failure=='create_timeout':raise subprocess.TimeoutExpired(argv,90)
            return SimpleNamespace(stdout=cid+'\n',stderr='')
        assert json.loads((tmp_path/'launch.json').read_text())['container_id']==cid
        assert (tmp_path/'start-intent.json').exists()
        if failure=='start_timeout':raise subprocess.TimeoutExpired(argv,90)
        return SimpleNamespace(stdout=cid+'\n',stderr='')
    monkeypatch.setattr(queue,'remote',remote)
    monkeypatch.setattr(queue,'inspect',lambda _:dict(Image='bad' if failure=='bad_created_image' else 'image',State={'Status':'created'}))
    if failure:
        with pytest.raises((subprocess.TimeoutExpired,ValueError)):queue.dispatch(tmp_path,prep)
    else:assert queue.dispatch(tmp_path,prep)==cid
    count=len(calls)
    with pytest.raises(FileExistsError):queue.dispatch(tmp_path,prep)
    assert len(calls)==count
    if failure=='create_timeout':assert (tmp_path/'create-error.json').exists()
    if failure=='start_timeout':assert (tmp_path/'start-error.json').exists()


@pytest.mark.parametrize('device,privileged,free',[('/dev/davinci6',False,False),('/dev/davinci7',False,False),
    ('/dev/davinci0',False,True),('/dev/davinci0',True,False)])
def test_device_occupancy(device,privileged,free,monkeypatch):
    monkeypatch.setattr(queue,'remote',lambda _:SimpleNamespace(stdout='a'*64))
    monkeypatch.setattr(queue,'inspect',lambda _:dict(HostConfig=dict(Privileged=privileged,Devices=[dict(PathOnHost=device)])))
    assert queue.free_devices() is free


@pytest.mark.parametrize('mutation',[None,'validator_failed','validator_command','acceptance','source','native_failed'])
def test_completed_sampling_requires_accepted_exact_dependencies(tmp_path,monkeypatch,mutation):
    # Synthetic completed dependency envelope; no real inference claimed.
    native=tmp_path/'native';sample='a'*64;validator='b'*64;child='c'*64
    save(tmp_path/'evaluation-launch.json',{})
    save(tmp_path/'evaluation-terminal.json',dict(exit_code=0))
    save(tmp_path/'config.json',{})
    save(tmp_path/'source/runner.py',{'code':'fixture'})
    prep=dict(argv=['docker','run','image','python','sample.py'],source_files={'runner.py':queue.file_hash(tmp_path/'source/runner.py')})
    save(tmp_path/'preparation.json',prep)
    plan=dict(image='image',source_commit='commit',files={k:queue.file_hash(tmp_path/k) for k in ('preparation.json','config.json')},
        acceptance_argv=['docker','run','image','python','verify.py'],dependency=dict(root=str(native),container_children=['test']))
    save(tmp_path/'queue-plan.json',plan)
    save(tmp_path/'sampling-launch.json',dict(container_id=sample))
    save(tmp_path/'acceptance-launch.json',dict(container_id=validator))
    save(tmp_path/'sampling-inspection.json',{'fixture':True})
    save(tmp_path/'run/result.json',dict(status='passed',model_loaded=True))
    save(tmp_path/'run/records.jsonl',{'fixture':True})
    save(tmp_path/'run/terminal.json',dict(exit_code=0))
    acceptance=dict(container_id=sample,image='image',source_commit='commit',responses=156,weight_updates=0,
        result_sha256=queue.file_hash(tmp_path/'run/result.json'),records_sha256=queue.file_hash(tmp_path/'run/records.jsonl'))
    if mutation=='acceptance':acceptance['responses']=155
    save(tmp_path/'acceptance.json',acceptance)
    save(native/'evaluation-terminal.json',dict(exit_code=1 if mutation=='native_failed' else 0))
    save(native/'test/launch.json',dict(container_id=child))
    commands={sample:['python','sample.py'],validator:['python','verify.py','--inspection',str(tmp_path/'sampling-inspection.json'),
        '--inspection-hash',queue.file_hash(tmp_path/'sampling-inspection.json')],child:[]}
    if mutation=='validator_command':commands[validator]=['python','other.py']
    if mutation=='source':(tmp_path/'source/runner.py').write_text('changed')
    def inspect(cid):
        return dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=int(mutation=='validator_failed' and cid==validator),OOMKilled=False),
                    Config=dict(Cmd=commands[cid]))
    original=queue.file_hash
    monkeypatch.setattr(queue,'file_hash',lambda p:queue.SAMPLING_PLAN_SHA if p==tmp_path/'queue-plan.json' else
        queue.SAMPLING_LAUNCH_SHA if p==tmp_path/'evaluation-launch.json' else original(p))
    monkeypatch.setattr(queue,'inspect',inspect)
    if mutation is None:assert queue.sampling_ready(tmp_path) is True
    else:
        with pytest.raises(ValueError):queue.sampling_ready(tmp_path)
