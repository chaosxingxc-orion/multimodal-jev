import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_audio2tool_decode as queue


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.mark.parametrize('fault',[None,'replaced','dead','failed'])
def test_wait_uses_actual_supervisor_identity(tmp_path,monkeypatch,fault):
    root=tmp_path/'audio2tool-jev-engine-capacity-v1'
    save(root/'evaluation-launch.json',dict(pid=2932524,process_start='291613588',queue_plan_sha256=queue.JEV_PLAN))
    original=queue.file_hash
    monkeypatch.setattr(queue,'file_hash',lambda p:queue.JEV_PLAN if p==root/'queue-plan.json' else original(p))
    stat=Path(f'/proc/{os.getpid()}/stat').read_text(); prefix,tail=stat.rsplit(')',1);fields=tail.split()
    fields[19]='wrong' if fault=='replaced' else '291613588'
    if fault=='dead':fields[0]='Z'
    proc=tmp_path/'proc-stat';proc.write_text(prefix+') '+' '.join(fields))
    monkeypatch.setattr(queue,'Path',lambda p:proc if str(p)=='/proc/2932524/stat' else Path(p))
    if fault=='failed':save(root/'evaluation-terminal.json',dict(exit_code=1))
    if fault:
        with pytest.raises(RuntimeError):queue.dependency_ready(tmp_path)
    else:assert queue.dependency_ready(tmp_path) is False


@pytest.mark.parametrize('fault',[None,'native','acceptance','container'])
def test_completed_dependency_rechecks_both_capacities(tmp_path,monkeypatch,fault):
    root=tmp_path/'audio2tool-jev-engine-capacity-v1'
    save(root/'evaluation-launch.json',dict(pid=2932524,process_start='291613588',queue_plan_sha256=queue.JEV_PLAN))
    save(root/'evaluation-terminal.json',dict(exit_code=0));save(root/'launch.json',dict(container_id='a'*64))
    receipt=dict(container_id='a'*64,intent_sha256='intent',result_sha256='result',source_commit='source')
    save(root/'acceptance.json',dict(receipt,result_sha256='other' if fault=='acceptance' else 'result'))
    monkeypatch.setattr(queue,'file_hash',lambda p:queue.JEV_PLAN if p==root/'queue-plan.json' else 'intent')
    monkeypatch.setattr(queue.jev,'check_pins',lambda *a:None)
    monkeypatch.setattr(queue.jev,'native_ready',lambda *a:fault!='native')
    monkeypatch.setattr(queue.native,'inspect',lambda cid:dict(Id='b'*64 if fault=='container' else cid))
    registry=tmp_path/'registry';registry.write_bytes(b'fixture');monkeypatch.setattr(queue.jev,'REGISTRY',str(registry))
    def accept(root,pin,inspection,inspection_pin,cid,*args):
        if json.loads(inspection.read_text())['Id']!=cid:raise ValueError('actual container differs')
        return receipt
    monkeypatch.setattr(queue.jev,'require_capacity',accept)
    if fault:
        with pytest.raises(ValueError):queue.dependency_ready(tmp_path)
    else:assert queue.dependency_ready(tmp_path) is True


@pytest.mark.parametrize('fault',[None,'create_timeout','start_timeout','created_identity'])
def test_one_shot_dispatch_preserves_ambiguous_attempt(tmp_path,monkeypatch,fault):
    cid='a'*64;calls=[]
    monkeypatch.setattr(queue,'capture_capacities',lambda root:queue.write_once(root/'capacity-evidence.json',{'fixture':True}))
    def remote(argv,**kwargs):
        calls.append(argv)
        if argv[1]=='create':
            assert (tmp_path/'intent.json').exists()
            if fault=='create_timeout':raise subprocess.TimeoutExpired(argv,90)
        else:
            assert json.loads((tmp_path/'launch.json').read_text())['container_id']==cid
            assert (tmp_path/'start-intent.json').exists()
            if fault=='start_timeout':raise subprocess.TimeoutExpired(argv,90)
        return SimpleNamespace(stdout=cid+'\n',stderr='')
    monkeypatch.setattr(queue.native,'remote',remote)
    monkeypatch.setattr(queue.native,'inspect',lambda _:dict(Id='b'*64 if fault=='created_identity' else cid,
                      Image='image',State=dict(Status='created')))
    prep=dict(argv=['docker','create','fixture'],image='image',source_commit='commit')
    if fault:
        with pytest.raises((subprocess.TimeoutExpired,ValueError)):queue.dispatch(tmp_path,prep)
    else:assert queue.dispatch(tmp_path,prep)==cid
    count=len(calls)
    with pytest.raises(FileExistsError):queue.dispatch(tmp_path,prep)
    assert len(calls)==count


def test_observation_retry_same_container_and_execution_not_acceptance(tmp_path,monkeypatch):
    cid='a'*64;calls=[];seen=[]
    prep={'image':'image'};plan=dict(dependency_seconds=350000,observation_seconds=7500,resource_lock=str(tmp_path/'device.lock'))
    monkeypatch.setattr(queue,'check_pins',lambda *a:(plan,prep))
    monkeypatch.setattr(queue,'dependency_ready',lambda *a:True)
    monkeypatch.setattr(queue.jev,'free_devices',lambda *a:True)
    def dispatch(*a):
        calls.append(cid);save(tmp_path/'intent.json',{});return cid
    monkeypatch.setattr(queue,'dispatch',dispatch)
    def inspect(current):
        seen.append(current)
        if len(seen)==1:raise subprocess.TimeoutExpired('inspect',30)
        return dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=0))
    monkeypatch.setattr(queue.native,'inspect',inspect)
    monkeypatch.setattr(queue,'check_container',lambda *a:None)
    monkeypatch.setattr(queue.time,'sleep',lambda _:None)
    save(tmp_path/'run/terminal.json',dict(exit_code=0))
    save(tmp_path/'run/result.json',dict(status='profile_complete',model_loaded=True,records=[{}]*16,weight_updates=0,targets_read=False))
    # Only create run artifacts during dispatch; existing run prevents replay.
    initial=tmp_path/'run';pending=tmp_path/'pending';initial.rename(pending)
    def dispatch_with_outputs(*a):pending.rename(initial);return dispatch(*a)
    monkeypatch.setattr(queue,'dispatch',dispatch_with_outputs)
    queue.execute(tmp_path,'pin')
    assert calls==[cid] and seen==[cid,cid]
    assert json.loads((tmp_path/'execution-complete.json').read_text())['accepted'] is False
    with pytest.raises(FileExistsError):queue.execute(tmp_path,'pin')


def test_prior_partial_inspection_prevents_replay(tmp_path,monkeypatch):
    save(tmp_path/'native-capacity-inspection.json',{})
    monkeypatch.setattr(queue,'check_pins',lambda *a:pytest.fail('must stop before observing dependencies'))
    with pytest.raises(FileExistsError):queue.execute(tmp_path,'pin')


@pytest.mark.parametrize('fault',[None,'preparation','source','command','preflight_result','duration','devices','queue_source'])
def test_preparation_binds_real_preflight_and_original_command(tmp_path,monkeypatch,fault):
    root=tmp_path/'profile';prior=tmp_path/'audio2tool-decode-preflight-v1'
    source=prior/'source';source.mkdir(parents=True);(source/'fixture.py').write_text('# frozen source\n')
    image='fixture-image';cmd=['timeout','--signal=TERM','--kill-after=10s','1800s','python3','-B','profile.py','--preflight-only']
    original=dict(argv=['docker','create',image,*cmd],source_files={'fixture.py':queue.file_hash(source/'fixture.py')},source_commit='commit')
    save(prior/'intent.json',original);save(prior/'run/result.json',{'fixture':True})
    monkeypatch.setattr(queue,'PREFLIGHT_INTENT',queue.file_hash(prior/'intent.json'))
    monkeypatch.setattr(queue,'PREFLIGHT_RESULT',queue.file_hash(prior/'run/result.json'))
    changed=cmd[:-1];changed[2]='--kill-after=30s';changed[3]='7200s'
    prep=dict(source_root=str(source),source_files=original['source_files'],source_commit='commit',image=image,
              preflight_container_id=queue.PREFLIGHT_ID,physical_devices=[6,7],wall_seconds=7200,
              automatic_retry=False,weight_updates=0,argv=['docker','create',image,*changed])
    if fault=='duration':prep['wall_seconds']=99999
    if fault=='devices':prep['physical_devices']=[4,5]
    if fault=='command':prep['argv'].append('--preflight-only')
    save(root/'preparation.json',prep)
    coordinator=root/'queue-source/coordinator.py';coordinator.parent.mkdir();coordinator.write_text('# frozen\n')
    plan=dict(schema='audio2tool-decode-queue-v1',automatic_retry=False,dependency_seconds=350000,observation_seconds=7500,
              resource_lock=str(tmp_path/'npu67-sampling-resource.lock'),preparation_sha256=queue.file_hash(root/'preparation.json'),
              source_files={'coordinator.py':queue.file_hash(coordinator)})
    save(root/'queue-plan.json',plan);pin=queue.file_hash(root/'queue-plan.json')
    monkeypatch.setattr(queue.native,'successful',lambda *a:dict(Config=dict(Cmd=cmd)))
    if fault=='source':(source/'fixture.py').write_text('# changed')
    if fault=='queue_source':coordinator.write_text('# changed')
    if fault=='preflight_result':save(prior/'run/result.json',{'changed':True})
    if fault=='preparation':save(root/'preparation.json',{})
    if fault:
        with pytest.raises(ValueError):queue.check_pins(root,pin)
    else:assert queue.check_pins(root,pin)==(plan,prep)
