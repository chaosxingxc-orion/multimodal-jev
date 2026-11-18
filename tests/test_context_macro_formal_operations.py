import ast
import copy
import json

import pytest

from test_context_macro_training_operations import fixture
from test_context_macro_formal_candidate import config as formal_config
from tools.reports import context_macro_formal_operations as formal
from tools.reports import context_macro_training_operations as finite
from tools.reports.context_macro_probe_operations import inspect_runtime, IMAGE


@pytest.mark.parametrize('role',['preflight','client','service-0','service-1'])
def test_formal_commands_preserve_all_device_storage_and_image_arguments(tmp_path,role):
    c,_=fixture(tmp_path)
    old=finite.command(c,role);new=formal.command(c,role)
    boundary=new.index(IMAGE)+1
    assert new[:boundary]==old[:boundary]
    if role.startswith('service-'):assert new==old
    if role=='client':
        assert new.count(c['remote_root']+'/package/tools/reports/context_macro_formal_candidate.py')==1
        assert '--nproc-per-node=2' in new
    if role=='preflight':
        ast.parse(new[-1]);assert 'contract(c);validate_schedule' in new[-1]
        assert 'context_macro_formal_candidate import generate' in new[-1]
    def values(flag):return [new[i+1] for i,v in enumerate(new) if v==flag]
    obs=dict(Id='a'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=new[boundary:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
            dict(PathOnHost=d.split(':')[0],PathInContainer=d.split(':')[1],CgroupPermissions='rwm')
            for d in values('--device')]))
    inspect_runtime(obs,dict(container_id='a'*64,argv=new),terminal=True)
    bad=copy.deepcopy(obs);bad['HostConfig']['Devices'].append(
        dict(PathOnHost='/dev/davinci0',PathInContainer='/dev/davinci0',CgroupPermissions='rwm'))
    with pytest.raises(ValueError):inspect_runtime(bad,dict(container_id='a'*64,argv=new),terminal=True)


def package(tmp_path,monkeypatch):
    c,t=fixture(tmp_path);c.update(host='npu',manifest_sha256='fixture')
    root=c['remote_root'];p=tmp_path/'package';runtime={'profile':'fixture'}
    t.update(formal_config(),source_policy_id='p0',physical_devices=[4,5],global_batch=128,microbatch=16,
        source_root=root+'/package',source_policy=root+'/package/policy',source_contract={},
        initial_adapter=root+'/package/adapter',reference_adapter=root+'/package/adapter',
        reference_cache=root+'/reference-cache',model=c['model'],dual_runtime_identity=runtime,
        macro_client=dict(max_queries=128,rollout_queries=32,runtime_identity=runtime,
            journal_root=root+'/journal',attempt_id='formal-source-v1',replicas=[
                dict(socket=root+f'/ipc/r{i}.sock',exchange=root+f'/ipc/e{i}') for i in range(2)]))
    receipt=dict(status='macro_formal_prepared_only',image=IMAGE,files_sha256='fixture',
        remote_root=root,ipc_root=root+'/ipc',training_dispatch_allowed=False,runtime_identity=runtime,
        formal_launch_admitted=False,selected_microbatch=16)
    for i in range(2):
        endpoint=t['macro_client']['replicas'][i]
        service=dict(policy=dict(physical_devices=[6+i],snapshot=t['source_policy'],
            adapter=t['initial_adapter'],policy_id='p0',contract={}),dual_runtime_identity=runtime,
            socket_path=endpoint['socket'],exchange=endpoint['exchange'])
        (p/f'service-{i}.json').write_text(json.dumps(service))
    (p/'training.json').write_text(json.dumps(t))
    (p/'FILES.json').write_text(json.dumps({'code.py':'hash'}))
    (tmp_path/'PREPARED.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(finite,'verify',lambda p:'fixture')
    return c,t,receipt


def test_formal_package_is_rejected_by_finite_launcher_and_admitted_by_explicit_validator(tmp_path,monkeypatch):
    c,_,_=package(tmp_path,monkeypatch)
    assert formal.validate(c)['selected_microbatch']==16
    with pytest.raises(ValueError):finite.validate(c)
    with pytest.raises(ValueError,match='command builder'):finite.validate(c,formal=True)


@pytest.mark.parametrize('fault',['micro','budget','flag','service_policy','service_device','runtime','identity'])
def test_mismatched_formal_package_cannot_validate(tmp_path,monkeypatch,fault):
    c,t,r=package(tmp_path,monkeypatch);p=tmp_path/'package'
    if fault=='micro':r['selected_microbatch']=32
    if fault=='budget':t['stop_update']=100
    if fault=='flag':r['formal_launch_admitted']=True
    if fault=='runtime':t['dual_runtime_identity']={'other':True}
    if fault=='identity':c['manifest_sha256']='wrong'
    if fault.startswith('service_'):
        path=p/'service-1.json';service=json.loads(path.read_text())
        if fault=='service_policy':service['policy']['policy_id']='other'
        else:service['policy']['physical_devices']=[4]
        path.write_text(json.dumps(service))
    (p/'training.json').write_text(json.dumps(t));(tmp_path/'PREPARED.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):formal.validate(c)
