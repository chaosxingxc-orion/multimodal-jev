"""Real sealed fixture/commands and lifecycle wiring; no NPU execution."""
import ast
import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_macro_recovery_operations as m
from tools.reports import prepare_context_macro_sparse_resume as prepare
from tools.reports import context_macro_training_operations as shared
from tools.reports.context_macro_probe_operations import IMAGE,inspect_runtime
from test_prepare_context_macro_sparse_resume import fixture as parent_fixture


def fixture(tmp_path,monkeypatch):
    c,parent=parent_fixture(tmp_path,monkeypatch)
    result=prepare.prepare(c,{})
    ops=dict(parent,prepared=c['output'],remote_root=c['remote_root'],
        operations_root=str(tmp_path/'resumed-operations'),manifest_sha256=result['files_sha256'],
        tag='formal-recovery')
    return ops,Path(c['output'])/'package'


def test_real_sparse_package_validates_without_relaxing_finite_entry(tmp_path,monkeypatch):
    c,p=fixture(tmp_path,monkeypatch)
    result=m.validate(c)
    assert result['status']=='macro_sparse_resume_prepared_only'
    assert result['prefix_learning_history_verified'] is True
    with pytest.raises(ValueError):shared.validate(c)


@pytest.mark.parametrize('role',['preflight','client','service-0','service-1'])
def test_exact_recovery_commands_keep_isolation_and_correct_initial_policy(tmp_path,monkeypatch,role):
    c,p=fixture(tmp_path,monkeypatch)
    old=shared.command(c,role);new=m.command(c,role);boundary=new.index(IMAGE)+1
    assert old[:boundary]==new[:boundary]
    if role=='client':assert new.count(c['remote_root']+'/package/tools/reports/context_macro_formal_resume.py')==1
    if role.startswith('service-'):assert new==old
    if role=='preflight':
        ast.parse(new[-1])
        assert 'validate_resume(c);validate_schedule' in new[-1]
        assert "verify_native_adapter(c['reference_adapter'],b)" in new[-1]
        assert "verify_native_adapter(c['initial_adapter'],b)" not in new[-1]
    def values(flag):return [new[i+1] for i,v in enumerate(new) if v==flag]
    obs=dict(Id='b'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=new[boundary:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
            dict(PathOnHost=d.split(':')[0],PathInContainer=d.split(':')[1],CgroupPermissions='rwm')
            for d in values('--device')]))
    inspect_runtime(obs,dict(container_id='b'*64,argv=new),terminal=True)


@pytest.mark.parametrize('fault',['step','history','manifest','recovery_hash','service_batch','service_policy','dependency'])
def test_resealed_mismatches_are_rejected(tmp_path,monkeypatch,fault):
    c,p=fixture(tmp_path,monkeypatch)
    receipt=json.loads((p.parent/'PREPARED.json').read_text())
    if fault=='step':receipt['step']=128
    if fault=='history':receipt['prefix_learning_history_verified']=False
    if fault=='manifest':c['manifest_sha256']='wrong'
    if fault=='recovery_hash':receipt['recovery_sha256']='wrong'
    if fault=='dependency':monkeypatch.setattr(m,'VALIDATOR_SHA256','0'*64)
    if fault.startswith('service_'):
        path=p/'service-0.json';v=json.loads(path.read_text())
        v['policy']['max_batch_queries' if fault=='service_batch' else 'policy_id']=48
        prepare.write(path,v)
        (p/'FILES.json').unlink()
        prepare.write(p/'FILES.json',{str(f.relative_to(p)):prepare.file_hash(f) for f in p.rglob('*') if f.is_file()})
        c['manifest_sha256']=receipt['files_sha256']=prepare.verify(p)
    prepare.write(p.parent/'PREPARED.json',receipt)
    with pytest.raises(ValueError):m.validate(c)


def test_explicit_dispatch_uses_common_lease_lifecycle(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(shared,'run_services',lambda *a,**kw:calls.append((a,kw)) or {'fixture':True})
    with pytest.raises(ValueError):m.run(c,{})
    assert calls==[]
    c['formal_recovery_dispatch_allowed']=True
    assert m.run(c,{})=={'fixture':True}
    args,kw=calls[0]
    assert args[0]['finite_inference_dispatch_allowed'] is True
    assert kw==dict(validator=m.validate,command_builder=m.command,required_devices=(4,5,6,7))


def test_collected_recovery_rejects_reused_parent_container_before_audit(tmp_path,monkeypatch):
    c,p=fixture(tmp_path,monkeypatch);root=Path(c['operations_root']);root.mkdir()
    binding=json.loads((p/'RECOVERY.json').read_text())
    observations={role:dict(Id=str(i)*64) for i,role in enumerate(binding['containers'])}
    observations['client']['Id']=binding['containers']['client']
    prepare.write(root/'collected-inspect.json',observations)
    calls=[]
    monkeypatch.setattr(m.collector,'audit_collected',lambda *a,**kw:calls.append(kw) or {})
    with pytest.raises(ValueError,match='fresh containers'):m.audit_collected(c)
    assert calls==[]
    observations['client']['Id']='c'*64;prepare.write(root/'collected-inspect.json',observations)
    assert m.audit_collected(c)['resumed_runtime_transport_verified'] is True
    assert calls==[dict(validator=m.validate,command_builder=m.command,output_auditor=m.audit_outputs)]


def test_collect_reaudits_full_transport_before_publishing_runtime_claim(tmp_path,monkeypatch):
    c,p=fixture(tmp_path,monkeypatch);root=Path(c['operations_root']);root.mkdir()
    calls=[]
    monkeypatch.setattr(m,'collect_and_audit',lambda *a,**kw:calls.append('collect'))
    def audit(c):calls.append('reaudit');return dict(resumed_runtime_transport_verified=True)
    monkeypatch.setattr(m,'audit_collected',audit)
    assert m.collect(c)['resumed_runtime_transport_verified'] is True
    assert calls==['collect','reaudit'] and (root/'recovery-audit.json').exists()


@pytest.mark.parametrize('fault',[None,'container','command','bytes','archive','driver','missing_role','reused'])
def test_actual_recovery_transport_and_container_checks(tmp_path,monkeypatch,fault):
    c,p=fixture(tmp_path,monkeypatch);root=Path(c['operations_root']);root.mkdir()
    collection=root/'collected';collection.mkdir();(collection/'output.json').write_text('{}')
    (root/'results.tar').write_bytes(b'fixture archive')
    def write(name,value):prepare.write(root/name,value)
    write('terminal.json',dict(exit_code=0))
    write('transport.json',dict(archive_sha256=prepare.file_hash(root/'results.tar'),
        files={'output.json':prepare.file_hash(collection/'output.json')}))
    inspections={}
    for i,role in enumerate(('preflight','client','service-0','service-1')):
        argv=m.command(c,role);identity=str(i+1)*64
        def values(flag):return [argv[j+1] for j,v in enumerate(argv) if v==flag]
        write(role+'-launch.json',dict(container_id=identity,argv=argv))
        inspections[role]=dict(Id=identity,Image=IMAGE,
            State=dict(Status='exited',ExitCode=0,OOMKilled=False),
            Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
            HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
                dict(PathOnHost=d.split(':')[0],PathInContainer=d.split(':')[1],CgroupPermissions='rwm')
                for d in values('--device')]))
    if fault=='container':inspections['client']['HostConfig']['Privileged']=True
    if fault=='missing_role':inspections.pop('service-1')
    if fault=='reused':inspections['client']['Id']='client'
    write('collected-inspect.json',inspections)
    if fault=='command':
        handle=json.loads((root/'client-launch.json').read_text());handle['argv'].append('--extra');write('client-launch.json',handle)
    if fault=='bytes':(collection/'output.json').write_text('changed')
    if fault=='archive':(root/'results.tar').write_bytes(b'changed')
    if fault=='driver':write('terminal.json',dict(exit_code=1))
    calls=[]
    def output(*args):calls.append(args);return dict(status='formal_dp2_recovered_outputs_audited')
    monkeypatch.setattr(m,'audit_outputs',output)
    if fault:
        with pytest.raises(ValueError):m.audit_collected(c)
        assert not calls
    else:
        result=m.audit_collected(c)
        assert len(calls)==1 and result['resumed_runtime_transport_verified'] is True
        assert result['containers']['client']==inspections['client']['Id']
        assert result['transport_sha256']==prepare.file_hash(root/'transport.json')
