"""Actual transport/container checks around explicitly selected output auditor."""
import copy
import json
from pathlib import Path

import pytest

from tools.reports import audit_context_macro_training as audit
from tools.reports import context_macro_formal_operations as formal
from tools.reports.context_macro_probe_operations import IMAGE
from test_context_macro_training_operations import fixture


def setup(tmp_path):
    c,_=fixture(tmp_path);root=tmp_path/'ops';root.mkdir()
    c['operations_root']=str(root)
    collection=root/'collected';collection.mkdir();(collection/'evidence.json').write_text('{}')
    (root/'results.tar').write_bytes(b'synthetic transport')
    def write(name,value):(root/name).write_text(json.dumps(value))
    write('terminal.json',{'exit_code':0})
    write('transport.json',dict(archive_sha256=audit.file_hash(root/'results.tar'),
        files={'evidence.json':audit.file_hash(collection/'evidence.json')}))
    inspections={}
    for i,role in enumerate(('preflight','client','service-0','service-1')):
        argv=formal.command(c,role);identity=str(i+1)*64
        def values(flag):return [argv[j+1] for j,v in enumerate(argv) if v==flag]
        write(role+'-launch.json',dict(container_id=identity,argv=argv))
        inspections[role]=dict(Id=identity,Image=IMAGE,
            State=dict(Status='exited',ExitCode=0,OOMKilled=False),
            Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
            HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
                dict(PathOnHost=d.split(':')[0],PathInContainer=d.split(':')[1],CgroupPermissions='rwm')
                for d in values('--device')]))
    write('collected-inspect.json',inspections)
    return c,root,write,inspections


@pytest.mark.parametrize('fault',[None,'container','command','bytes','archive','driver'])
def test_formal_reaudit_keeps_actual_transport_and_inspection_gates(tmp_path,monkeypatch,fault):
    c,root,write,inspections=setup(tmp_path);calls=[]
    monkeypatch.setattr(formal,'validate',lambda c:dict(files_sha256='fixture',image=IMAGE))
    def output(p,r):calls.append((p,r));return dict(status='formal_dp2_outputs_audited')
    monkeypatch.setattr(formal,'audit_outputs',output)
    if fault=='container':
        inspections['client']['HostConfig']['Privileged']=True;write('collected-inspect.json',inspections)
    if fault=='command':
        handle=json.loads((root/'client-launch.json').read_text());handle['argv'].append('--extra');write('client-launch.json',handle)
    if fault=='bytes':(root/'collected/evidence.json').write_text('changed')
    if fault=='archive':(root/'results.tar').write_bytes(b'changed')
    if fault=='driver':write('terminal.json',{'exit_code':1})
    if fault:
        with pytest.raises(ValueError):formal.audit_collected(c)
        assert not calls
    else:
        r=formal.audit_collected(c)
        assert len(calls)==1 and r['status']=='formal_dp2_outputs_audited'
        assert r['containers']['client']==inspections['client']['Id']
        assert r['transport_sha256']==audit.file_hash(root/'transport.json')
