import ast
import copy

import pytest

from tools.reports import context_transfer_parity_operations as m
from tools.reports.context_macro_probe_operations import inspect_runtime,IMAGE


def config():
    return dict(remote_root='/data/user/parity',tag='parity',model='/models/asr',
        driver_root='/driver',cache_root='/data/user/cache')


@pytest.mark.parametrize('role,device',[('preflight',6),('client',None),('service-0',6),('service-1',7)])
def test_exact_device_and_readonly_cache_scope(role,device):
    argv=m.command(config(),role)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    devices=values('--device');binds=values('-v')
    numbered=[p for p in devices if p.split(':')[0][-1].isdigit()]
    assert numbered==([] if device is None else [f'/dev/davinci{device}:/dev/davinci{device}'])
    if role in ('preflight','client'):
        assert '/data/user/cache:/data/user/cache:ro' in binds
        ast.parse(argv[-1])
        assert "'hf'" in argv[-1] if role=='preflight' else "'macro'" in argv[-1]
    else:assert '/data/user/cache:/data/user/cache:ro' not in binds
    obs=dict(Id='a'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=binds,Devices=[
            dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm') for v in devices]))
    handle=dict(container_id='a'*64,argv=argv);inspect_runtime(obs,handle,terminal=True)
    bad=copy.deepcopy(obs);bad['HostConfig']['Binds'].append('/:/host')
    with pytest.raises(ValueError):inspect_runtime(bad,handle)


@pytest.mark.parametrize('path',['relative','/data/user/parity/cache','/data','/foo:bar'])
def test_bad_mount_rejected(path):
    c=config();c['cache_root']=path
    with pytest.raises(ValueError):m.command(c,'preflight')


def test_collector_recomputes_reference_binding_and_rejects_saved_claim(tmp_path,monkeypatch):
    from test_context_transfer_macro_parity import reports
    selected,hf,records=reports()
    package=tmp_path/'package';collection=tmp_path/'collected'
    e=dict(portable_parity_inputs={},policy_id='p',contract={},output='/data/user/parity/evaluation',
        macro_client=dict(attempt_id='panel',runtime_identity={'profile':'fixture'}))
    proof={'status':'complete_macro_evaluation_outputs_audited'}
    result=m.compare(hf,records,selected,policy_id='p')
    report=dict(result,hf_reference={'path':'/data/user/parity/hf/reference.json','sha256':'hash'},
        macro_output_audit_sha256='hash')
    def read(p):
        p=str(p)
        if p.endswith('evaluation.json'):return e
        if p.endswith('terminal.json'):return {'exit_code':0}
        if p.endswith('panel.json'):return selected
        if p.endswith('hf-runtime.json'):
            from multimodal_jev.context_runtime_profile import PROFILE
            return {'profile':PROFILE}
        if p.endswith('ready.json'):
            return dict(physical_devices=[6 if 'service-0' in p else 7],policy_id='p',contract={},
                max_batch_queries=32,model_forward_mode='fixture')
        if p.endswith(('service-0.json','service-1.json')):return {'policy':{'max_batch_queries':32}}
        if p.endswith('reference.json'):return hf
        if p.endswith('predictions.json'):return {'records':records}
        if p.endswith('comparison.json'):return report
        if p.endswith('macro-output-audit.json'):return proof
        raise AssertionError(p)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m,'load_inputs',lambda _:dict(selected_panel=selected))
    monkeypatch.setattr(m,'audit',lambda *a,**kw:proof)
    monkeypatch.setattr(m,'file_hash',lambda _:'hash')
    assert m.audit_outputs(package,collection)['comparison']['exact']
    report['hf_reference']['sha256']='other'
    with pytest.raises(ValueError):m.audit_outputs(package,collection)
