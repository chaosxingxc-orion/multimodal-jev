"""Real package sealing/commands with explicit source/model/input audit substitutes."""
import ast
import copy
import json
from pathlib import Path

import pytest

from test_prepare_context_transfer_parity import setup as parent_fixture
from tools.reports import prepare_context_full_transfer as m
from tools.reports import context_full_transfer_operations as ops
from tools.reports.context_macro_probe_operations import inspect_runtime,IMAGE


def setup(tmp_path,monkeypatch):
    old,source=parent_fixture(tmp_path,monkeypatch)
    parent=json.loads(Path(old['parent_operations']).read_text())
    training=json.loads((source/'training.json').read_text())
    (source/'FILES.json').unlink();m.write(source/'FILES.json',{
        str(p.relative_to(source)):m.file_hash(p) for p in source.rglob('*') if p.is_file()})
    parent['manifest_sha256']=m.verify(source)
    r=json.loads((source.parent/'PREPARED.json').read_text());r['runtime_identity']['processor_sha256']='processor'
    m.write(source.parent/'PREPARED.json',r)
    a=dict(status='source_role_runtime_and_weights_admitted',source_operations=parent,training=training,
        binding=dict(role='source_selected_new_GRPO',policy_id=training['source_policy_id']),
        point=dict(snapshot=str(source/'policy'),adapter=str(source/'adapter'),
            policy_id=training['source_policy_id'],manifest_sha256=m.file_hash(source/'policy/MANIFEST.json')))
    monkeypatch.setattr(m,'admit',lambda _:a);monkeypatch.setattr(ops,'admit',lambda _:a)
    source_entry=dict(path=old['remote_root']+'/package/source/source-evaluation.json',sha256='source')
    def seal(admission,out,remote):
        out.mkdir();m.write(out/'source-evaluation.json',{})
        return source_entry
    monkeypatch.setattr(m,'seal',seal);monkeypatch.setattr(ops,'verify_bundle',lambda *a:source_entry)
    admitted=dict(dataset='MINDS_native14',candidate_ids=list(range(14)),population=[{}]*8168,
        processor_sha256='processor',source_release=None)
    monkeypatch.setattr(m,'load',lambda _:admitted);monkeypatch.setattr(ops,'load',lambda _:admitted)
    monkeypatch.setattr(m,'coverage',lambda *a:None);monkeypatch.setattr(ops,'coverage',lambda *a:None)
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(ops,'read_execution_bundle',lambda *a,**kw:kw)
    monkeypatch.setattr(ops,'verify_native_adapter',lambda *a:None)
    full=tmp_path/'full';full.mkdir();m.write(full/'MANIFEST.json',{})
    c=dict(source_admission=dict(policy_role='source_selected_new_GRPO'),
        project=str(Path.cwd()),output=old['output'],remote_root=old['remote_root'],
        operations_root=old['operations_root'],tag='full-target',portable_parity_inputs=old['portable_parity_inputs'],
        portable_full_inputs=dict(root=str(full),manifest_sha256=m.file_hash(full/'MANIFEST.json'),
            population_evidence_sha256='population',cache_root='/data/cache'))
    return c,source


def prepared(tmp_path,monkeypatch):
    c,source=setup(tmp_path,monkeypatch);m.prepare(c,{})
    root=Path(c['output']);return c,source,json.loads((root/'operations.json').read_text())


def test_sealed_full_target_keeps_base_readonly_and_source_bytes(tmp_path,monkeypatch):
    c,source=setup(tmp_path,monkeypatch)
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=m.prepare(c,{})
    package=Path(c['output'])/'package';o=json.loads((package.parent/'operations.json').read_text())
    assert ops.validate(o)['recordings']==8168 and result['launched'] is False
    assert o['finite_inference_dispatch_allowed'] is False
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert (package/'model_code.py').read_bytes()==before['model_code.py']
    assert (package/'adapter/weights').read_bytes()==b'only delta'
    assert not (package/'training.json').exists()
    with pytest.raises(ValueError):m.prepare(c,{})


@pytest.mark.parametrize('role,device',[('preflight',6),('client',None),('service-0',6),('service-1',7)])
def test_full_commands_hf_first_and_no_labels_or_npu_on_client(tmp_path,monkeypatch,role,device):
    _,_,o=prepared(tmp_path,monkeypatch);argv=ops.command(o,role)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    numbered=[d for d in values('--device') if d.split(':')[0][-1].isdigit()]
    assert numbered==([] if device is None else [f'/dev/davinci{device}:/dev/davinci{device}'])
    if role in ('preflight','client'):
        ast.parse(argv[-1]);assert '/data/cache:/data/cache:ro' in values('-v')
        assert ('context_transfer_macro_parity' if role=='preflight' else 'evaluate_context_portable_transfer') in argv[-1]
    obs=dict(Id='a'*64,Image=IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),Devices=[
            dict(PathOnHost=d.split(':')[0],PathInContainer=d.split(':')[1],CgroupPermissions='rwm')
            for d in values('--device')]))
    inspect_runtime(obs,dict(container_id='a'*64,argv=argv),terminal=True)


@pytest.mark.parametrize('fault',['hf_policy','hf_device','width','service_device','service_setting','pins','cache','source','manifest'])
def test_resealed_full_target_mismatch_refused(tmp_path,monkeypatch,fault):
    c,_,o=prepared(tmp_path,monkeypatch);root=Path(c['output']);p=root/'package'
    if fault.startswith('hf_'):
        path=p/'hf.json';v=json.loads(path.read_text());v['policy_id' if fault=='hf_policy' else 'physical_devices']='bad';m.write(path,v)
    if fault in ('width','service_device','service_setting','pins'):
        path=p/'service-1.json';v=json.loads(path.read_text())
        if fault=='width':v['policy']['max_decisions']=60
        elif fault=='service_device':v['policy']['physical_devices']=[4]
        elif fault=='service_setting':v['policy']['execution_dtype']='float32'
        else:v['pins']={}
        m.write(path,v)
    if fault=='cache':o['cache_root']='/wrong'
    if fault=='source':
        def fail(*a):raise ValueError('source failed')
        monkeypatch.setattr(ops,'admit',fail)
    (p/'FILES.json').unlink();m.write(p/'FILES.json',{str(f.relative_to(p)):m.file_hash(f) for f in p.rglob('*') if f.is_file()})
    r=json.loads((root/'PREPARED.json').read_text());r['files_sha256']=m.verify(p);m.write(root/'PREPARED.json',r)
    o['manifest_sha256']='wrong' if fault=='manifest' else r['files_sha256']
    with pytest.raises(ValueError):ops.validate(o)
