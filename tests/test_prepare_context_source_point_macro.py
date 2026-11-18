"""Real cloned policy/pin packaging with synthetic origin and cache evidence."""
import json
from pathlib import Path

import pytest

from test_prepare_context_macro_replay import fixture as parent_fixture
from tools.reports import prepare_context_selection_macro as shared
from tools.reports import prepare_context_source_point_macro as entry
from tools.reports import context_selection_macro_operations as old_ops
from tools.reports import context_source_point_macro_operations as new_ops
from tools.reports.prepare_context_macro_replay import write
from multimodal_jev import context_policy_artifacts, cached_context_preparation, local_runtime_storage


def setup(tmp_path,monkeypatch,step=64):
    _,ops,_=parent_fixture(tmp_path);parent=Path(ops['prepared']);source=parent/'package'
    t=json.loads((source/'training.json').read_text());t['release']='/shared/release'
    write(source/'training.json',t)
    for directory in ('policy','adapter'):(source/directory).mkdir()
    write(source/'policy/MANIFEST.json',dict(policy_id=t['source_policy_id'],contract=t['source_contract']))
    (source/'adapter/weights').write_bytes(b'initial')
    for i in range(2):
        p=source/f'service-{i}.json';s=json.loads(p.read_text())
        s['policy'].update(model=ops['model'],base_id=t['base_id'])
        write(p,s)
    write(source/'FILES.json',{str(p.relative_to(source)):shared.file_hash(p) for p in source.rglob('*')
        if p.is_file() and p.name!='FILES.json'})
    ops['manifest_sha256']=shared.verify(source)
    r=json.loads((parent/'PREPARED.json').read_text());r['files_sha256']=ops['manifest_sha256']
    r['runtime_identity']['source_manifest_sha256']=shared.file_hash(source/'RUNTIME-SOURCES.json')
    write(parent/'PREPARED.json',r)
    proof=dict(status='formal_dp2_outputs_audited',files_sha256=ops['manifest_sha256'])
    write(tmp_path/'audit.json',proof)
    snapshot=tmp_path/'updated-policy';snapshot.mkdir()
    write(snapshot/'MANIFEST.json',dict(policy_id='updated',contract=t['source_contract']))
    adapter=tmp_path/'updated-adapter';adapter.mkdir();(adapter/'weights').write_bytes(b'updated')
    point=dict(step=step,snapshot=str(snapshot),adapter=str(adapter),policy_id='updated',
        manifest_sha256=shared.file_hash(snapshot/'MANIFEST.json'))
    points=[dict(point,step=s,policy_id=('updated' if s==step else f'other-{s}')) for s in [0,*range(64,634,64),634]]
    plan=dict(schema='jev-macro-source-evaluation-plan-v1',checkpoints=points,source_operations=ops,
        algorithm='dual_credit_grpo_floor_clipped_v1',target_tuning_allowed=False,model_selection_performed=False,
        source_audit=proof,initial_policy_id=t['source_policy_id'],experiment=dict(release_sha256=t['release_sha256']))
    binding=tmp_path/'binding';binding.mkdir();collection=tmp_path/'cache';collection.mkdir()
    original_audit=tmp_path/'original-audit.json';write(original_audit,{})
    original_stream=tmp_path/'original.jsonl';original_stream.write_text('{}\n')
    write(binding/'binding.json',dict(cache_audit=dict(path=str(original_audit),sha256=shared.file_hash(original_audit)),
        cache_receipts=dict(path=str(original_stream),sha256=shared.file_hash(original_stream))))
    write(collection/'input-receipts.json',[])
    cache_proof=dict(image=shared.IMAGE,state=dict(Status='exited',ExitCode=0,OOMKilled=False))
    write(tmp_path/'cache-audit.json',cache_proof)
    monkeypatch.setattr(shared,'audit_receipts',lambda *a:{})
    monkeypatch.setattr(context_policy_artifacts,'read_execution_bundle',lambda *a,**kw:kw)
    monkeypatch.setattr(context_policy_artifacts,'verify_native_adapter',lambda *a:None)
    monkeypatch.setattr(cached_context_preparation,'processor_cache_identity',lambda p:'same')
    monkeypatch.setattr(local_runtime_storage,'check_paths',lambda p:None)
    c=dict(parent=str(parent),parent_manifest_sha256=ops['manifest_sha256'],
        parent_audit=str(tmp_path/'audit.json'),parent_audit_sha256=shared.file_hash(tmp_path/'audit.json'),
        cache_audit=str(tmp_path/'cache-audit.json'),cache_audit_sha256=shared.file_hash(tmp_path/'cache-audit.json'),
        binding_root=str(binding),binding_sha256=shared.file_hash(binding/'binding.json'),
        cache_collection=str(collection),cache_root='/cache',release=t['release'],release_sha256=t['release_sha256'],
        output=str(tmp_path/'evaluation'),remote_root='/local/evaluation',project=str(Path.cwd()))
    return c,point,plan,ops


def test_actual_sealing_changes_policy_and_both_services_preserves_parent(tmp_path,monkeypatch):
    c,point,plan,ops=setup(tmp_path,monkeypatch)
    parent=Path(c['parent'])/'package';old=shared.verify(parent)
    r=shared.prepare(c,{},source_point=point,source_plan=plan);dest=Path(c['output'])/'package'
    evaluation=json.loads((dest/'evaluation.json').read_text())
    assert r['source_step']==64 and r['policy_id']=='updated'
    assert evaluation['policy_id']=='updated'
    assert (dest/'adapter/weights').read_bytes()==b'updated'
    for i in range(2):
        s=json.loads((dest/f'service-{i}.json').read_text())
        assert s['policy']['policy_id']=='updated' and s['policy']['snapshot']==evaluation['snapshot']
        assert s['policy']['physical_devices']==[6+i]
    operations=dict(ops,prepared=c['output'],manifest_sha256=r['files_sha256'],remote_root=c['remote_root'],
        cache_root=c['cache_root'],release=c['release'],release_sha256=c['release_sha256'])
    assert new_ops.validate(operations)['policy_role']=='source_curve_point'
    with pytest.raises(ValueError):old_ops.validate(operations)
    assert shared.verify(parent)==old


@pytest.mark.parametrize('fault',['processor','plan','origin','policy','overlap'])
def test_no_updated_package_with_wrong_origin_or_inputs(tmp_path,monkeypatch,fault):
    c,point,plan,_=setup(tmp_path,monkeypatch)
    if fault=='processor':monkeypatch.setattr(cached_context_preparation,'processor_cache_identity',lambda p:str(p))
    if fault=='plan':plan['checkpoints']=[]
    if fault=='origin':plan['initial_policy_id']='other'
    if fault=='policy':point['manifest_sha256']='other'
    if fault=='overlap':c['remote_root']='/local/prefix/child'
    with pytest.raises(ValueError):shared.prepare(c,{},source_point=point,source_plan=plan)
    assert not Path(c['output']).exists()


def test_public_entry_reaudits_and_rebuilds_plan_before_preparing(tmp_path,monkeypatch):
    c,point,plan,ops=setup(tmp_path,monkeypatch)
    p=tmp_path/'plan.json';write(p,plan)
    c.update(source_plan=str(p),source_plan_sha256=shared.file_hash(p),source_step=64)
    calls=[]
    def audit(o):assert o==ops;calls.append('audit');return plan['source_audit']
    def build(o,a):calls.append('build');return plan
    monkeypatch.setattr(entry,'audit_collected',audit);monkeypatch.setattr(entry,'build',build)
    result=entry.prepare(c,{})
    assert calls==['audit','build'] and result['source_step']==64


def test_public_entry_refuses_changed_plan_after_reaudit(tmp_path,monkeypatch):
    c,point,plan,ops=setup(tmp_path,monkeypatch)
    p=tmp_path/'plan.json';write(p,plan)
    c.update(source_plan=str(p),source_plan_sha256=shared.file_hash(p),source_step=64)
    monkeypatch.setattr(entry,'audit_collected',lambda o:plan['source_audit'])
    monkeypatch.setattr(entry,'build',lambda o,a:dict(plan,initial_policy_id='other'))
    with pytest.raises(ValueError,match='independently audited'):entry.prepare(c,{})
    assert not Path(c['output']).exists()


@pytest.mark.parametrize('fault',['step','algorithm','target','origin','policy','audit','incomplete'])
def test_resealed_wrong_source_metadata_is_rejected(tmp_path,monkeypatch,fault):
    c,point,plan,ops=setup(tmp_path,monkeypatch)
    r=shared.prepare(c,{},source_point=point,source_plan=plan);dest=Path(c['output'])/'package'
    if fault=='step':r['source_step']=128
    if fault=='algorithm':plan['algorithm']='rloo_v1'
    if fault=='target':plan['target_tuning_allowed']=True
    if fault=='origin':plan['source_operations']['manifest_sha256']='other'
    if fault=='policy':plan['checkpoints'][1]['policy_id']='other'
    if fault=='audit':plan['source_audit']['status']='unverified'
    if fault=='incomplete':plan['checkpoints'].pop()
    write(dest/'SOURCE-PLAN.json',plan)
    digest=shared.file_hash(dest/'SOURCE-PLAN.json')
    r['source_plan_sha256']=digest
    for name in ('evaluation.json','service-0.json','service-1.json'):
        v=json.loads((dest/name).read_text());v['pins'][c['remote_root']+'/package/SOURCE-PLAN.json']=digest
        write(dest/name,v)
    write(dest/'FILES.json',{str(p.relative_to(dest)):shared.file_hash(p) for p in dest.rglob('*')
        if p.is_file() and p.name!='FILES.json'})
    r['files_sha256']=shared.verify(dest);write(dest.parent/'PREPARED.json',r)
    operations=dict(ops,prepared=c['output'],manifest_sha256=r['files_sha256'],remote_root=c['remote_root'],
        cache_root=c['cache_root'],release=c['release'],release_sha256=c['release_sha256'])
    with pytest.raises(ValueError,match='source curve'):new_ops.validate(operations)
