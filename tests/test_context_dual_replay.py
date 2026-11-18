"""Synthetic audited assets test preparation only, never NPU acceptance."""
import json
from pathlib import Path

import pytest

from tools.reports import prepare_context_dual_replay as m
from tools.reports.context_dual_training_operations import validate,command
from test_context_dual_training_operations import fixture as operations_fixture


def fixture(tmp_path):
    ops=operations_fixture(tmp_path);package=Path(ops['package'])
    config=json.loads((package/'config.json').read_text())
    config.update({k:k for k in ('base_id','source_policy_id','source_contract','runtime_profile',
        'release_sha256','schedule_sha256','behavior_bounds')})
    config.update(stage='task',initial_adapter='/inputs/initial/adapter',reference_adapter='/inputs/initial/adapter')
    (package/'config.json').write_text(json.dumps(config))
    (package/'run.py').write_text("if c['start_update']!=0 or c['stop_update']!=2:\n raise ValueError('cursor')\n")
    manifest=json.loads((package/'PREPARED.json').read_text())
    manifest['files']={str(p.relative_to(package)):m.sha(p) for p in package.rglob('*') if p.is_file() and p!=package/'PREPARED.json'}
    (package/'PREPARED.json').write_text(json.dumps(manifest))
    ops['manifest_sha256']=m.sha(package/'PREPARED.json')
    template=tmp_path/'ops.json';m.write(template,ops)
    local=tmp_path/'local.json';m.write(local,dict(schema='jev-local-macros-v1',macros={}))
    collection=tmp_path/'collection';collection.mkdir()
    for n in ('config.json','service-config.json','PREPARED.json'):
        (collection/n).write_bytes((package/n).read_bytes())
    shards={}
    for rank in range(4):
        p=collection/f'run/rank-{rank}/checkpoint-0001.pt';p.parent.mkdir(parents=True)
        p.write_bytes(b'synthetic checkpoint fixture');shards[str(p.relative_to(collection/'run'))]=m.sha(p)
    policy=collection/'run/policy-0001';policy.mkdir()
    m.write(policy/'MANIFEST.json',dict(policy_id='step1',contract=config['source_contract']))
    (policy/'execution.pt').write_bytes(b'synthetic LoRA/head fixture')
    adapter=collection/'run/adapter-0001';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'adapter')
    m.write(collection/'run/checkpoint-0001-committed.json',dict(updates=1,contract=m.expected_contract(config),
        rank_shards=shards,joint_policy_directory='policy-0001',joint_policy={'policy_id':'step1'},
        joint_policy_manifest_sha256=m.sha(policy/'MANIFEST.json')))
    admission=dict(passed=True,schema='context-dual-prefix-audit-v1',container_id='a'*64,
        manifest_sha256=ops['manifest_sha256'],files={str(p.relative_to(collection)):m.sha(p) for p in collection.rglob('*') if p.is_file()})
    source=dict(operations_config=str(template),operations_local=str(local),collection=str(collection))
    c=dict(output=str(tmp_path/'replay'),remote_root='/run/replay')
    return c,source,admission,ops


def test_replay_preserves_runtime_reference_and_pins_only_delta_state(tmp_path):
    c,source,proof,ops=fixture(tmp_path);result=m.seal(c,{},proof,source)
    p=Path(c['output']);config=json.loads((p/'config.json').read_text());server=json.loads((p/'service-config.json').read_text())
    original=json.loads((Path(ops['package'])/'config.json').read_text())
    assert config['start_update']==1 and config['stop_update']==2
    assert config['resume_root']=='/run/prefix/run'
    assert config['source_policy']==original['source_policy'] and config['reference_adapter']==original['reference_adapter']
    assert config['dual_runtime_identity']==original['dual_runtime_identity']
    assert m.expected_contract(config)==m.expected_contract(original)
    assert server['policy']['policy_id']=='step1'
    assert not list(p.rglob('*.pt')) and not list(p.rglob('*.safetensors'))
    assert (p/'RUNTIME-SOURCES.json').read_bytes()==(Path(ops['package'])/'RUNTIME-SOURCES.json').read_bytes()
    replay_ops=dict(ops,package=str(p),manifest_sha256=m.sha(p/'PREPARED.json'))
    manifest,config,_=validate(replay_ops);argv=command(replay_ops,manifest,config)
    assert '/run/prefix:/run/prefix:ro' in argv
    assert '/run/replay/PARENT-AUDIT.json:/run/replay/PARENT-AUDIT.json:ro' in argv
    assert all(m.sha(p/n)==h for n,h in result['files'].items())


@pytest.mark.parametrize('fault',['shard','contract','proof','runtime_path','source'])
def test_changed_or_unbound_parent_is_rejected(tmp_path,fault):
    c,source,proof,ops=fixture(tmp_path)
    if fault=='shard':(Path(source['collection'])/'run/rank-0/checkpoint-0001.pt').write_bytes(b'changed')
    if fault=='contract':
        p=Path(source['collection'])/'run/checkpoint-0001-committed.json';d=json.loads(p.read_text())
        d['contract']['algorithm']='other';p.write_text(json.dumps(d));proof['files'][str(p.relative_to(source['collection']))]=m.sha(p)
    if fault=='proof':proof['passed']=False
    if fault=='runtime_path':c['remote_root']='/run/prefix/child'
    if fault=='source':(Path(ops['package'])/'source/code.py').write_text('changed')
    with pytest.raises(ValueError):m.seal(c,{},proof,source)


def test_public_prepare_reaudits_before_writing(tmp_path,monkeypatch):
    c,source,proof,ops=fixture(tmp_path)
    monkeypatch.setattr(m,'load_config',lambda *a:(source,{}))
    def reject(_):raise ValueError('parent not actually audited')
    monkeypatch.setattr(m,'audit',reject)
    c.update(parent_audit_config='template',parent_audit_local='local')
    with pytest.raises(ValueError,match='not actually audited'):m.prepare(c,{})
    assert not Path(c['output']).exists()
