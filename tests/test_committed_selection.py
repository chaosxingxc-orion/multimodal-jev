import json
from pathlib import Path
import pytest
from test_context_policy_artifacts import fixture
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import export_native_adapter,read_execution_bundle
from tools.reports import prepare_committed_selection as module


def setup(tmp_path,monkeypatch):
    _,snapshot,policy,contract=fixture(tmp_path,monkeypatch)
    import shutil
    root=tmp_path/'assets';run=root/'run';run.mkdir(parents=True)
    shutil.copytree(snapshot,run/'policy-0064')
    bundle=read_execution_bundle(snapshot,expected_policy_id=policy,expected_contract=contract,expected_base_id='a'*64)
    import multimodal_jev.context_policy_artifacts as artifacts
    monkeypatch.setattr('multimodal_jev.local_runtime_storage.check_paths',lambda *a:[])
    export_native_adapter(run/'adapter-0064',bundle)
    def write(p,v):p.write_text(json.dumps(v));return dict(path=str(p),sha256=file_hash(p))
    commit=dict(updates=64,joint_policy={'policy_id':policy},joint_policy_directory='policy-0064',joint_policy_manifest_sha256=file_hash(run/'policy-0064/MANIFEST.json'))
    write(run/'policy-0064-committed.json',commit)
    write(run/'checkpoint-0064-committed.json',dict(commit,rank_shards={'rank-0/checkpoint-0064.pt':'x','rank-1/checkpoint-0064.pt':'y'}))
    report=dict(status='sealed_subset_collected_training_audit_pending',container_id='test-container',root=str(root),files={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()})
    return dict(step=64,training_launch=write(tmp_path/'launch.json',dict(container_id='test-container')),assets_report=write(tmp_path/'report.json',report),training_config=write(tmp_path/'config.json',dict(source_contract=contract,base_id='a'*64))),root


def test_checkpoint_authentication_and_readonly_inputs(tmp_path,monkeypatch):
    c,root=setup(tmp_path,monkeypatch);before={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
    training,snapshot,adapter,policy=module.source(c)
    assert snapshot.name=='policy-0064' and adapter.name=='adapter-0064'
    assert before=={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}

@pytest.mark.parametrize('fault',['weights','commit','report','unregistered'])
def test_corrupt_source_rejected(tmp_path,monkeypatch,fault):
    c,root=setup(tmp_path,monkeypatch)
    if fault=='weights':(root/'run/policy-0064/execution.pt').write_bytes(b'bad')
    elif fault=='commit':(root/'run/checkpoint-0064-committed.json').write_text('{}')
    elif fault=='report':Path(c['assets_report']['path']).write_text('{}')
    else:c['step']=65
    with pytest.raises(ValueError):module.source(c)

@pytest.mark.parametrize('fault',['missing_commit','extra_file','wrong_container'])
def test_manifest_scope_rejected(tmp_path,monkeypatch,fault):
    c,root=setup(tmp_path,monkeypatch)
    p=Path(c['assets_report']['path']);r=json.loads(p.read_text())
    if fault=='missing_commit':r['files'].pop('run/checkpoint-0064-committed.json')
    elif fault=='extra_file':(root/'run/adapter-0064/unsealed').write_text('x')
    else:r['container_id']='other'
    p.write_text(json.dumps(r));c['assets_report']['sha256']=file_hash(p)
    with pytest.raises(ValueError):module.source(c)
