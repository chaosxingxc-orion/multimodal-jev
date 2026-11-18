"""Repeated production audit replays actual reference tables and macro journals."""
import json

import pytest

from tools.reports import prepare_context_reference_job as module


def fixture(tmp_path,monkeypatch):
    from test_context_reference_audit import run
    import tools.reports.audit_context_reference_dataset as auditor
    import tools.reports.context_reference_input_evidence as evidence
    import multimodal_jev.cached_context_preparation as processor
    args=run(tmp_path,monkeypatch)
    root=tmp_path/'build';root.mkdir();(tmp_path/'reference').rename(root/'dataset')
    identity=args['identity'];policy=tmp_path/'policy';policy.mkdir()
    (policy/'MANIFEST.json').write_text('{}')
    commit=tmp_path/'commit.json'
    commit.write_text(json.dumps({'joint_policy_manifest_sha256':module.file_hash(policy/'MANIFEST.json')}))
    build=dict(source_policy_id=identity['policy_id'],source_contract=identity['contract'],base_id=identity['base_id'],
        release='/fixture/release',release_sha256=identity['release_sha256'],stage1_commit=str(commit),
        pins={str(commit):module.file_hash(commit)},macro_client=dict(runtime_identity=identity['runtime_identity'],
            max_queries=2,rollout_queries=1,replicas=[{},{}]))
    config=tmp_path/'build-config.json';config.write_text(json.dumps(build))
    result=dict(identity=identity,population=args['population'],recordings=4,manifest_sha256=args['manifest_sha256'])
    (root/'result.json').write_text(json.dumps(result));(root/'terminal.json').write_text('{"exit_code":0}')
    stream=tmp_path/'receipts.jsonl';stream.write_text('fixture')
    report=tmp_path/'report.json';report.write_text('{"original_report":"retain"}')
    # Fixture bypasses NPU provenance and full native60 input parsing only.
    # The real endpoint bindings, source hashes, publication table and complete
    # macro reconstruction in the public audit still execute below.
    monkeypatch.setattr(auditor,'read_execution_bundle',lambda *a,**kw:{'policy_id':identity['policy_id']})
    monkeypatch.setattr(auditor,'validate_endpoint',lambda *a:None)
    monkeypatch.setattr(auditor,'load',lambda *a,**kw:(list(args['population']),
        {'schema':{'candidates':[{'id':v,'description':v} for v in identity['candidate_ids']]}},{}))
    monkeypatch.setattr(evidence,'load',lambda *a,**kw:({}, {'fixture':'input-proof'}))
    monkeypatch.setattr(evidence,'bindings_from_records',lambda *a,**kw:args['input_bindings'])
    monkeypatch.setattr(processor,'processor_cache_identity',lambda *a:'processor')
    c=dict(build_config=str(config),build_config_sha256=module.file_hash(config),build_output=str(root),
        stage1_policy=str(policy),stage1_commit=str(commit),stage1_commit_sha256=module.file_hash(commit),
        input_receipts=str(stream),input_receipts_sha256=module.file_hash(stream),input_proof_sha256='proof',
        journal=str(tmp_path/'journal'),output=str(report))
    return c,report


def test_repeated_reaudit_is_readonly_and_replays_current_evidence(tmp_path,monkeypatch):
    c,report=fixture(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    a=module.replay_reference_outputs(c,{'caller':'prepare'})
    b=module.replay_reference_outputs(c,{'caller':'audit'})
    assert a['status']==b['status']=='reference_dataset_journal_audited'
    assert a['recordings']==4 and a['distinct_inputs']==3
    assert {k:v for k,v in a.items() if k!='provenance'}=={k:v for k,v in b.items() if k!='provenance'}
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    # A previous successful report must never hide later evidence corruption.
    complete=tmp_path/'journal/batch-00000000/COMPLETE.json';v=json.loads(complete.read_text())
    v['actor_update_performed']=True;complete.write_text(json.dumps(v))
    with pytest.raises(ValueError):module.replay_reference_outputs(c,{})
    assert report.read_bytes()==before[str(report)]


def test_explicit_auditor_publication_still_rejects_overwrite(tmp_path,monkeypatch):
    from tools.reports.audit_context_reference_dataset import run
    c,report=fixture(tmp_path,monkeypatch);before=report.read_bytes()
    with pytest.raises(FileExistsError):run(c,{})
    assert report.read_bytes()==before
