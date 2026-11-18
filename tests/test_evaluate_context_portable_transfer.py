import json
from pathlib import Path

import pytest

from tools.reports import evaluate_context_portable_transfer as m
from tools.reports import evaluate_context_transfer_macro as shared
from tools.reports import audit_context_transfer_macro as auditor
from test_context_transfer_macro_evaluation import source_config
from test_context_macro_evaluation import setup


def test_real_macro_client_and_journal_with_portable_input_adapter(tmp_path,monkeypatch):
    c=source_config(tmp_path)
    pool,population,encode,args,clients=setup(tmp_path,monkeypatch,n=5)
    independent=[encode(asset)[1] for asset in population]
    candidates=independent[0]['candidate_ids']
    value=dict(dataset='FSC_native31',population=population,input_receipts=independent,
        candidate_ids=candidates,processor_sha256='fixture',cache_root=str(tmp_path/'cache'))
    monkeypatch.setattr(m,'load',lambda _:value)
    bundle=dict(policy_id='p0',contract=dict(request_version='jev-context-exclusive_event-v1',
        runtime_profile=shared.PROFILE,token=dict(registration_status='admitted',id=151706)))
    monkeypatch.setattr(shared,'read_execution_bundle',lambda *a,**kw:bundle)
    monkeypatch.setattr(shared,'audited_encoder',lambda *a,**kw:encode)
    monkeypatch.setattr(shared,'check_paths',lambda *a:None)
    monkeypatch.setattr(shared,'connect_pool',lambda *a,**kw:pool)
    c.update(dataset='FSC_native31',portable_full_inputs=dict(root='/portable',manifest_sha256='manifest',
        population_evidence_sha256='population',cache_root='/new/cache'),pins={},snapshot='fixture',
        contract=bundle['contract'],base_id='base',output=str(tmp_path/'evaluation'),
        macro_client=dict(runtime_identity=args['runtime_identity']),audio_token_id=9,macro_queries=4)
    result=m.run(c,{})
    assert result['recordings']==5 and result['target_scored'] is False
    proof=m.audit_outputs(c,tmp_path/'evaluation',pool.root)
    assert proof['recordings']==5 and proof['inference_audit']['batches']==2
    assert all(client.phase=='closed' for client in clients)
    p=tmp_path/'evaluation/input-evidence.json';e=json.loads(p.read_text());e['inputs']['manifest_sha256']='changed'
    p.write_text(json.dumps(e))
    with pytest.raises(ValueError,match='saved portable'):m.audit_outputs(c,p.parent,pool.root)


def test_invalid_source_fails_before_portable_inputs_or_engine(tmp_path,monkeypatch):
    c=source_config(tmp_path);c['policy_id']='wrong'
    c['portable_full_inputs']=dict(manifest_sha256='m',population_evidence_sha256='p')
    def forbidden(*a,**kw):pytest.fail('target inputs accessed before source role admission')
    monkeypatch.setattr(m,'load',forbidden)
    monkeypatch.setattr(shared,'connect_pool',forbidden)
    with pytest.raises(ValueError):m.run(c,{})


def test_dataset_release_and_dependency_mismatch_refused(monkeypatch):
    monkeypatch.setattr(m,'load',lambda _:dict(dataset='SLURP_clean_TEST60',source_release={'sha256':'a'}))
    with pytest.raises(ValueError):m.admitted(dict(dataset='FSC_native31',portable_full_inputs={}))
    with pytest.raises(ValueError):m.admitted(dict(dataset='SLURP_clean_TEST60',portable_full_inputs={},source_release={'sha256':'b'}))
    monkeypatch.setitem(m.PINS,'evaluate_context_transfer_macro.py','0'*64)
    with pytest.raises(ValueError):m.bound(shared,'run')
