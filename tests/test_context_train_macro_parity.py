import json
from pathlib import Path

import pytest

from tools.reports import context_train_macro_parity as module


@pytest.mark.parametrize('fault',[None,'difference','wrong-hf'])
def test_real_cache_replica_pool_journals_and_exact_comparison(tmp_path,monkeypatch,fault):
    from test_context_train_parity_inputs import fixture
    from test_context_macro_rollout import Client
    from test_context_replica_macro_rollout import pool
    import multimodal_jev.policy_identity as identities
    import tools.reports.context_macro_evaluation as evaluation
    inputs,_,_,_=fixture(tmp_path,monkeypatch)
    snapshot=tmp_path/'policy';snapshot.mkdir();(snapshot/'MANIFEST.json').write_text('{}')
    commit=tmp_path/'commit.json'
    commit.write_text(json.dumps({'joint_policy_manifest_sha256':module.file_hash(snapshot/'MANIFEST.json')}))
    contract={'base_id':'base','token':{'id':99}};bundle=dict(policy_id='p0',contract=contract)
    runtime={'profile':'fixture','processor_sha256':'a'*64}
    c=dict(pins={str(commit):module.file_hash(commit)},source_policy=str(snapshot),source_policy_id='p0',
        source_contract=contract,initial_adapter='adapter',model='base',base_id='base',stage1_commit=str(commit),
        stage1_endpoint=318,expected_population=40513,train_inputs=inputs,dual_runtime_identity=runtime,release_sha256='c'*64,
        audio_token_id=98,output=str(tmp_path/'result'),macro_client=dict(runtime_identity=runtime,
            max_queries=128,rollout_queries=32,replicas=[{},{}],journal_root=str(tmp_path/'journal'),attempt_id='test'))
    # Model provenance and physical NPU are explicit fixtures; all cache, macro
    # scheduling, persisted journals and independent output comparisons are real.
    monkeypatch.setattr(module,'verify_runtime',lambda c:runtime)
    monkeypatch.setattr(identities,'checkpoint_identity',lambda p:('base',{}))
    monkeypatch.setattr(module,'read_execution_bundle',lambda *a,**kw:bundle)
    monkeypatch.setattr(module,'verify_native_adapter',lambda *a:None)
    monkeypatch.setattr(module,'validate_endpoint',lambda *a:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(evaluation,'check_paths',lambda _:None)
    selected,_=module.panel(c,bundle)
    records=[dict(id=a['id'],request_identity=dict(policy_id='p0',inputs=r['inputs'],positions=r['positions']),
        prob=[.5]*60,logits=[0.]*60) for a,r in zip(selected['assets'],selected['input_receipts'],strict=True)]
    hf=tmp_path/'hf.json'
    hf.write_text(json.dumps(dict(schema='jev-target-width-hf-reference-v1',policy_id='wrong' if fault=='wrong-hf' else 'p0',
        panel_sha256=module.digest(selected),records=records)))
    c['hf_reference']=dict(path=str(hf),sha256=module.file_hash(hf))
    clients=[Client(),Client()]
    engine=pool(tmp_path/'journal/test',monkeypatch,clients)
    engine.runtime_identity=runtime
    for client in clients:
        client.runtime_identity=runtime;original=client.score_many
        def expanded(rows,_original=original,**kw):
            values=_original(rows,**kw)
            for value in values:value.update(prob=[.5]*60,logits=[1e-9 if fault=='difference' else 0.]*60)
            return values
        client.score_many=expanded
    connected=[]
    monkeypatch.setattr(module,'connect_pool',lambda *a,**kw:(connected.append(True) or engine))
    if fault:
        with pytest.raises(ValueError):module.run(c,{},'macro')
        assert json.loads((Path(c['output'])/'terminal.json').read_text())['exit_code']==1
        if fault=='wrong-hf':assert not connected
        else:
            result=json.loads((Path(c['output'])/'comparison.json').read_text())
            assert not result['exact'] and result['probability_max_abs']==0 and result['logit_max_abs']>0
    else:
        result=module.run(c,{},'macro')
        assert result['exact'] and result['queries']==256 and result['decisions']==60
        assert not result['training_dispatch_allowed']
        assert sorted(len(call) for client in clients for call in client.calls)==[32]*8
        assert len(list((tmp_path/'journal/test').glob('evaluation-*/COMPLETE.json')))==2
        assert all(client.phase=='closed' for client in clients)
