from pathlib import Path

import pytest
import torch

from tools.reports import context_replica_selection_operations as module
from test_prepare_context_cached_training import write


@pytest.mark.parametrize('fault',[None,'positions','input','panel','short'])
def test_full_independent_receipt_binding(tmp_path,monkeypatch,fault):
    from test_context_replica_selection_client import population
    assets,candidates,receipts=population();positions=list(range(60))
    identity=module.tensor_identity(torch.tensor([positions],dtype=torch.long))
    for r in receipts:r['tensors']['positions']=identity
    expected=[dict(a,candidate_ids=candidates,positions=positions,
        inputs={k:v for k,v in r['tensors'].items() if k!='positions'}) for a,r in zip(assets,receipts)]
    package=tmp_path/'package';collection=tmp_path/'collection'
    config=dict(release=str(tmp_path/'release'),dual_runtime_identity={'processor_sha256':'fixture'},cache_audit={},cache_receipts={})
    write(Path(config['release'])/'MANIFEST.json',{})
    monkeypatch.setattr(module.client,'inputs',lambda c:(assets,{},candidates))
    monkeypatch.setattr(module.client,'load_evidence',lambda *a,**k:(receipts,{'fixture':'independent'}))
    selected=module.client.panel(assets,candidates,receipts)
    write(package/'selection-inputs/panel.json',selected)
    lookup={r['id']:r for r in expected};selected['input_receipts']=[lookup[a['id']] for a in selected['assets']]
    for role in ('hf','evaluation'):
        write(collection/role/'panel.json',selected);write(collection/role/'input-evidence.json',{'fixture':'independent'})
    observed=[dict(r) for r in expected]
    if fault=='positions':observed[0]['positions']=[*range(59),61]
    elif fault=='input':receipts[0]['audio_sha256']='wrong'
    elif fault=='panel':write(collection/'hf/panel.json',{})
    elif fault=='short':observed.pop()
    write(collection/'evaluation/inference/input-receipts.json',observed)
    if fault:
        with pytest.raises(ValueError):module.independent_inputs(package,collection,config)
    else:
        assert module.independent_inputs(package,collection,config)==(assets,candidates,expected,selected)


@pytest.mark.parametrize('fault',[None,'parity','journal','score','service'])
def test_replays_actual_macro_journal_and_metric_gate(tmp_path,monkeypatch,fault):
    from test_audit_context_train_parity import outputs
    package,collection=outputs(tmp_path,monkeypatch)
    # Actual 256-row cached tensor/pool/journal fixture; full 9226 population
    # reconstruction is tested independently above, model outputs are stubs.
    selected=module.read(collection/'hf/panel.json')
    monkeypatch.setattr(module,'independent_inputs',lambda *a:(selected['assets'],selected['candidate_ids'],selected['input_receipts'],selected))
    (collection/'parity-journal').rename(collection/'selection-journal')
    c=module.read(package/'parity-macro.json');c.update(policy_id=c['source_policy_id'],contract=c['source_contract'],release='fixture')
    write(package/'parity-macro.json',c)
    m=module.read(package/'PREPARED.json');m['source_step']=318;write(package/'PREPARED.json',m)
    inference=collection/'evaluation/inference';records=module.read(inference/'predictions.json')['records']
    proof=module.client.parity(module.read(collection/'hf/reference.json'),records,selected,c['policy_id'])
    write(collection/'evaluation/parity.json',proof)
    result=module.read(inference/'result.json')
    reported=dict(fixture_score=1,input_receipts_sha256=result['input_receipts_sha256'],
        inference_result_sha256=module.file_hash(inference/'result.json'),parity_sha256=module.file_hash(collection/'evaluation/parity.json'))
    write(collection/'evaluation/result.json',reported)
    monkeypatch.setattr(module.client,'score',lambda *a:{'fixture_score':1})
    def metrics(actual,reported):
        if actual['fixture_score']!=reported['fixture_score']:raise ValueError('score differs')
        return {'fixture':'metric scorer substituted'}
    monkeypatch.setattr(module,'compare_metrics',metrics)
    if fault=='parity':write(collection/'evaluation/parity.json',{'exact':True})
    elif fault=='journal':
        p=collection/'selection-journal/test/evaluation-00000000/INPUT.json';v=module.read(p);v['jobs'][0]['replica_id']=1;write(p,v)
    elif fault=='score':reported['fixture_score']=2;write(collection/'evaluation/result.json',reported)
    elif fault=='service':
        p=collection/'service-a/ready.json';v=module.read(p);v['max_batch_queries']=4;write(p,v)
    if fault:
        with pytest.raises(ValueError):module.audit_outputs(package,collection)
    else:
        result=module.audit_outputs(package,collection)
        assert result['source_step']==318 and result['parity']['exact'] and len(result['dispatch_audit'])==2


@pytest.mark.parametrize('failure',[None,'launch','collection','audit','status'])
def test_single_attempt_collection_then_audit(tmp_path,monkeypatch,failure):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    calls=[];c=dict(selection_dispatch_allowed=True,operations_root=str(tmp_path/'ops'),tag='fixture',selection_package={})
    def launch(c,p):
        calls.append('launch')
        if failure=='launch':raise ValueError('launch failed')
        return dict(status='wrong' if failure=='collection' else 'selection_collected_independent_audit_required')
    def audit(c,p):
        calls.append('audit')
        if failure=='audit':raise ValueError('audit failed')
        return dict(status='wrong' if failure=='status' else 'complete_replica_infonce_selection_audited',source_step=318)
    monkeypatch.setattr(module,'launch',launch);monkeypatch.setattr(module,'audit',audit)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    assert calls==(['launch'] if failure in ('launch','collection') else ['launch','audit'])
    terminal=module.read(Path(c['operations_root'])/'finalization/terminal.json')
    assert terminal['exit_code']==int(failure is not None)
    with pytest.raises(FileExistsError):module.run(c,{})


def test_launch_refuses_without_authorization_or_valid_package(monkeypatch):
    with pytest.raises(ValueError,match='authorization'):module.launch({}, {})
    def refuse(c):raise ValueError('invalid package')
    monkeypatch.setattr(module,'validate',refuse)
    with pytest.raises(ValueError,match='invalid package'):module.launch({'selection_dispatch_allowed':True},{})
