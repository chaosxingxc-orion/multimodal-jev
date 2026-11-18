import json

import pytest

from test_context_reference_builder import setup
from multimodal_jev.context_reference_builder import build
from multimodal_jev.context_reference_audit import audit
from multimodal_jev.context_replica_macro_rollout import MacroRolloutPool
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.span_request_identity import tensor_identity


def run(tmp_path,monkeypatch):
    import multimodal_jev.context_replica_macro_rollout as pool_module
    monkeypatch.setattr(pool_module,'check_paths',lambda _:None)
    kw=setup(monkeypatch);clients=[]
    for _ in range(2):
        engine=type(kw['engine'])();engine.bundle=dict(engine.bundle,contract=kw['identity']['contract']);engine.phase='ready'
        clients.append(engine)
    pool=MacroRolloutPool(clients,tmp_path/'journal',runtime_identity=kw['identity']['runtime_identity'],rollout_queries=1,max_queries=2)
    kw['engine']=pool
    report=build(tmp_path/'reference',**kw)
    bindings={}
    for key,asset in kw['assets'].items():
        request=compile_candidates(asset['audio'],kw['candidates'],[],semantics='exclusive_event')
        value=kw['preparer']((request,asset['audio_sha256']))
        bindings[key]=dict(policy_id=report['identity']['policy_id'],positions=value.positions[0].tolist(),
            inputs={k:tensor_identity(v) for k,v in sorted(value.inputs.items())})
    args=dict(manifest_sha256=report['manifest_sha256'],identity=report['identity'],population=report['population'],
        input_bindings=bindings,macro_queries=2,rollout_queries=1,replicas=2)
    return args


@pytest.mark.parametrize('fault',[None,'missing','probability','input','duplicate','receipt'])
def test_real_macro_journal_reconstructs_cache_and_rejects_corruption(tmp_path,monkeypatch,fault):
    args=run(tmp_path,monkeypatch)
    journal=tmp_path/'journal';batch=journal/'batch-00000000'
    if fault=='missing':(batch/'COMPLETE.json').unlink()
    if fault=='input':args['input_bindings']['s0']['policy_id']='f'*64
    if fault in ('probability','duplicate'):
        p=batch/'rollout-0000.json';row=json.loads(p.read_text())
        if fault=='probability':row['records'][0]['prob'][0]=.5
        else:row['indices']=[1]
        p.write_text(json.dumps(row))
    if fault=='receipt':
        p=batch/'COMPLETE.json';row=json.loads(p.read_text());row['actor_update_performed']=True;p.write_text(json.dumps(row))
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):audit(tmp_path/'reference',journal,**args)
    else:
        report=audit(tmp_path/'reference',journal,**args)
        assert report['recordings']==4 and report['distinct_inputs']==3 and report['macro_batches']==2
