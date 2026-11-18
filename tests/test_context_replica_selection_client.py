import copy
from pathlib import Path

import pytest

from tools.reports import context_replica_selection_client as module
from test_prepare_context_cached_training import write


def population():
    assets=[dict(id=f's{i:05}',audio_sha256=f'a{i}',request_sha256=f'r{i}') for i in range(9226)]
    receipts=[dict(a,tensors=dict(input_ids={'shape':[1,100+i%3900]},
        input_features={'shape':[1,128,10+i]},positions={'shape':[1,60]})) for i,a in enumerate(assets)]
    return assets,[str(i) for i in range(60)],receipts


def test_full_selection_panel_is_deterministic_input_only_and_preserves_extremes():
    assets,candidates,receipts=population();p=module.panel(assets,candidates,receipts)
    assert len(p['assets'])==256 and p['labels_used'] is False
    chosen={a['id'] for a in p['assets']}
    assert {a['id'] for a in assets[:16]+assets[-16:]}<=chosen
    for r in receipts:r['gold']='arbitrary'
    # Panel receipt digest authenticates evidence; selection itself uses only
    # lengths and IDs even if unrelated metadata is supplied by a caller.
    changed=module.panel(assets,candidates,receipts)
    assert changed['assets']==p['assets'] and changed['selection']['lengths']==p['selection']['lengths']
    assert module.panel(list(reversed(assets)),candidates,list(reversed(receipts)))['assets']==p['assets']


@pytest.mark.parametrize('fault',['partial','duplicate','order','gold','width','shape'])
def test_invalid_population_cannot_define_panel(fault):
    assets,candidates,receipts=population()
    if fault=='partial':assets.pop()
    elif fault=='duplicate':assets[-1]=assets[0]
    elif fault=='order':receipts.reverse()
    elif fault=='gold':assets[0]['gold']='c'
    elif fault=='width':candidates.pop()
    else:receipts[0]['tensors']['input_ids']['shape']=[1,9999]
    with pytest.raises(ValueError):module.panel(assets,candidates,receipts)


def test_saturated_probability_cannot_hide_logit_divergence():
    from test_context_transfer_macro_parity import reports
    panel,ref,records=reports('SLURP_clean_TEST60')
    assert module.parity(ref,list(reversed(records)),panel,'p')['exact']
    records[0]['logits'][0]=1e-9
    with pytest.raises(ValueError,match='exact'):module.parity(ref,records,panel,'p')
    with pytest.raises(ValueError,match='unique'):module.parity(ref,records+records[:1],panel,'p')


@pytest.mark.parametrize('fault',[None,'parity','hf-policy','partial','score'])
def test_scores_only_after_same_policy_full_population_and_exact_parity(tmp_path,monkeypatch,fault):
    from multimodal_jev import local_checkpoint
    from test_context_transfer_macro_parity import reports
    selected,reference,records=reports('SLURP_clean_TEST60')
    assets=selected['assets'];candidates=selected['candidate_ids'];calls=[]
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'validate_config',lambda c:{'fixture':'NPU bundle substituted'})
    monkeypatch.setattr(module,'prepare_inputs',lambda *a:(assets,candidates,selected,None,{}))
    class Pool:
        def close(self):calls.append('close')
    monkeypatch.setattr(module,'connect_pool',lambda *a,**k:(calls.append('connect') or Pool()))
    def evaluate(*a,**kw):
        calls.append('evaluate');r=copy.deepcopy(records)
        if fault=='parity':r[0]['logits'][0]=1e-9
        if fault=='partial':r.pop()
        write(kw['output']/'predictions.json',{'records':r});write(kw['output']/'result.json',{})
        return dict(predictions_sha256=module.file_hash(kw['output']/'predictions.json'),input_receipts_sha256='fixture')
    monkeypatch.setattr(module,'evaluate',evaluate)
    def score(*a):
        calls.append('score')
        if fault=='score':raise ValueError('scorer failed')
        assert a[-1]=='selection'
        return {'fixture':'not a real score'}
    monkeypatch.setattr(module,'score',score)
    if fault=='hf-policy':reference['policy_id']='other'
    hf=tmp_path/'hf.json';write(hf,reference)
    config=dict(output=str(tmp_path/'out'),policy_id='p',dual_runtime_identity={'fixture':True},
        macro_client={},audio_token_id=1,release='release',release_sha256='release-hash',
        hf_reference=dict(path=str(hf),sha256=module.file_hash(hf)))
    if fault:
        with pytest.raises(ValueError):module.run(config,{},'macro')
    else:module.run(config,{},'macro')
    expected=[] if fault=='hf-policy' else ['connect','evaluate','close']
    if fault in (None,'score'):expected+=['score']
    assert calls==expected
    assert (Path(config['output'])/'result.json').exists()==(fault is None)
    assert module.read(Path(config['output'])/'terminal.json')['exit_code']==int(fault is not None)


@pytest.mark.parametrize('fault',['schema','runtime','policy','macro','replicas'])
def test_legacy_or_mismatched_transport_refused_before_model_read(fault,monkeypatch):
    c=dict(macro_queries=128,policy_id='p',source_policy_id='p',source_policy='policy',snapshot='policy',
        source_contract={},contract={},dual_runtime_identity={'fixture':True},
        macro_client=dict(schema='jev-replica-macro-client-v1',max_queries=128,rollout_queries=32,
            replicas=[{},{}],runtime_identity={'fixture':True}))
    if fault=='schema':c['macro_client']['schema']='jev-macro-client-v1'
    elif fault=='runtime':c['macro_client']['runtime_identity']={}
    elif fault=='policy':c['source_policy_id']='other'
    elif fault=='macro':c['macro_queries']=32
    else:c['macro_client']['replicas'].pop()
    monkeypatch.setattr(module,'read_execution_bundle',lambda *a,**kw:pytest.fail('unexpected model read'))
    with pytest.raises(ValueError):module.validate_config(c)
