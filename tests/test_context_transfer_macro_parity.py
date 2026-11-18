import copy
import json

import pytest
import torch

from tools.reports import context_transfer_macro_parity as m
from multimodal_jev.context_logits_wire import WIRE


def admitted(dataset='MINDS_native14',n=80):
    width=m.WIDTHS[dataset];candidates=[str(i) for i in range(width)]
    population=[dict(id=f'a{i:03}',audio_sha256=str(i),request_sha256=str(i)) for i in range(n)]
    receipts=[dict(asset,candidate_ids=candidates,positions=list(range(width)),
        inputs={'input_ids':{'shape':[1,100+i]},'input_features':{'shape':[1,128,200+n-i]}})
        for i,asset in enumerate(population)]
    return dict(dataset=dataset,population=population,input_receipts=receipts,candidate_ids=candidates,
        source_rows=[dict(id=r['id'],speaker_id=str(i%20)) for i,r in enumerate(population)],processor_sha256='processor')


@pytest.mark.parametrize('dataset',list(m.WIDTHS))
def test_panel_uses_length_extremes_and_no_gold(dataset):
    a=admitted(dataset);p=m.panel(a)
    assert set(range(16))|set(range(64,80))<=set(p['indices'])
    if dataset=='FSC_native31':assert p['covered_strata']==sorted(str(i) for i in range(20))
    else:assert p['covered_strata']==[] and p['language_coverage_verified'] is False
    for r in a['source_rows']:r.update(gold='changed',intent='bad',prob=[.9])
    assert m.panel(a)==p
    a['input_receipts'][0]['inputs']['input_ids']['shape'][1]=4097
    with pytest.raises(ValueError):m.panel(a)


def reports(dataset='MINDS_native14'):
    p=m.panel(admitted(dataset,3));records=[];hf=[]
    for asset,r in zip(p['assets'],p['input_receipts'],strict=True):
        identity=dict(policy_id='p',positions=r['positions'],inputs=r['inputs'])
        row=dict(id=asset['id'],request_identity=identity,prob=[.5]*len(p['candidate_ids']),logits=[0.]*len(p['candidate_ids']))
        hf.append(copy.deepcopy(row))
        records.append(dict(row,policy_id='p',request_id=m.digest(identity),output_protocol=WIRE))
    return p,dict(schema='jev-target-width-hf-reference-v1',panel_sha256=m.digest(p),policy_id='p',records=hf),records


@pytest.mark.parametrize('dataset',list(m.WIDTHS))
def test_exact_width_comparison(dataset):
    p,hf,vl=reports(dataset);r=m.compare(hf,vl,p,policy_id='p')
    assert r['exact'] and r['queries']==3 and r['decisions']==m.WIDTHS[dataset]
    assert r['runtime_provenance_verified'] is False and r['full_population_parity'] is False


@pytest.mark.parametrize('dataset',['MINDS_native14','SLURP_clean_TEST60'])
def test_new_panel_preserves_entire_existing_256_input_admission(dataset):
    from tools.reports.minds_context_runtime import panel_indices
    a=admitted(dataset,400)
    old=panel_indices(a['source_rows'],[dict(id=r['id'],tokens=r['inputs']['input_ids']['shape'][1])
        for r in a['input_receipts']])
    assert len(old)==256 and set(old)<=set(m.panel(a)['indices'])


def test_actual_logits_detect_difference_even_when_probabilities_round_equal():
    p,hf,vl=reports()
    # Both FP32 sigmoid outputs round to the same value, but actual logits differ.
    a=1e-9;b=2e-9
    hf['records'][0]['logits'][0]=a;vl[0]['logits'][0]=b
    assert torch.tensor(a).sigmoid().item()==torch.tensor(b).sigmoid().item()==.5
    r=m.compare(hf,vl,p,policy_id='p')
    assert r['probability_max_abs']==0 and r['logit_max_abs']>0 and not r['exact']


@pytest.mark.parametrize('fault',['missing','width','policy','panel','identity','nan','order'])
def test_rejects_incomparable_or_corrupt_outputs(fault):
    p,hf,vl=reports()
    if fault=='missing':vl.pop()
    if fault=='width':vl[0]['prob'].pop();vl[0]['logits'].pop()
    if fault=='policy':hf['policy_id']='other'
    if fault=='panel':hf['panel_sha256']='other'
    if fault=='identity':vl[0]['request_identity']={}
    if fault=='nan':hf['records'][0]['logits'][0]=float('nan')
    if fault=='order':vl.reverse()
    with pytest.raises(ValueError):m.compare(hf,vl,p,policy_id='p')


def test_prepare_only_writes_label_free_panel_without_engine(tmp_path,monkeypatch):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(m,'load',lambda c:admitted())
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(m,'read_execution_bundle',lambda *a,**kw:pytest.fail('unexpected model read'))
    c=dict(dataset='MINDS_native14',output=str(tmp_path/'panel'))
    r=m.run(c,{},'prepare')
    assert r['npu_executed'] is False
    assert json.loads((tmp_path/'panel/panel.json').read_text())['target_labels_used'] is False
    with pytest.raises(FileExistsError):m.run(c,{},'prepare')


def test_macro_phase_runs_real_pool_and_independent_journal_audit(tmp_path,monkeypatch):
    from test_context_macro_rollout import Client,pool,rows
    from multimodal_jev import policy_identity
    from multimodal_jev.context_runtime_profile import PROFILE
    from multimodal_jev.span_request_identity import tensor_identity
    from tools.reports import context_macro_client,context_macro_evaluation
    a=admitted(n=5);tensors=rows(5);width=14
    for i,row in enumerate(tensors):
        row['inputs']['input_ids']=torch.full((1,20),i,dtype=torch.long)
        row['inputs']['attention_mask']=torch.ones(1,20,dtype=torch.long)
        row['positions']=torch.arange(width).reshape(1,width)
        a['input_receipts'][i].update(positions=list(range(width)),
            inputs={k:tensor_identity(v) for k,v in row['inputs'].items()})
    selected=m.panel(a);clients=[Client(),Client()]
    for client in clients:
        original=client.score_many
        def expanded(batch,_original=original,**kw):
            result=_original(batch,**kw)
            for r in result:r.update(prob=[.5]*width,logits=[0.]*width)
            return result
        client.score_many=expanded
    engine=pool(tmp_path/'journal/panel',monkeypatch,clients)
    monkeypatch.setattr(m,'load',lambda c:a)
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(context_macro_evaluation,'check_paths',lambda _:None)
    monkeypatch.setattr(policy_identity,'checkpoint_identity',lambda _:('base',{}))
    contract=dict(runtime_profile=PROFILE,request_version='jev-context-exclusive_event-v1',
        token={'registration_status':'admitted','id':151706})
    monkeypatch.setattr(m,'read_execution_bundle',lambda *args,**kw:dict(policy_id='p0',contract=contract))
    monkeypatch.setattr(m,'verify_native_adapter',lambda *args:None)
    def encode(asset):
        i=int(asset['id'][1:]);return tensors[i],a['input_receipts'][i]
    monkeypatch.setattr(m,'audited_encoder',lambda *args,**kw:encode)
    monkeypatch.setattr(context_macro_client,'connect_pool',lambda *args,**kw:engine)
    records=[dict(id=asset['id'],request_identity=dict(policy_id='p0',positions=r['positions'],inputs=r['inputs']),
        prob=[.5]*width,logits=[0.]*width) for asset,r in zip(selected['assets'],selected['input_receipts'],strict=True)]
    reference=tmp_path/'hf.json'
    reference.write_text(json.dumps(dict(schema='jev-target-width-hf-reference-v1',policy_id='p0',
        panel_sha256=m.digest(selected),records=records)))
    c=dict(dataset=a['dataset'],output=str(tmp_path/'output'),model='model',base_id='base',
        snapshot='snapshot',adapter='adapter',contract=contract,policy_id='p0',audio_token_id=9,
        macro_client=dict(journal_root=str(tmp_path/'journal'),attempt_id='panel',runtime_identity={'profile':'fixture'}),
        hf_reference=dict(path=str(reference),sha256=m.file_hash(reference)))
    result=m.run(c,{},'macro')
    assert result['exact'] and result['queries']==5
    proof=json.loads((tmp_path/'output/macro-output-audit.json').read_text())
    assert proof['status']=='complete_macro_evaluation_outputs_audited'
    assert proof['recordings']==5 and all(c.phase=='closed' for c in clients)
