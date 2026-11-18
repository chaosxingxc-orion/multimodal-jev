import copy
import json
from pathlib import Path
import pytest
from tools.reports import score_slurp_context_test_pair as module
from multimodal_jev.slurp_training import content_hash


def fixture():
    candidates=[f'c{i}' for i in range(60)]
    inputs=[dict(id=f'u{i}',source_group=f'g{i//2}',audio_sha256=str(i)) for i in range(3)]
    receipts=[dict(id=r['id'],audio_sha256=r['audio_sha256'],positions=[list(range(60))],
        tensors=dict(input_ids={'sha256':r['id']},positions={'sha256':'positions'})) for r in inputs]
    predictions=[]
    for i,(r,t) in enumerate(zip(inputs,receipts)):
        identity=dict(policy_id='p',positions=t['positions'][0],inputs={'input_ids':t['tensors']['input_ids']})
        prob=[.01]*60;prob[i]=.8 if i!=1 else .4
        predictions.append(dict(id=r['id'],policy_id='p',request_identity=identity,request_id=content_hash(identity),prob=prob))
    return dict(predictions=predictions,inputs=inputs,receipts=receipts,targets={r['id']:f'c{i}' for i,r in enumerate(inputs)},
        candidates=candidates,policy_id='p')


def test_independent_sigmoid_empty_set_and_top1_remain_distinct():
    r=module.score_records(**fixture())
    assert r['metrics']['top1_event_accuracy']==1
    assert r['metrics']['query_macro_set_f1']==pytest.approx(2/3)
    assert r['empty_selection_rate']==pytest.approx(1/3)
    assert r['metrics']['candidate_binary_nll']>0 and r['metrics']['candidate_brier']>0


def test_macro_identity_uses_actual_logits_without_rewriting_request_ids():
    import torch
    from multimodal_jev.context_logits_wire import WIRE
    from multimodal_jev.context_macro_rollout import digest
    a=fixture()
    for p in a['predictions']:
        z=torch.tensor(p['prob']).logit()
        p.update(logits=z.tolist(),prob=z.sigmoid().tolist(),output_protocol=WIRE)
    legacy=module.score_records(**a)
    for p in a['predictions']:p['request_id']=digest(p['request_identity'])
    actual=module.score_records(**a,request_identity_protocol='macro_compact_json_v1')
    assert actual==legacy
    with pytest.raises(ValueError):module.score_records(**a)
    a['predictions'][0]['logits'][0]+=1
    with pytest.raises(ValueError):module.score_records(**a,request_identity_protocol='macro_compact_json_v1')


@pytest.mark.parametrize('fault',['missing','duplicate','order','audio','probability','policy','gold'])
def test_incomplete_or_invalid_scoring_fails(fault):
    a=fixture()
    if fault=='missing':a['predictions'].pop()
    elif fault=='duplicate':a['inputs'][1]['id']='u0'
    elif fault=='order':a['predictions'].reverse()
    elif fault=='audio':a['receipts'][1]['audio_sha256']='other'
    elif fault=='probability':a['predictions'][0]['prob'][0]=float('nan')
    elif fault=='policy':a['predictions'][0]['policy_id']='other'
    else:a['targets']['u0']='unknown'
    with pytest.raises(ValueError):module.score_records(**a)


def test_paired_difference_preserves_source_groups_and_no_target_selection():
    a=fixture();before=module.score_records(**a)
    a['predictions'][1]['prob'][1]=.9;after=module.score_records(**a)
    r=module.compare(before,after)
    assert r['delta']['query_macro_set_f1']==pytest.approx(1/3)
    assert r['top1_improved']==r['top1_regressed']==0
    after['records'][0]['source_group']='other'
    with pytest.raises(ValueError):module.compare(before,after)


def test_native_pair_mapping_uses_train_definition_not_raw_alias():
    requests=[dict(id='slurp:a.wav',audio_sha256='h',source_group='slurp:1',input={'audio':'/a.wav'})]
    native=dict(scenario='s',action='a',entities=[])
    labels=[dict(id='slurp:a.wav',target=native,metadata=dict(slurp_id=1,file='a.wav'))]
    metadata=[dict(slurp_id=1,scenario='s',action='a',tokens=[],entities=[],recordings=[{'file':'a.wav'}])]
    population=dict(ids=['slurp:a.wav'],excluded_ids=[])
    inputs=[dict(id='slurp:a.wav',audio_sha256='h',source_group='slurp:1',audio='/a.wav')]
    task=dict(schema={'candidates':[{'id':'train-canonical-name'}]},candidate_pairs={'train-canonical-name':['s','a']})
    assert module.reconstruct_targets(requests,labels,metadata,[],population,inputs,task)=={'slurp:a.wav':'train-canonical-name'}
    labels[0]['target']['action']='other'
    with pytest.raises(ValueError):module.reconstruct_targets(requests,labels,metadata,[],population,inputs,task)


def test_target_artifacts_not_read_before_runtime_audit(tmp_path,monkeypatch):
    c=tmp_path/'contract.json';c.write_text(json.dumps(dict(schema='slurp-context-clean-test-scoring-v1',threshold=.5,
        optimization_allowed=False,artifacts={'native_targets':{'path':'DO_NOT_OPEN','sha256':'none'}})))
    spec=dict(scoring_contract=module.artifact(c),package='fixture',digest='fixture',collection='fixture')
    for key in ('launch','observed','transport'):
        p=tmp_path/(key+'.json');p.write_text('{}');spec[key]=str(p)
    calls=[]
    def reject(**kw):calls.append('audit');raise ValueError('failed actual runtime')
    monkeypatch.setattr(module,'audit',reject)
    with pytest.raises(ValueError,match='failed actual runtime'):module.run(spec,tmp_path/'score.json')
    assert calls==['audit'] and not (tmp_path/'score.json').exists()
