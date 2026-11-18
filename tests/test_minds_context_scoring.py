import copy
import json
from pathlib import Path

import pytest
from multimodal_jev.slurp_training import content_hash
from tools.reports import score_minds_context_pair as module


def fixture():
    candidates=[str(i) for i in range(14)]
    variants=['en-US','en-US','en-GB','en-AU','de-DE']
    inputs=[dict(id=str(i),audio_sha256=str(i)) for i in range(5)]
    receipts=[dict(r,positions=[list(range(14))],tensors=dict(input_ids=dict(sha256=r['id']),positions={})) for r in inputs]
    truth={r['id']:dict(intent='0',language=v) for r,v in zip(inputs,variants,strict=True)}
    def predictions(correct):
        result=[]
        for r,receipt,ok in zip(inputs,receipts,correct,strict=True):
            prob=[.01]*14;prob[0 if ok else 1]=.99
            identity=dict(policy_id='policy',positions=receipt['positions'][0],inputs=dict(input_ids=receipt['tensors']['input_ids']))
            result.append(dict(id=r['id'],policy_id='policy',request_identity=identity,
                request_id=content_hash(identity),prob=prob))
        return result
    return candidates,inputs,receipts,truth,predictions


def test_recording_variety_and_language_weights_are_distinct():
    c,r,e,t,p=fixture()
    result=module.score_records(p([1,1,0,0,0]),r,e,t,c,'policy')
    assert result['recording_weighted']['top1_event_accuracy']==.4
    assert result['variety_equal_mean']['top1_event_accuracy']==.25
    assert result['language_equal_mean']['top1_event_accuracy']==.25
    assert result['languages']['en']['queries']==4
    assert result['languages']['en']['top1_event_accuracy']==.5
    assert result['variants']['en-US']['top1_event_accuracy']==1


def test_paired_delta_and_regressions():
    c,r,e,t,p=fixture()
    a=module.score_records(p([1,0,0,0,0]),r,e,t,c,'policy')
    b=module.score_records(p([0,1,1,0,0]),r,e,t,c,'policy')
    diff=module.compare(a,b)
    assert diff['top1_improved']==2 and diff['top1_regressed']==1
    assert diff['recording_weighted_delta']['top1_event_accuracy']==pytest.approx(.2)


@pytest.mark.parametrize('fault',['count','order','gold','nonfinite','candidate'])
def test_partial_or_unbound_scores_rejected(fault):
    c,r,e,t,p=fixture();pred=p([1]*5)
    if fault=='count':pred.pop()
    elif fault=='order':pred.reverse()
    elif fault=='gold':t.pop('0')
    elif fault=='nonfinite':pred[0]['prob'][0]=float('nan')
    else:c.pop()
    with pytest.raises(ValueError):module.score_records(pred,r,e,t,c,'policy')


def test_infinite_nll_is_not_silently_zero_in_group_means():
    c,r,e,t,p=fixture();pred=p([1]*5);pred[0]['prob'][0]=0
    result=module.score_records(pred,r,e,t,c,'policy')
    assert result['recording_weighted']['candidate_binary_nll'] is None
    assert result['variety_equal_mean']['candidate_binary_nll'] is None
    assert result['language_equal_mean']['candidate_binary_nll'] is None


def test_failed_runtime_audit_cannot_open_target_labels(tmp_path,monkeypatch):
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(dict(
        schema='minds-native-descriptor-scoring-v1',threshold=.5,target_tuning=False,
        artifacts=dict(inventory=dict(path=str(tmp_path/'forbidden-labels'),sha256='x')))))
    empty=tmp_path/'empty.json';empty.write_text('{}')
    spec=dict(scoring_contract=module.artifact(contract),package='unused',digest='unused',collection='unused',
        launch=str(empty),observed=str(empty),transport=str(empty))
    def fail(**kw):raise ValueError('actual runtime rejected')
    monkeypatch.setattr(module,'audit',fail)
    with pytest.raises(ValueError,match='actual runtime rejected'):module.run(spec,tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()
