from copy import deepcopy
import pytest
from multimodal_jev.span_selection_rules import project_selection
from multimodal_jev.slurp_training import content_hash


def fixture():
    tasks={k:dict(schema=dict(id=k,decision_type='noul',question=k,allow_unknown=False),
                  thinking=dict(requirements=['Listen.'])) for k in ('date','time','timeofday')}
    req=dict(id='slurp:x',role='selection',official_split='train',source_group='g',audio_sha256='hash',input=dict(audio='x.wav'))
    target=dict(native_target=dict(entities=[dict(type='date')]))
    packet=[dict(request=req,target=target)]
    entry=dict(id=req['id'],request_sha256=content_hash(req),native_target_sha256=content_hash(target),
        task_sha256={k:content_hash(v) for k,v in tasks.items()},reviewer='fixture',note='fixture',evidence=['fixture'],
        status=dict(date='reviewed',time='conflict',timeofday='reviewed'),
        reviewed_values=dict(date=True,time=None,timeofday=False))
    return packet,dict(schema='jev-selection-reference-review-v1',records=[entry],scope='fixture reference review'),tasks


def test_selection_never_becomes_training_and_masks_all_dependent_rules():
    rows,audit=project_selection(*fixture())
    assert len(rows)==18 and len(audit['excluded'])==6
    assert all(r['split']=='selection' and r['optimization_allowed'] is False for r in rows)
    assert all(r['provenance']['dependency_keys']==['date','timeofday'] for r in rows)
    assert all('reference' not in r['request'] and 'targets' not in r['request'] for r in rows)


@pytest.mark.parametrize('fault',['learning','stale','wrong_gold','masked_gold','missing_review'])
def test_reject_invalid_selection_review(fault):
    packet,review,tasks=deepcopy(fixture());entry=review['records'][0]
    if fault=='learning':packet[0]['request']['role']='learning'
    elif fault=='stale':entry['task_sha256']['date']='wrong'
    elif fault=='wrong_gold':entry['reviewed_values']['date']=False
    elif fault=='masked_gold':entry['reviewed_values']['time']=False
    else:review['records']=[]
    with pytest.raises(ValueError):project_selection(packet,review,tasks)
