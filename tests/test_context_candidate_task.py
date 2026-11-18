import pytest

from multimodal_jev.context_candidate_task import compile_candidates,bind_candidate_targets


def request(semantics='exclusive_event',candidates=None):
    return compile_candidates('/a.wav',candidates or [dict(id='x',definition='Alarm request'),
        dict(id='y',definition='Music request')],['Use the supplied ontology.'],semantics=semantics)


def test_target_changes_never_change_input_and_missing_is_not_negative():
    r=request();original=r.model_input()
    for event,labels in [('x',[True,False]),('y',[False,True]),(None,[None,None])]:
        t=bind_candidate_targets(r,semantics='exclusive_event',exclusive_event=event)
        assert t['labels']==labels and r.model_input()==original
    assert t['supervised']==[False,False]


def test_multilabel_and_empty_sets_preserved_without_exclusive_conversion():
    r=request('candidate_applicability')
    for values in [dict(x=True,y=True),dict(x=False,y=False),dict(x=None,y=True)]:
        t=bind_candidate_targets(r,semantics='candidate_applicability',applicability=values)
        assert t['labels']==list(values.values())
    with pytest.raises(ValueError):
        bind_candidate_targets(r,semantics='candidate_applicability',exclusive_event='x')


def test_order_and_id_rebinding_follow_definitions():
    r=request();a=bind_candidate_targets(r,semantics='exclusive_event',exclusive_event='x')
    swapped=request(candidates=[dict(id='new-y',definition='Music request'),dict(id='new-x',definition='Alarm request')])
    b=bind_candidate_targets(swapped,semantics='exclusive_event',exclusive_event='new-x')
    assert a['labels']==b['labels'][::-1]
    assert a['request_sha256']!=b['request_sha256']
    assert r.questions[0].question==swapped.questions[1].question


@pytest.mark.parametrize('fault',['cross_semantics','unknown','missing_key','integer'])
def test_wrong_semantics_or_incomplete_labels_fail(fault):
    with pytest.raises(ValueError):
        if fault=='cross_semantics':bind_candidate_targets(request(),semantics='candidate_applicability',applicability=dict(x=True,y=False))
        elif fault=='unknown':bind_candidate_targets(request(),semantics='exclusive_event',exclusive_event='z')
        else:bind_candidate_targets(request('candidate_applicability'),semantics='candidate_applicability',
                                    applicability=dict(x=True) if fault=='missing_key' else dict(x=1,y=False))
