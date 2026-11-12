import copy
import pytest
from multimodal_jev.intent_process_outcomes import summarize,transitions


def row(uid, pred, probabilities=None):
    gold=['alarm','set','alarm_set']
    return dict(id=uid,gold=gold,prediction=pred,gold_probability=probabilities or [.2,.3,.4],
                correct=pred[-1]==gold[-1],compatible=pred[0]+'_'+pred[1]==pred[-1])


def test_correct_final_and_consistent_wrong_are_separate():
    rows=[row('a',['alarm','set','alarm_set']),row('b',['calendar','query','calendar_query']),
          row('c',['alarm','set','calendar_query']),row('d',['calendar','query','alarm_set'])]
    r=summarize(rows)
    assert r['all_three_correct']==1 and r['final_correct']==2
    assert r['compatible']==2 and r['compatible_but_final_wrong']==1
    assert r['both_process_correct_but_final_wrong']==1 and r['final_correct_with_process_error']==1
    assert r['conditional_final']['both_process_correct']['final_accuracy']==.5
    assert r['mean_expected_process_reward']==pytest.approx(.05)
    assert sum(r['correctness_patterns'].values())==4


def test_auxiliary_repair_cannot_be_counted_as_final_gain():
    before=[row('a',['calendar','query','calendar_query']),row('b',['calendar','query','calendar_query']),row('c',['alarm','set','alarm_set'])]
    after=[row('a',['alarm','set','calendar_query']),row('b',['alarm','set','alarm_set']),row('c',['alarm','set','calendar_query'])]
    r=transitions(before,after)
    assert r['final_gains']==r['final_losses']==1
    assert r['both_process_repaired_final_still_wrong']==1
    assert r['both_process_and_final_repaired']==1


def test_empty_conditional_population_is_not_perfect_accuracy():
    r=summarize([row('a',['alarm','set','alarm_set'])])
    assert r['conditional_final']['some_process_wrong']['final_accuracy'] is None


@pytest.mark.parametrize('field,value',[('correct',False),('compatible',False),('gold_probability',[float('nan'),.2,.3]),('gold',['alarm','set','calendar_query'])])
def test_corrupt_semantics_rejected(field,value):
    x=row('a',['alarm','set','alarm_set']);x[field]=value
    with pytest.raises(ValueError):summarize([x])


def test_changed_labels_or_population_cannot_enter_paired_transitions():
    a=row('a',['alarm','set','alarm_set']);b=copy.deepcopy(a);b['id']='b'
    with pytest.raises(ValueError):transitions([a],[b])
    with pytest.raises(ValueError):summarize([a,a])
    b=copy.deepcopy(a);b['gold']=['calendar','query','calendar_query'];b['correct']=False
    with pytest.raises(ValueError):transitions([a],[b])
