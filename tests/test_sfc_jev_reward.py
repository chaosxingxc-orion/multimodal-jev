from copy import deepcopy
import json
import pytest

from multimodal_jev.sfc_jev_reward import bundle_rewards
from multimodal_jev.sfc_jev import restore_bundle,split_bundle
from multimodal_jev.sfc_consumption import split_record
from test_sfc_jev_runtime import bundle,predictions


def fixture():
    r,g=bundle();return r,g,predictions(r,g),{r['source_group']}


def test_oracle_rewards_and_fixed_function_normalization():
    r,g,ps,learning=fixture()
    base=bundle_rewards(r,g,ps,learning_groups=learning)
    paired=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    assert base['bundle_mean_reward']==1 and paired['bundle_mean_reward']==pytest.approx(1.2)
    assert paired['correct_function_triples']==2 and paired['native_calls_exact']
    assert [p['decision_id'] for p in paired['decisions']]==[d['id'] for d in r['input']['decisions']]
    assert all(p['function_bonus']==.2 for p in paired['decisions'])


def test_same_correct_endpoint_changes_bonus_only_when_its_peers_correct():
    r,g,ps,learning=fixture()
    good=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    # Flip absence but preserve the exact call-list and presence answers.
    next(p for p in ps if p['decision_id']=='f00.absent')['raw_text']='true'
    bad=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    before=next(p for p in good['decisions'] if p['decision_id']=='f00.calls')
    after=next(p for p in bad['decisions'] if p['decision_id']=='f00.calls')
    assert before['base_reward']==after['base_reward']==1
    assert before['function_bonus']==.2 and after['function_bonus']==0
    assert bad['native_calls_exact']  # Noul errors never gate call reconstruction.
    assert not bad['all_decisions_correct'] and bad['correct_function_triples']==1


def test_consistent_empty_and_all_constant_solutions_do_not_earn_conjunction_bonus():
    r,g,ps,learning=fixture()
    for p in ps:
        p['raw_text']='[]' if p['decision_id'].endswith('.calls') else ('true' if p['decision_id'].endswith('.absent') else 'false')
    result=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    assert result['consistent_function_triples']==2 and result['correct_function_triples']==0
    assert all(p['function_bonus']==0 for p in result['decisions'])
    assert not result['native_calls_exact']
    for p in ps:p['raw_text']='false'
    result=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    assert result['correct_function_triples']==0 and all(p['function_bonus']==0 for p in result['decisions'])


def test_true_no_call_target_is_rewarded_without_confusing_empty_with_unknown():
    r,g,_,_=fixture();native=restore_bundle(r,g)
    native['supervision']={'official_solution':'[]','calls':[]}
    r,g=split_bundle(*split_record(native));ps=predictions(r,g)
    result=bundle_rewards(r,g,ps,learning_groups={r['source_group']},mode='function-conjunction')
    assert result['native_calls_exact'] and result['correct_function_triples']==2
    assert result['bundle_mean_reward']==pytest.approx(1.2)
    for p in ps:p['raw_text']='null'
    invalid=bundle_rewards(r,g,ps,learning_groups={r['source_group']},mode='function-conjunction')
    assert invalid['bundle_mean_reward']==-1 and not invalid['native_calls_exact']


@pytest.mark.parametrize('text',['```json\ntrue\n```','{"x":1,"x":2}','NaN','null','{"execution_success":true}'])
def test_malformed_wrong_type_and_self_report_get_no_bonus(text):
    r,g,ps,learning=fixture()
    for p in ps:p['raw_text']=text
    result=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    assert all(p['total']==-1 for p in result['decisions'])


def test_schema_failure_distinct_from_format_and_wrong_typed_value():
    r,g,ps,learning=fixture()
    call=next(p for p in ps if p['decision_id']=='f00.calls')
    call['raw_text']='[{"undeclared":"x"}]'
    result=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    scored=next(p for p in result['decisions'] if p['decision_id']=='f00.calls')
    assert scored['format_valid'] and not scored['schema_valid']
    assert scored['total']==-.1 and scored['format_penalty']==0
    call['raw_text']='[]'
    result=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    scored=next(p for p in result['decisions'] if p['decision_id']=='f00.calls')
    assert scored['schema_valid'] and scored['total']==0


def test_duplicate_calls_and_nested_array_order_are_semantic():
    r,g,ps,learning=fixture()
    call=next(p for p in ps if p['decision_id']=='f00.calls')
    values=json.loads(call['raw_text']);assert len(values)==2
    call['raw_text']=json.dumps(values[:1])
    result=bundle_rewards(r,g,ps,learning_groups=learning,mode='function-conjunction')
    assert not result['native_calls_exact'] and result['correct_function_triples']==1
    call['raw_text']=json.dumps(list(reversed(values)))
    assert bundle_rewards(r,g,ps,learning_groups=learning)['native_calls_exact']
    values[0]['x'].reverse();call['raw_text']=json.dumps(values)
    assert not bundle_rewards(r,g,ps,learning_groups=learning)['native_calls_exact']


@pytest.mark.parametrize('change',['test','selection','missing','duplicate','binding','gold','mode','weight','bool_weight'])
def test_no_partial_bundle_or_unregistered_training_input(change):
    r,g,ps,learning=fixture();kwargs=dict(learning_groups=learning)
    if change=='test':r['split']='test_id'
    if change=='selection':kwargs['learning_groups']={'another-group'}
    if change=='missing':ps.pop()
    if change=='duplicate':ps.append(deepcopy(ps[0]))
    if change=='binding':ps[0]['request_sha256']='changed'
    if change=='gold':g['answers']['f00.calls']=[]
    if change=='mode':kwargs['mode']='consistency-only'
    if change=='weight':kwargs['bonus_weight']=float('nan')
    if change=='bool_weight':kwargs['bonus_weight']=True
    with pytest.raises(ValueError):bundle_rewards(r,g,ps,**kwargs)
