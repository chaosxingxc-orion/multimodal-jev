from copy import deepcopy

import pytest

from multimodal_jev.sfc_reward_batch import compose
from multimodal_jev.sfc_reward_signal import summarize
from test_sfc_reward_batch import fixture


def test_equal_correct_and_equal_wrong_are_separate():
    args=fixture();batch=compose(**args);report=summarize(args['records'],batch)
    assert report['groups']['equal_reward_all_correct']==12 and report['groups']['contrast_groups']==0
    for row in args['records']:
        row['raw_text']='invalid';row['token_ids']=args['tokenizer'].encode('invalid')
    report=summarize(args['records'],compose(**args))
    assert report['groups']['equal_reward_all_wrong']==12
    assert report['groups']['contrast_groups']==0 and report['all_decisions_correct']==0


def test_relationship_bonus_exposes_peer_contrast_on_correct_call():
    args=fixture();row=args['records'][3];row['raw_text']='true';row['token_ids']=args['tokenizer'].encode('true')
    independent=summarize(args['records'],compose(**{**args,'mode':'independent'}))
    conjunction=summarize(args['records'],compose(**args))
    assert independent['groups']['contrast_groups']==1
    assert conjunction['groups']['contrast_groups']==3
    assert independent['native_calls_exact']==conjunction['native_calls_exact']
    assert independent['decision_types']['calls']['correct']==conjunction['decision_types']['calls']['correct']


@pytest.mark.parametrize('kind',['missing','group','uid','component','nan','summary'])
def test_summary_requires_complete_alignment(kind):
    args=fixture();batch=compose(**args);records=deepcopy(args['records'])
    if kind=='missing':records.pop()
    if kind=='group':batch['prompt_groups']+=1
    if kind=='uid':batch['uid'][0]=batch['uid'][2]
    if kind=='component':batch['components'][0]['decision_id']='other'
    if kind=='nan':batch['rewards'][0]=float('nan')
    if kind=='summary':batch['replicate_summaries'].pop()
    with pytest.raises(ValueError):summarize(records,batch)
