import pytest

from tests.test_context_infonce_formal import config
from tools.reports.context_infonce_source_plan import evaluation_protocol
from tools.jev.context_selection_curve import summarize


def test_infonce_curve_uses_full_preregistered_source_population_and_refuses_prefix_selection():
    c=config();c['release_sha256']='release';protocol=evaluation_protocol(c)
    assert protocol['evaluation']['checkpoints']==list(range(0,161,16))
    assert protocol['selection_recordings']==9226
    metrics=dict(queries=9226,candidate_ids=[str(i) for i in range(60)],threshold=.5,
        query_macro_set_f1=.2,top1_event_accuracy=.3,top1_macro_f1=.25,candidate_binary_nll=.5,
        candidate_brier=.1,set_precision=.2,set_recall=.2,exact_set=.2,mean_selected_count=1.,
        binary_nll_infinite=False,set_precision_defined_queries=9226)
    points=[dict(step=s,result=dict(policy_id=str(s),role='selection',release_sha256='release',
        predictions_sha256=str(s),metrics=dict(metrics))) for s in protocol['evaluation']['checkpoints']]
    partial=summarize(points[:3],protocol,initial_policy_id='0')
    assert partial['complete'] is False and partial['selected_checkpoint'] is None
    full=summarize(points,protocol,initial_policy_id='0')
    assert full['complete'] and full['selected_checkpoint']['step']==0
    points[-1]['result']['metrics']['queries']=10
    with pytest.raises(ValueError,match='population'):summarize(points,protocol,initial_policy_id='0')


def test_curve_requires_fixed_selection_rule():
    c=config();c.update(release_sha256='r',selection_rule='best_training_reward')
    with pytest.raises(ValueError,match='selection rule'):evaluation_protocol(c)


def test_new_layout_protocol_ranks_by_event_not_thresholded_set():
    c=config()
    c.update(release_sha256='r',stage1_layout={'design':{'sha256':'a'*64},'plan':{'sha256':'b'*64}})
    e=evaluation_protocol(c)['evaluation']
    assert e['primary']=='top1_macro_f1' and 'query_macro_set_f1' in e['secondary']
    assert e['primary'] not in e['secondary'] and e['checkpoints']==c['evaluation_steps']
