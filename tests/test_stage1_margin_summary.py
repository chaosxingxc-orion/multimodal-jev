import copy
import pytest
from tools.reports.summarize_stage1_margin import summarize
from test_context_infonce_margin import O


def records():
    m={'jev/'+k:[v] for k,v in dict(infonce_numerator=4.,probability_margin_numerator=.2,threshold_anchor_numerator=.4,probability_gap_sum=.3,margin_satisfied_count=1.,anchored_count=1.,local_real_queries=2.).items()}
    m['actor/pg_loss']=2.15
    return [dict(update=1,real_queries=8,metrics=copy.deepcopy(m),total_seconds=2.) for _ in range(4)]


def test_all_rank_reduction_and_missing_rank_fails():
    r=records();s=summarize(r,O)
    assert s['loss']==pytest.approx(2.15) and s['anchor_coverage']==.5
    with pytest.raises(ValueError,match='denominator'):summarize(r[:-1],O)


def test_objective_mismatch_is_detected():
    r=records();r[0]['metrics']['actor/pg_loss']=1
    with pytest.raises(ValueError,match='native'):summarize(r,O)
