import math
import pytest
from multimodal_jev.span_decision_margin import measure,summarize


def test_rank_margin_and_tie_are_distinct():
    a=measure([math.log(.5),math.log(.5)],0)
    b=measure([math.log(.5),math.log(.5)],1)
    assert a['correct'] and not b['correct'] and a['gold_log_margin']==b['gold_log_margin']==0
    assert a['gold_rank']==1 and b['gold_rank']==2
    out=summarize([dict(id='a',**a),dict(id='b',**b)])
    assert out['correct']==1 and out['positive_margin']==0 and out['zero_margin']==2


def test_probability_increase_can_preserve_rank_and_negative_margin():
    a=measure(list(map(math.log,[.8,.1,.1])),1)
    b=measure(list(map(math.log,[.7,.2,.1])),1)
    assert a['gold_rank']==b['gold_rank']==2
    assert a['gold_log_margin']<b['gold_log_margin']<0


@pytest.mark.parametrize('values,target',[([0.,0.],0),([float('nan'),0.],0),([math.log(.5)]*2,True),([math.log(.5)]*2,2)])
def test_rejects_invalid_candidate_support(values,target):
    with pytest.raises(ValueError):measure(values,target)
