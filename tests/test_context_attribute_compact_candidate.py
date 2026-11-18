import copy
import pytest
from tools.reports.context_attribute_compact_candidate import pack,unpack,render
from tools.reports.build_context_attribute_candidate import catalog,request
from test_context_attribute_candidate import fixture
from multimodal_jev.context_decision_contract import DECISION_TOKEN


@pytest.mark.parametrize('arm',['direct','q_input','q_credit'])
def test_lossless_questions_order_definitions_and_rules(arm):
    req=request('audio.wav',catalog(fixture()),arm=arm)
    value=pack(req)
    assert unpack(req.audio,value).model_input()==req.model_input()
    _,text=render(req)
    assert text.count(DECISION_TOKEN)==len(req.questions)
    assert [g[0] for g in value['groups']]==[q.id for q in req.questions]


def test_q_arms_identical_and_catalog_shared_with_direct():
    menu=catalog(fixture())
    q=pack(request('a',menu,arm='q_input'))
    assert q==pack(request('a',menu,arm='q_credit'))
    d=pack(request('a',menu,arm='direct'))
    for key in ('requirements','events','question','history'):assert q[key]==d[key]
    assert d['groups']==[g for g in q['groups'] if g[0].startswith('final:')]


@pytest.mark.parametrize('index',[-1,999,True])
def test_bad_reference_rejected(index):
    value=pack(request('a',catalog(fixture()),arm='q_input'))
    value['groups'][0][1]=[index]
    with pytest.raises(ValueError):unpack('a',value)
