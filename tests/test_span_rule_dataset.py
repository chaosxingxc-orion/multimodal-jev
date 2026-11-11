import itertools
import json
import pytest
from multimodal_jev.span_rule_dataset import build_rule_controls,predicted_fact_rule
from multimodal_jev.span_contract import SpanQuestion,SpanRequest
from multimodal_jev.task_schema import DecisionSchema


def bundle(values):
    questions=tuple(SpanQuestion(k,DecisionSchema('fact','noul',f'Does the audio contain {k}?',allow_unknown=False)) for k in ('date','time','timeofday'))
    return dict(id=str(values),request=SpanRequest('audio.wav',questions,'timeofday',('Listen.',)),
        targets=list(values),supervised=[True]*3,source_group=str(values),audio_sha256='audio',decision_ids=['labels'])


def test_all_truth_signatures_and_matched_controls():
    bundles=[bundle(v) for v in itertools.product((False,True),repeat=3)]
    rows,report=build_rule_controls(bundles)
    assert len(rows)==216 and report['matched_triplets']==72
    for direct,structured,reordered in zip(rows[::3],rows[1::3],rows[2::3],strict=True):
        p=direct['provenance'];a,b=p['dependency_targets'];op=p['rule']['op']
        expected={'all':a and b,'any':a or b,'except':a and not b}[op]
        assert direct['targets']==[expected]
        assert structured['targets']==[a,b,expected]
        assert reordered['targets']==[b,a,expected]
        assert direct['request']['questions'][-1]==structured['request']['questions'][-1]==reordered['request']['questions'][-1]
        assert direct['request']['requirements']==structured['request']['requirements']
        assert not {'targets','provenance','rule_id','source_group','supervised'} & direct['request'].keys()
        assert predicted_fact_rule(p['rule'],dict(zip(p['dependency_keys'],[a,b]))) is expected


def test_masked_fact_excludes_all_its_rules_even_if_short_circuit_resolves():
    b=bundle((False,True,None));b['supervised'][2]=False
    rows,report=build_rule_controls([b])
    assert len(rows)==9 and len(report['excluded'])==6
    assert all(r['provenance']['dependency_keys']==['date','time'] for r in rows)


def test_targets_cannot_change_rendered_model_input():
    a,_=build_rule_controls([bundle((False,True,False))])
    b,_=build_rule_controls([bundle((True,False,True))])
    assert [r['request'] for r in a]==[r['request'] for r in b]
    assert [r['targets'] for r in a]!=[r['targets'] for r in b]


def test_executor_uses_predictions_and_rejects_missing_or_nonboolean():
    rule={'op':'except','base':{'op':'fact','key':'a'},'exception':{'op':'fact','key':'b'}}
    assert predicted_fact_rule(rule,{'a':True,'b':False})
    assert not predicted_fact_rule(rule,{'a':True,'b':True})
    with pytest.raises(ValueError):predicted_fact_rule(rule,{'a':True})
    with pytest.raises(ValueError):predicted_fact_rule(rule,{'a':1,'b':False})
