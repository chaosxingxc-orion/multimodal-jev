from pathlib import Path
import runpy
import pytest

m = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_current_semantic_coverage.py'))


def a(value, current=True):
    return dict(proposed_value=value, applicability='current_semantics_reference_only' if current else 'stale_semantics_requires_rereview')


def test_stale_votes_do_not_override_current_or_supply_support():
    assert m['reference_state']([a(True, False), a(False)]) == ('current_reference_proposal', False)
    assert m['reference_state']([a(True, False)]) == ('requires_current_rule_review', None)


def test_disagreement_and_unknown_are_never_negative_support():
    assert m['reference_state']([a(True), a(False)])[0] == 'conflicting_current_proposals'
    assert m['reference_state']([a(False), a(None)])[0] == 'current_proposal_with_unresolved_assessment'
    assert m['reference_state']([a(None)])[0] == 'current_reference_unavailable'


def test_multisource_requires_complete_agreement_without_group_propagation():
    f = m['recording_state']
    c = {('a','t'): ('current_reference_proposal',True)}
    assert f(['a','b'],'t',c) == 'incomplete_or_unresolved'
    c['b','t'] = ('current_reference_proposal',False)
    assert f(['a','b'],'t',c) == 'source_disagreement'
    c['b','t'] = ('current_reference_proposal',True)
    assert f(['a','b'],'t',c) == 'positive_reference'
    assert f(['c'],'t',c) == 'incomplete_or_unresolved'


def fixture():
    s = dict(source_id='s',optimization_allowed=False,semantic_annotation=None,
             recording_bindings=[dict(recording_id='r',all_source_ids=['s'],source_group='g')])
    aa = dict(a(True),assessment_id='a',source_id='s',definition_id='t',admitted_value=None,optimization_allowed=False)
    c = dict(source_id='s',definition_id='t',assessment_ids=['a'],status='current_reference_proposal',admitted_value=None,optimization_allowed=False)
    return [s],['t','empty'],[aa],[c]


def test_unassessed_types_and_one_sided_support_remain_visible():
    result = m['audit'](*fixture())
    assert result['summary']['polarity_support_counts'] == {'neither':1,'positive_only':1}
    assert result['summary']['admitted_coordinates'] == 0
    assert result['per_type'][0]['constant_false_agreement_on_clear_references'] is None
    assert result['per_type'][1]['largest_positive_group_recording_share'] == 1


def test_duplicate_binding_and_admitted_truth_rejected():
    data = fixture()
    data[0][0]['recording_bindings'] *= 2
    with pytest.raises(ValueError,match='duplicate'):
        m['audit'](*data)
    data = fixture()
    data[2][0]['admitted_value'] = True
    with pytest.raises(ValueError,match='admitted'):
        m['audit'](*data)
