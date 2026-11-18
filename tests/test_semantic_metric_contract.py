import itertools
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_semantic_metric_contract import binary_summary, native_summary, panel_summary, STATES


def test_native_fixed_macro_and_source_weighting_have_different_denominators():
    r = native_summary(['a', 'b', 'c'], [('a', 'a', 'g1'), ('b', 'a', 'g1'), ('a', None, 'g2')])
    assert r['accuracy'] == pytest.approx(1/3)
    assert r['macro_f1_fixed'] == pytest.approx(1/6)
    assert r['macro_f1_supported'] == .25
    assert r['source_equal_accuracy'] == .25
    assert r['per_class']['c']['recall'] is None
    assert r['invalid_rate'] == pytest.approx(1/3)


def test_invalid_negative_is_not_correct_and_fpr_is_not_failure_rate():
    r = binary_summary([('positive', None), ('negative', 'false')])
    assert r['balanced_accuracy'] == 0
    assert r['false_positive_rate'] == 0
    assert r['negative_failure_rate'] == 1
    assert r['specificity'] == 0
    assert r['counts']['fn'] == r['counts']['invalid_positive'] == 1
    assert r['invalid_rate_admitted'] == 1


def test_masks_and_integer_predictions_never_fabricate_truth_or_typed_answers():
    r = binary_summary([(s, True) for s in STATES[2:]] + [('positive', 1)])
    assert r['coverage'] == .25 and r['positive_recall'] == 0
    assert r['counts']['tp'] == r['counts']['fp'] == r['counts']['tn'] == 0
    assert r['false_positive_rate'] is None and r['balanced_accuracy'] is None
    assert all(r['counts'][s] == 1 for s in STATES[2:])
    missing = binary_summary([('unassessed', True)])
    assert missing['positive_f1'] is None
    assert binary_summary([])['positive_f1'] is None


def test_valid_empty_set_is_distinct_from_no_slot_schema_and_invalid_prediction():
    valid = panel_summary(['s1'], [({'s1': 'negative'}, {'s1': False})])
    invalid = panel_summary(['s1'], [({'s1': 'negative'}, {})])
    assert valid['complete_set_em'] == 1 and invalid['complete_set_em'] == 0
    assert valid['complete_micro_positive_f1'] == invalid['complete_micro_positive_f1'] == 0
    assert valid['complete_macro_positive_f1'] is None


def test_all_single_row_panel_states_and_outputs_against_direct_definition():
    # Exhaustively distinguishes complete truth from partial/masked and no prediction.
    for states in itertools.product(STATES, repeat=2):
        for outputs in itertools.product([False, True, None], repeat=2):
            labels = dict(zip(['s1', 's2'], states)); pred = dict(zip(['s1', 's2'], outputs))
            r = panel_summary(['s1', 's2'], [(labels, pred)])
            complete = all(s in ('positive', 'negative') for s in states)
            assert r['complete_recordings'] == int(complete)
            if complete:
                assert r['complete_set_em'] == int(all(type(p) is bool and p == (s == 'positive')
                                                       for s, p in zip(states, outputs)))
            else:
                assert r['complete_set_em'] is None
                assert r['complete_micro_positive_f1'] is None


def test_complete_set_population_differs_from_masked_decision_population():
    r = panel_summary(['s1', 's2'], [
        ({'s1': 'positive', 's2': 'negative'}, {'s1': False, 's2': False}),
        ({'s1': 'positive', 's2': 'unassessed'}, {'s1': True, 's2': True})])
    assert r['decision_metrics']['s1']['positive_f1'] == pytest.approx(2/3)
    assert r['complete_micro_positive_f1'] == 0
    assert r['complete_macro_positive_f1'] == 0
    assert r['complete_coverage'] == .5
    empty = panel_summary(['s1'], [({'s1': 'negative'}, {})])
    assert empty['complete_set_em'] == 0 and empty['complete_invalid_rate'] == 1
    assert empty['complete_micro_positive_f1'] == 0
    assert empty['complete_macro_positive_f1'] is None
    none = panel_summary([], [({}, {})])
    assert none['inapplicable_no_slot_schema'] and none['complete_set_em'] is None
    assert none['complete_coverage'] is None


def test_reject_bad_supervision_and_unknown_prediction_coordinates():
    with pytest.raises(ValueError): binary_summary([('missing', False)])
    with pytest.raises(ValueError): panel_summary(['s1'], [({}, {})])
    with pytest.raises(ValueError): panel_summary(['s1'], [({'s1': 'positive'}, {'s2': True})])
