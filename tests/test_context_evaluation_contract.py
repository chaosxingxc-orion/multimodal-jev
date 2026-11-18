import ast
import math
from pathlib import Path

import pytest

from multimodal_jev.context_event_metrics import ContextEventMetrics
from multimodal_jev.context_evaluation_contract import (
    DERIVED_FIELDS, IDENTITY_FIELDS, METRIC_FIELDS, SELECTION_FIELDS, SCOPE,
    delta, from_result, ranked_key, validate_row)

ROOT = Path(__file__).resolve().parents[1]

# Writers pinned inside frozen packages: their literals stay byte-identical, so
# the contract is asserted against them instead of rewriting them.
WRITERS = {
    'tools/jev/fsc_context_metrics.py': (
        'query_macro_set_f1', 'exact_set', 'top1_event_accuracy', 'top1_macro_f1',
        'candidate_binary_nll', 'candidate_brier', 'mean_selected_count'),
    'tools/reports/score_slurp_context_test_pair.py': (
        'query_macro_set_f1', 'exact_set', 'top1_event_accuracy', 'top1_macro_f1',
        'candidate_binary_nll', 'candidate_brier'),
    'tools/reports/score_minds_context_pair.py': (
        'query_macro_set_f1', 'exact_set', 'top1_event_accuracy', 'top1_macro_f1',
        'candidate_binary_nll', 'candidate_brier', 'empty_selection_rate',
        'multiple_selection_rate'),
}
VOCABULARY = set(METRIC_FIELDS) | set(DERIVED_FIELDS)


def metric_literals(path):
    """Every literal tuple of metric names in a file, without importing it."""
    tree = ast.parse((ROOT/path).read_text())
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Tuple, ast.List)) and node.elts
                and all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in node.elts)):
            values = tuple(item.value for item in node.elts)
            if set(values) & VOCABULARY:
                found.append(values)
    return found


def test_pinned_writers_keep_exactly_the_contract_fields():
    # Two of these modules cannot even be imported on a host without the
    # inference dependency, so the assertion reads source and not runtime state.
    for path, expected in WRITERS.items():
        assert metric_literals(path) == [expected], path
        assert set(expected) <= VOCABULARY, path


def test_every_writer_field_is_in_canonical_order():
    for path, expected in WRITERS.items():
        order = [name for name in METRIC_FIELDS + DERIVED_FIELDS if name in expected]
        assert list(expected) == order, path


def test_union_of_writer_fields_covers_the_whole_vocabulary():
    union = set().union(*(set(fields) for fields in WRITERS.values()))
    assert union == VOCABULARY


def test_field_groups_are_disjoint_and_selection_is_a_subset():
    assert not set(METRIC_FIELDS) & set(DERIVED_FIELDS)
    assert not set(METRIC_FIELDS) & set(IDENTITY_FIELDS)
    assert set(SELECTION_FIELDS) <= set(METRIC_FIELDS)
    assert len(set(METRIC_FIELDS)) == len(METRIC_FIELDS)


def test_result_keys_are_exactly_metrics_plus_identity():
    metrics = ContextEventMetrics(['a', 'b', 'c'])
    metrics.add('q1', 'a', [.9, .1, .0])
    metrics.add('q2', 'b', [.2, .7, .1])
    result = metrics.result()
    assert set(result) == set(METRIC_FIELDS) | set(IDENTITY_FIELDS)
    assert result['scope'] == SCOPE
    assert from_result(result) == validate_row(
        {name: result[name] for name in METRIC_FIELDS}, METRIC_FIELDS)


def test_missing_labels_do_not_become_negatives():
    metrics = ContextEventMetrics(['a', 'b'])
    metrics.add('q1', 'a', [.9, .1])
    result = metrics.result()
    assert result['queries'] == 1 and result['exact_set'] == 1.
    assert result['binary_nll_infinite'] is False
    # The projection carries the comparable metrics only; identity stays behind.
    assert set(from_result(result)) == set(METRIC_FIELDS)


def test_validate_row_requires_exact_keys_and_finite_values():
    row = {name: 0.5 for name in METRIC_FIELDS}
    assert validate_row(row, METRIC_FIELDS) == row
    assert validate_row(dict(row, exact_set=None), METRIC_FIELDS)['exact_set'] is None
    with pytest.raises(ValueError):
        validate_row(dict(row, extra=1.), METRIC_FIELDS)
    with pytest.raises(ValueError):
        validate_row({name: row[name] for name in METRIC_FIELDS[:-1]}, METRIC_FIELDS)
    with pytest.raises(ValueError):
        validate_row(dict(row, exact_set=math.nan), METRIC_FIELDS)
    with pytest.raises(ValueError):
        validate_row(dict(row, exact_set=math.inf), METRIC_FIELDS)
    with pytest.raises(ValueError):
        validate_row(dict(row, exact_set=True), METRIC_FIELDS)
    with pytest.raises(ValueError):
        validate_row(dict(row, exact_set='1'), METRIC_FIELDS)
    with pytest.raises(ValueError):
        validate_row([row], METRIC_FIELDS)


def test_from_result_rejects_an_incomplete_evaluator_result():
    with pytest.raises(ValueError, match='lacks contracted metrics'):
        from_result({name: 0. for name in METRIC_FIELDS[:-1]})


def test_ranked_key_is_highest_f1_then_lowest_nll():
    better = {name: None for name in SELECTION_FIELDS}
    better = dict(better, query_macro_set_f1=.7, candidate_binary_nll=.4)
    worse = dict(better, query_macro_set_f1=.6)
    undefined = dict(better, candidate_binary_nll=None)
    assert ranked_key(better) < ranked_key(worse)
    assert ranked_key(better) < ranked_key(undefined)
    assert ranked_key(undefined)[1] == math.inf
    with pytest.raises(ValueError):
        ranked_key(dict(better, query_macro_set_f1=None))


def test_delta_propagates_undefined_and_never_silently_zeroes():
    initial = {name: 1. for name in METRIC_FIELDS}
    later = {name: 1.5 for name in METRIC_FIELDS}
    later['candidate_binary_nll'] = None
    result = delta(initial, later)
    assert result['exact_set'] == pytest.approx(.5)
    assert result['candidate_binary_nll'] is None
    assert delta(initial, initial) == {name: 0. for name in METRIC_FIELDS}
    with pytest.raises(ValueError):
        delta(initial, {name: 1. for name in METRIC_FIELDS[:-1]})
