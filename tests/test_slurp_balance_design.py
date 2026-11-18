from pathlib import Path
import math
import runpy
import pytest

m = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slurp_balance.py'))


def test_inverse_frequency_equalizes_class_mass_but_sqrt_is_tempered():
    counts = {'head': 1000, 'tail': 1}
    exact = m['class_weights'](counts, 'inverse_frequency')
    tempered = m['class_weights'](counts, 'sqrt_cap4')
    assert math.isclose(exact['head']*1000, exact['tail'])
    assert 1 < tempered['tail']/tempered['head'] < exact['tail']/exact['head']
    assert math.isclose(sum(counts[k]*tempered[k] for k in counts), 1001)


def test_outer_weights_cannot_repair_absent_positive_menu_or_complete_prevalence():
    rows = [dict(intent='head', source_group='g1')]*9 + [dict(intent='tail', source_group='g2')]
    result = m['audit'](rows, {'intent':['head','tail']},
                         {('g1',0):{'intent':['head']}, ('g2',0):{'intent':['head']}}, 1)
    for profile in result['profiles'].values():
        assert profile['complete_I_native_positive_fraction'] == .5
        assert profile['rotating_menus']['1I']['classes_with_zero_native_positive'] == ['tail']


def test_weight_ess_does_not_claim_extra_samples():
    assert m['effective_size']([1,1,1]) == 3
    assert 1 < m['effective_size']([1,1,100]) < 2
    with pytest.raises(ValueError):
        m['class_weights']({'missing':0}, 'sqrt_cap4')
