import math
import pytest
from multimodal_jev.span_population_geometry import summarize


def test_identical_biased_policies_have_zero_information():
    result=summarize([[math.log(.9),math.log(.1)]]*3,['x','y'],['A','b'])
    assert abs(result['mean_kl_request_to_pooled'])<1e-14
    assert result['dominant_prediction_fraction']==1
    assert result['symbol_groups']['uppercase']['mass_over_uniform']==pytest.approx(1.8)


def test_discriminating_population_obeys_entropy_decomposition():
    result=summarize([[math.log(.9),math.log(.1)],[math.log(.1),math.log(.9)]],['x','y'],['A','b'])
    assert result['mean_kl_request_to_pooled']>0.3
    assert abs(result['entropy_decomposition_residual'])<1e-14
    assert result['predicted_classes']==2
    assert result['mean_total_variation_to_pooled']==pytest.approx(.4)


def test_group_masses_account_for_unequal_candidate_counts():
    result=summarize([[math.log(1/3)]*3],['x','y','z'],['A','B','0'])
    assert result['symbol_groups']['uppercase']['mass_over_uniform']==pytest.approx(1)
    assert result['symbol_groups']['digit']['mass_over_uniform']==pytest.approx(1)


@pytest.mark.parametrize('rows',[[[0.,0.]],[[float('nan'),-1]],[[0.]],[]])
def test_invalid_support_rejected(rows):
    with pytest.raises(ValueError):summarize(rows,['x','y'],['A','b'])
