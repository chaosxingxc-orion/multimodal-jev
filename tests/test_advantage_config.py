import pytest
from multimodal_jev.advantage_config import normalization,validate_continuation


def test_default_legacy_normalization_and_continuation():
    assert normalization({}) is True
    validate_continuation({}, {'norm_adv_by_std':True})
    validate_continuation({'norm_adv_by_std':False},{'norm_adv_by_std':False})
    with pytest.raises(ValueError,match='changed within trajectory'):
        validate_continuation({}, {'norm_adv_by_std':False})
    with pytest.raises(ValueError,match='changed within trajectory'):
        validate_continuation({'norm_adv_by_std':False},{})


@pytest.mark.parametrize('value',[0,1,'false',None])
def test_normalization_rejects_non_boolean(value):
    with pytest.raises(ValueError,match='boolean'):normalization({'norm_adv_by_std':value})
