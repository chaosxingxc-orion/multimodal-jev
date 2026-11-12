import copy
import pytest
from multimodal_jev.intent_optimizer import optimizer_kwargs, validate_saved_optimizer


def test_implicit_and_explicit_original_defaults_equal():
    assert optimizer_kwargs(dict(lr=1e-4)) == optimizer_kwargs(dict(lr=1e-4,adam_betas=[.9,.999]))
    assert optimizer_kwargs(dict(lr=1e-4,adam_betas=[0,.999]))['betas'] == (0,.999)


@pytest.mark.parametrize('value', [[1,.999], [-.1,.999], [True,.999], [.9,float('nan')], [.9], 'default'])
def test_invalid_betas_fail(value):
    with pytest.raises(ValueError):optimizer_kwargs(dict(lr=1e-4,adam_betas=value))


def test_saved_optimizer_cannot_silently_restore_old_momentum():
    config=dict(lr=1e-4,adam_betas=[0,.999])
    state=dict(param_groups=[dict(optimizer_kwargs(config),params=[0],amsgrad=False,maximize=False)])
    validate_saved_optimizer(config,state)
    for key,value in [('betas',(.9,.999)),('lr',2e-5),('weight_decay',.01),('eps',1e-6),('maximize',True)]:
        broken=copy.deepcopy(state);broken['param_groups'][0][key]=value
        with pytest.raises(ValueError):validate_saved_optimizer(config,broken)
