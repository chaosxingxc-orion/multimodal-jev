from copy import deepcopy
import pytest
from multimodal_jev.reference_policy import reference_identity,validate_reference_continuation
from test_rl_preflight import fixture
from multimodal_jev.rl_preflight import validate_schedule


def test_reference_identity_and_schedule(tmp_path):
    config=fixture(tmp_path,adapted=True)
    config['actor']['reference']={'checkpoint':config['initial_checkpoint'],'coefficient':0.01,'estimator':'k3'}
    identity=reference_identity(config['actor']['reference'])
    assert validate_schedule(config)['reference_policy']==identity
    validate_reference_continuation(identity,deepcopy(identity))
    for field,value in [('coefficient',0.02),('policy_id','changed'),('estimator','k2')]:
        changed=deepcopy(identity);changed[field]=value
        with pytest.raises(ValueError,match='changed within trajectory'):
            validate_reference_continuation(identity,changed)
    with pytest.raises(ValueError,match='changed within trajectory'):
        validate_reference_continuation(identity,None)
    with pytest.raises(ValueError,match='changed within trajectory'):
        validate_reference_continuation(None,identity)
    validate_reference_continuation(None,None)


@pytest.mark.parametrize('value',[0,-1,True,float('inf'),float('nan')])
def test_reference_rejects_invalid_coefficient(value):
    with pytest.raises(ValueError,match='coefficient'):
        reference_identity({'checkpoint':'unused','coefficient':value,'estimator':'k3'})


def test_reference_rejects_unknown_or_incomplete_contract():
    assert reference_identity(None) is None
    with pytest.raises(ValueError,match='requires'):reference_identity({})
    with pytest.raises(ValueError,match='estimator'):
        reference_identity({'checkpoint':'unused','coefficient':0.01,'estimator':'full'})
