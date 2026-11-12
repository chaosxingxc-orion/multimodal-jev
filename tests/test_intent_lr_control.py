import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_intent_lr_control import audit_control


def fixture():
    original=dict(arm='D',step=0,lr=1e-4,seed=317,initial_adapter='/initial',initial_optimizer_checkpoint=None,
                  global_requests=8,runtime_profile='fixed',pins={'old':'hash'})
    control=dict(original,lr=2e-5,pins={'relocated':'hash'})
    return original,control


def test_only_lr_changes():
    a,b=fixture();assert audit_control(a,b)['control_lr']==2e-5


@pytest.mark.parametrize('key,value',[('lr',1e-5),('arm','P'),('seed',318),('initial_adapter','/step2'),
    ('global_requests',16),('runtime_profile','different'),('initial_optimizer_checkpoint','/adam'),('step',1)])
def test_rejects_confounders(key,value):
    a,b=fixture();b[key]=value
    with pytest.raises(ValueError):audit_control(a,b)
