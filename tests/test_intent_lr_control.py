import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_intent_lr_control import audit_control, audit_momentum_control, freeze_source, MOMENTUM_FILES


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


def test_only_first_momentum_changes():
    a,_=fixture();b=dict(a,adam_betas=[0.,.999])
    assert audit_momentum_control(a,b)['control_betas']==[0.,.999]


@pytest.mark.parametrize('key,value', [('adam_betas',[0.,.99]),('lr',2e-5),('arm','P'),
    ('seed',318),('initial_adapter','/step5'),('initial_optimizer_checkpoint','/adam'),
    ('global_requests',16),('step',1)])
def test_momentum_control_rejects_confounders(key,value):
    a,_=fixture();b=dict(a,adam_betas=[0.,.999]);b[key]=value
    with pytest.raises(ValueError):audit_momentum_control(a,b)


def test_momentum_snapshot_preserves_reference_compute(tmp_path):
    repo=tmp_path/'repo';template=tmp_path/'template';destination=tmp_path/'snapshot'
    paths=(*MOMENTUM_FILES,'src/multimodal_jev/span_finite_credit.py','tools/jev/other.py')
    for base in (repo,template/'source'):
        for path in paths:
            p=base/path;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(str(base)+'/'+path)
    # An unrelated new algorithm in the live repo must not enter the control.
    (repo/'src/new_algorithm.py').write_text('unrelated')
    manifest=freeze_source(repo,template,destination,'momentum')
    assert 'src/new_algorithm.py' not in manifest
    for path in paths:
        expected=repo if path in MOMENTUM_FILES else template/'source'
        assert (destination/path).read_bytes()==(expected/path).read_bytes()
