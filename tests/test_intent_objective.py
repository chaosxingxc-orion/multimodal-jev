import copy
import pytest
from multimodal_jev.intent_objective import identity,specification


def test_legacy_correctness_identity_is_explicit_default():
    assert identity(dict(arm='D'))==identity(dict(arm='D',objective={'name':'correctness-v1'}))
    assert identity(dict(arm='D'))!=identity(dict(arm='D',objective={'name':'smoothed-utility-v1'}))


@pytest.mark.parametrize('objective', [None,{},'utility',{'name':'unknown'},
    {'name':'correctness-v1','scale':2},{'name':'smoothed-utility-v1','tau':.01},
    {'name':'uniform-credit-v1','scale':True},{'name':'uniform-credit-v1','scale':0},
    {'name':'uniform-credit-v1','scale':float('nan')},{'name':'uniform-credit-v1','scale':1001}])
def test_unknown_or_silent_objective_tuning_rejected(objective):
    with pytest.raises(ValueError):specification(dict(arm='D',objective=objective))


def test_no_implicit_multi_span_utility_and_no_resume_switch(tmp_path):
    from test_intent_resume import configs
    from multimodal_jev.intent_resume import validate_lineage
    for name in ('smoothed-utility-v1','uniform-credit-v1'):
        value={'name':name}
        if name=='uniform-credit-v1':value['scale']=6
        with pytest.raises(ValueError):specification(dict(arm='P',objective=value))
        previous,current=configs(tmp_path);previous['arm']=current['arm']='D'
        current['objective']=value
        with pytest.raises(ValueError,match='objective'):validate_lineage(previous,current)
        previous['objective']=copy.deepcopy(value);validate_lineage(previous,current)
        if name=='uniform-credit-v1':
            current['objective']['scale']=7
            with pytest.raises(ValueError,match='objective'):validate_lineage(previous,current)


def test_explicit_objective_resume_requires_accepted_identity_and_credit_files(tmp_path):
    import json
    from pathlib import Path
    from test_intent_resume import fixture
    from multimodal_jev.intent_resume import authenticate
    from multimodal_jev.baseline import atomic_json,file_hash
    cfg=fixture(tmp_path);root=Path(cfg['previous_root']);suite=Path(cfg['previous_suite'])
    cfg['objective']={'name':'correctness-v1'}
    old=json.loads((root/'config.json').read_text());old['objective']=cfg['objective']
    atomic_json(root/'config.json',old);cfg['pins'][str(root/'config.json')]=file_hash(root/'config.json')
    with pytest.raises(ValueError,match='objective identity'):authenticate(cfg)
    verified=json.loads((suite/'verified.json').read_text());verified['objective']=identity(cfg)
    def save_verified():
        atomic_json(suite/'verified.json',verified)
        accepted=json.loads((suite/'ACCEPTED.json').read_text())
        accepted['verified_sha256']=file_hash(suite/'verified.json');atomic_json(suite/'ACCEPTED.json',accepted)
        for p in (suite/'verified.json',suite/'ACCEPTED.json'):cfg['pins'][str(p)]=file_hash(p)
    save_verified()
    with pytest.raises(ValueError,match='objective evidence'):authenticate(cfg)
    for name in ('objective.json','supervision.json'):
        p=root/'run'/name;atomic_json(p,{'fixture':'pinned evidence'})
        verified['files'][str(p)]=file_hash(p);cfg['pins'][str(p)]=file_hash(p)
    save_verified();assert authenticate(cfg)['previous_updates']==1
