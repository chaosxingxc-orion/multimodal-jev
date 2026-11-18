"""Numerical-release refusal cases; actual NPU replay is audited separately."""
import copy
import json
from pathlib import Path

import pytest
from tools.reports import prepare_context_arithmetic_formal as publisher
from tools.reports.context_arithmetic_formal_runtime import validate
from tools.reports.context_probability_policy_v2 import PROFILE
from test_context_formal_runtime import release


@pytest.fixture
def repaired_release(release):
    c,m,publish=release;root=Path(c['formal_release']).parent
    c['binary_policy_arithmetic']=PROFILE;m['config_contract']['binary_policy_arithmetic']=PROFILE
    evidence=dict(status='arithmetic_affected_paths_accepted_no_dispatch',
        binary_policy_arithmetic=PROFILE,source_policy_id=c['source_policy_id'])
    for name,value in [('arithmetic.json',evidence),('continuous.json',dict(evidence,passed=True)),
                       ('replay.json',dict(evidence,passed=True))]:
        (root/name).write_text(json.dumps(value));m['files'][name]=publisher.sha(root/name)
    m['numerical_repair']=dict(profile=PROFILE,acceptance_file='arithmetic.json');publish()
    return c,m,publish


@pytest.mark.parametrize('algorithm',['grpo_v1','rloo_v1'])
def test_same_runtime_binds_both_algorithms(repaired_release,algorithm):
    c,m,publish=repaired_release;c['algorithm']=m['config_contract']['algorithm']=algorithm;publish()
    assert validate(c)['binary_policy_arithmetic']==PROFILE


@pytest.mark.parametrize('fault',['profile','missing','source','status','old_report','mutated_bytes'])
def test_reject_old_or_unbound_numerics(repaired_release,fault):
    c,m,publish=repaired_release;root=Path(c['formal_release']).parent
    if fault=='profile':c['binary_policy_arithmetic']=m['config_contract']['binary_policy_arithmetic']='old';publish()
    elif fault=='missing':m.pop('numerical_repair');publish()
    else:
        name='replay.json' if fault=='old_report' else 'arithmetic.json'
        value=json.loads((root/name).read_text())
        if fault=='source':value['source_policy_id']='other'
        elif fault=='status':value['status']='pending'
        else:value['binary_policy_arithmetic']='old'
        (root/name).write_text(json.dumps(value))
        if fault!='mutated_bytes':m['files'][name]=publisher.sha(root/name);publish()
    with pytest.raises(ValueError):validate(c)


def test_trainer_preserves_accepted_body():
    p=Path('.local/decision-objective-research/context-checkpoint-shared-npu-v3-package/source/tools/jev/train_context_online_candidate.py')
    old=p.read_text();candidate=publisher.transform_trainer(old)
    new=publisher.formal_trainer(candidate,old)
    guard='        from context_arithmetic_formal_runtime import validate\n        validate(c)\n'
    assert new.replace(guard,publisher.GUARD)==candidate
    for c,o in [(candidate+'\n',old),(candidate,old+'\n'),(new,old)]:
        with pytest.raises(ValueError):publisher.formal_trainer(c,o)


def test_new_orchestrator_routes_actual_training_audit(tmp_path,monkeypatch):
    from tools.reports import run_context_arithmetic_experiment as m
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda args,**kw:calls.append(args))
    m.invoke('audit_context_rl_training',['--spec','fixture'],tmp_path/'audit.log')
    m.invoke('launch_context_formal_training_v2',[],tmp_path/'launch.log')
    assert calls[0][2]=='tools.reports.audit_context_arithmetic_training'
    assert calls[1][2]=='tools.jev.launch_context_formal_training_v2'
