"""Binding/negative cases only; NPU execution is verified on real collections."""
import copy
import json
from pathlib import Path
import pytest
from tools.reports import context_arithmetic_acceptance as module
from tools.reports.prepare_context_arithmetic_candidate import transform_trainer


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old=tmp_path/'old';rel='source/tools/jev/train_context_online_candidate.py'
    p=old/rel;p.parent.mkdir(parents=True)
    p.write_text(Path('.local/decision-objective-research/context-checkpoint-shared-npu-v3-package'/Path(rel)).read_text())
    previous=tmp_path/'previous.json';previous.write_text(json.dumps(dict(candidate_package=str(old))))
    accepted={};entries={}
    for index,role in enumerate(('continuous','fresh_replay')):
        root=tmp_path/role;root.mkdir();package=tmp_path/(role+'-package')
        p=package/rel;p.parent.mkdir(parents=True);p.write_text(transform_trainer((old/rel).read_text()))
        c=dict(start_update=index,checkpoint_interval=1,source_policy_id='stage1',source_contract={},
            base_id='base',runtime_profile='profile',release_sha256='data',schedule_sha256='schedule',
            algorithm='grpo_v1',behavior_bounds={},binary_policy_arithmetic=module.PROFILE)
        (root/'config.json').write_text(json.dumps(c))
        report=tmp_path/(role+'-report.json');report.write_text(json.dumps(dict(actor_container=role,passed=True)))
        transport=tmp_path/(role+'-transport.json');transport.write_text('{}')
        launch=tmp_path/(role+'-launch.json');launch.write_text('{}')
        entries[role]=dict(collection=str(root),transport=str(transport),launch=str(launch),
            package=str(package),initial_policy='fixture',schedule='fixture',report=str(report))
        accepted[str(root)]=json.loads(report.read_text())
    monkeypatch.setattr(module,'original_accept',lambda **kw:dict(scope='mock previous NPU binding'))
    monkeypatch.setattr(module,'audit',lambda **kw:copy.deepcopy(accepted[kw['collection']]))
    monkeypatch.setattr(module,'verify_update',lambda *a:dict(exact_masters=True,fixture=True))
    return dict(**entries,previous_acceptance=str(previous))


def test_affected_path_reaudit_and_explicit_old_evidence_scope(fixture):
    result=module.verify(**fixture)
    assert result['status']=='arithmetic_affected_paths_accepted_no_dispatch'
    assert result['replay']['exact_masters']
    assert 'Earlier-policy' in result['reused_checkpoint_frequency']['qualification']


@pytest.mark.parametrize('fault',['report','same_source','arithmetic','cursor','interval','trainer'])
def test_reject_mixed_or_mutated_evidence(fixture,fault):
    entry=fixture['fresh_replay'];root=Path(entry['collection']);p=root/'config.json'
    c=json.loads(p.read_text())
    if fault=='report':Path(entry['report']).write_text('{}')
    elif fault=='same_source':c['source_policy_id']='other'
    elif fault=='arithmetic':c['binary_policy_arithmetic']='old'
    elif fault=='cursor':c['start_update']=0
    elif fault=='interval':c['checkpoint_interval']=64
    elif fault=='trainer':
        p2=Path(fixture['continuous']['package'])/'source/tools/jev/train_context_online_candidate.py'
        p2.write_text(p2.read_text()+'\n# unregistered change\n')
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):module.verify(**fixture)
