"""Admission tests only: no actual NPU process or resumed optimizer step."""
import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_arithmetic_resume_runtime as runtime
from tools.reports.prepare_context_arithmetic_resume import transform,TRAINER_SHA,RUNNER_SHA
from tests.test_context_arithmetic_formal import repaired_release
from tests.test_context_formal_runtime import release


@pytest.fixture
def recovery(repaired_release,tmp_path):
    original,manifest,publish=repaired_release
    original['reference_cache']=str(tmp_path/'reference-cache')
    manifest['config_contract']['reference_cache']=original['reference_cache'];publish()
    origin=tmp_path/'config.json';origin.write_text(json.dumps(original))
    remote=tmp_path.parent/(tmp_path.name+'-recovery');remote.mkdir()
    config=runtime.expected_config(original,origin_config=str(origin),remote_root=str(remote),step=64)
    commit=tmp_path/'run/checkpoint-0064-committed.json';commit.parent.mkdir()
    commit.write_text(json.dumps(dict(updates=64,joint_policy=dict(policy_id='resumed'))))
    files={str(commit.relative_to(tmp_path)):runtime.sha(commit)}
    for rank in range(4):
        p=tmp_path/f'run/rank-{rank}/checkpoint-0064.pt';p.parent.mkdir();p.write_bytes(b'fixture')
        files[str(p.relative_to(tmp_path))]=runtime.sha(p)
    binding=dict(schema='context-arithmetic-recovery-prefix-v1',original_config_sha256=runtime.sha(origin),
        step=64,runtime_root=str(remote),formal_release_id=manifest['release_id'],
        original_files={},changed_files={},checkpoint_files=files,behavior_policy_id='resumed',
        training_dispatch_allowed=False)
    record=remote/'RESUME.json';record.write_text(json.dumps(binding))
    config['pins']={**original['pins'],str(origin):runtime.sha(origin),str(record):runtime.sha(record),
        **{str(tmp_path/n):h for n,h in files.items()}}
    return config,original,binding,record


def test_two_update_recovery_validates_actual_original_release(recovery):
    c,original,_,_=recovery
    assert runtime.validate(c)['finite_prefix_only']
    assert c['start_update']==64 and c['stop_update']==66
    assert c['source_policy_id']==original['source_policy_id']
    assert c['reference_adapter']==original['reference_adapter']
    assert not c['training_dispatch_allowed'] and c['engineering_prefix']


@pytest.mark.parametrize('fault',['algorithm','reference','microbatch','end','dispatch','pins','shard','origin','release','behavior'])
def test_recovery_rejects_semantic_or_dependency_changes(recovery,fault):
    c,original,binding,record=recovery
    if fault=='algorithm':c['algorithm']='rloo_v1'
    if fault=='reference':c['source_policy_id']='new-reference'
    if fault=='microbatch':c['microbatch']=32
    if fault=='end':c['stop_update']=634
    if fault=='dispatch':c['training_dispatch_allowed']=True
    if fault=='pins':c['pins'].pop(str(Path(c['resume_root'])/'rank-0/checkpoint-0064.pt'))
    if fault=='shard':(Path(c['resume_root'])/'rank-0/checkpoint-0064.pt').write_bytes(b'changed')
    if fault=='origin':Path(c['resume_origin_config']).write_text('{}')
    if fault=='release':Path(original['formal_release']).write_text('{}')
    if fault=='behavior':
        binding['behavior_policy_id']='wrong';record.write_text(json.dumps(binding));c['pins'][str(record)]=runtime.sha(record)
    with pytest.raises((ValueError,KeyError)):runtime.validate(c)


@pytest.mark.parametrize('step',[0,1,63,633,634,True])
def test_only_explicit_recoverable_cursor(step):
    with pytest.raises(ValueError):runtime.expected_config({},origin_config='/old/config.json',remote_root='/new',step=step)


def test_transform_cannot_admit_unpinned_trainer():
    for digest in (TRAINER_SHA,RUNNER_SHA):
        with pytest.raises(ValueError):transform('from context_arithmetic_formal_runtime import validate',digest)
