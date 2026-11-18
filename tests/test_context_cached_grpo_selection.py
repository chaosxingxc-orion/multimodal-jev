"""Reuse real shared packaging/runner and scoring tests with new origin adapter."""
import copy
from pathlib import Path

import pytest

from tools.reports import prepare_context_cached_grpo_selection as preparation
from tools.reports import context_cached_grpo_selection_operations as operations
from tools.reports import collect_context_cached_grpo_curve as collector
from tools.reports.evaluate_context_transfer_macro import source_binding
from test_prepare_context_replica_selection import setup,write


def test_actual_shared_packager_reconstructs_grpo_guard_without_changing_model(tmp_path,monkeypatch):
    c,parent=setup(tmp_path,monkeypatch)
    original=preparation.shared.inputs(c,{})
    point=dict(original[5],step=634)
    c['source_step']=634
    monkeypatch.setattr(preparation,'inputs',lambda *a:(*original[:5],point,*original[6:]))
    c['manifest_sha256']=preparation.prepare(c,{})['manifest_sha256']
    root=Path(c['output']);m=preparation.shared.read(root/'PREPARED.json')
    assert m['status']=='replica_cached_grpo_selection_prepared_only' and m['source_step']==634
    assert 'range(64,634,64)' in (root/'run.py').read_text()
    assert 'range(32,318,32)' not in (root/'run.py').read_text()
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    assert preparation.audit(c,{})['status']=='replica_cached_grpo_selection_package_audited_not_executed'
    (root/'run.py').write_text('pass')
    with pytest.raises(ValueError):preparation.audit(c,{})


def test_grpo_packager_rejects_an_unrelated_parent_before_model_loading(monkeypatch):
    monkeypatch.setattr(preparation,'reconstruct',lambda c:{'source_operations':{'training_package':{'parent':{'correct':True}}}})
    with pytest.raises(ValueError,match='formal-training parent'):preparation.inputs({'parent':{}},{})


@pytest.mark.parametrize('failure',[None,'launch','collection','audit','status'])
def test_actual_point_lifecycle_keeps_grpo_status_and_failure_gate(tmp_path,monkeypatch,failure):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(operations,'check_paths',lambda _:None)
    calls=[];c=dict(selection_dispatch_allowed=True,operations_root=str(tmp_path/'ops'),tag='fixture',selection_package={})
    def launch(c,p):
        calls.append('launch')
        if failure=='launch':raise ValueError('failed launch')
        return dict(status='wrong' if failure=='collection' else 'selection_collected_independent_audit_required')
    def audit(c,p):
        calls.append('audit')
        if failure=='audit':raise ValueError('failed independent audit')
        return dict(status='wrong' if failure=='status' else 'complete_replica_cached_grpo_selection_audited',source_step=634)
    monkeypatch.setattr(operations,'launch',launch);monkeypatch.setattr(operations,'audit',audit)
    if failure:
        with pytest.raises(ValueError):operations.run(c,{})
    else:operations.run(c,{})
    assert calls==(['launch'] if failure in ('launch','collection') else ['launch','audit'])
    assert operations.shared.read(Path(c['operations_root'])/'finalization/terminal.json')['exit_code']==int(failure is not None)


def test_complete_grpo_curve_produces_transfer_consumable_frozen_roles(tmp_path,monkeypatch):
    from test_collect_context_replica_infonce_curve import setup as curve_setup
    from tools.reports.context_cached_formal import STEPS
    from multimodal_jev import local_checkpoint
    c,plan,audits,_=curve_setup(tmp_path,monkeypatch)
    plan['algorithm']='dual_credit_grpo_floor_clipped_v1'
    plan['experiment']['task_stage']['updates']=634;plan['experiment']['evaluation']['checkpoints']=STEPS
    for e,checkpoint,step in zip(c['evaluations'],plan['checkpoints'],STEPS,strict=True):
        old=e['step'];e['step']=step;checkpoint['step']=step
        bound=collector.shared.read(e['audit_config']);spec=bound['selection_package'];spec['source_step']=step
        write(Path(e['audit_config']),bound);e['audit_config_sha256']=collector.shared.file_hash(e['audit_config'])
        root=Path(spec['output']);m=collector.shared.read(root/'PREPARED.json');m['source_step']=step;write(root/'PREPARED.json',m)
        audits[old]['source_step']=step;audits[old]['status']='complete_replica_cached_grpo_selection_audited'
    for e in c['evaluations']:
        root=Path(collector.shared.read(e['audit_config'])['selection_package']['output']);write(root/'SOURCE-PLAN.json',plan)
    monkeypatch.setattr(collector,'reconstruct_plan',lambda c:copy.deepcopy(plan))
    lookup={p['step']:audits[old] for p,old in zip(plan['checkpoints'],list(audits),strict=True)}
    monkeypatch.setattr(collector.point,'audit',lambda bound,p:copy.deepcopy(lookup[bound['selection_package']['source_step']]))
    monkeypatch.setattr(collector.publication,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    result=collector.collect(c,{})
    assert result['status']=='complete_source_curve_audited_and_roles_frozen'
    assert result['roles']['common_Stage1']['policy_id']=='p0'
    assert result['roles']['fixed_endpoint_new_GRPO']['step']==634
    chosen=result['roles']['source_selected_new_GRPO']
    assert chosen['step']==128
    assert source_binding(dict(source_evaluation=result['source_evaluation'],policy_role='source_selected_new_GRPO',
        policy_id=chosen['policy_id']))['step']==128
