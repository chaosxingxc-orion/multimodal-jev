"""Full634 audit orchestration; model, credit and tensor audits are substitutes.

The actual per-step state machine, checkpoint inventory, tail denominator,
policy/export comparison and dual acknowledgement checks execute unchanged.
"""
import copy
from pathlib import Path

import pytest
import torch

from test_context_macro_formal_candidate import config, schedule
from tools.reports import audit_context_macro_training as module
from tools.reports.context_macro_formal_candidate import contract as formal_contract, STEPS
from multimodal_jev import slurp_portable_release, context_training_schedule, context_policy_artifacts
from multimodal_jev.span_length_groups import length_group_indices


def fixture(tmp_path, monkeypatch, fault=None):
    package=tmp_path/'package';run=tmp_path/'collection/run'
    package.mkdir();run.mkdir(parents=True)
    journal=tmp_path/'collection/journal/formal'
    journal.mkdir(parents=True)
    for rank in range(2):(run/f'rank-{rank}').mkdir()
    for step in range(1,635):
        (run/f'policy-{step:04d}-committed.json').touch()
        (journal/f'batch-{step-1:08d}').mkdir()
    for step in STEPS[1:]:
        (run/f'checkpoint-{step:04d}-committed.json').touch()
        for rank in range(2):(run/f'rank-{rank}/checkpoint-{step:04d}.pt').touch()
    if fault=='extra_checkpoint':(run/'checkpoint-0001-committed.json').touch()
    if fault=='missing_policy':(run/'policy-0317-committed.json').unlink()
    if fault=='missing_rank':(run/'rank-1/checkpoint-0634.pt').unlink()
    batches,ids=schedule()
    lengths=[(20,30)]*128
    old=length_group_indices(lengths,microbatch=8,world_size=4,order='length_grouped')
    ranks=length_group_indices(lengths,microbatch=8,world_size=2,order='length_grouped')
    for batch in batches:batch['rank_indices']=old
    c=dict(config(),physical_devices=[4,5],global_batch=128,microbatch=8,
        source_policy_id='p0',source_contract={},base_id='base',runtime_profile='fixture',
        release='release',release_sha256='release-hash',schedule_sha256='schedule-hash',
        behavior_bounds={},dual_runtime_identity={'profile':'fixture'},
        macro_client=dict(max_queries=128,rollout_queries=32,runtime_identity={'profile':'fixture'},attempt_id='formal'))
    contract=module.expected_contract(c);contract['formal_experiment']=formal_contract(c)
    task=dict(schema={'candidates':[{'id':str(i)} for i in range(60)]})
    monkeypatch.setattr(slurp_portable_release,'load',lambda *a,**kw:([{'id':i} for i in ids],task,{}))
    monkeypatch.setattr(context_training_schedule,'load_schedule',lambda *a,**kw:(batches,{}))
    def tensors(step):
        result={f'backbone.p{i}':torch.tensor([step/1024],dtype=torch.bfloat16) for i in range(112)}
        result.update({f'decision_head.p{i}':torch.tensor([step/1024],dtype=torch.bfloat16) for i in range(16)})
        return result
    def bundle(path,**kw):
        policy=kw['expected_policy_id'];step=int(policy[1:])
        return dict(policy_id=policy,tensors=dict(tensors(step),decision_token_row=torch.tensor([1.])))
    monkeypatch.setattr(context_policy_artifacts,'read_execution_bundle',bundle)
    monkeypatch.setattr(context_policy_artifacts,'verify_native_adapter',lambda *a:None)
    audits=[]
    monkeypatch.setattr(module,'verify_checkpoint',lambda root,step,*a,**kw:audits.append(step))
    monkeypatch.setattr(module,'audit_masters',lambda *a,**kw:{'fixture_master':True})
    monkeypatch.setattr(module,'validate_trainable_scope',lambda *a:{'fixture_scope':True})
    monkeypatch.setattr(module,'file_hash',lambda p:'hash')
    monkeypatch.setattr(module,'reference_transition',lambda *a,**kw:[])
    monkeypatch.setattr(module,'audit_journal',lambda *a,**kw:{'fixture_journal':True})
    monkeypatch.setattr(module,'update_targets',lambda *a,**kw:{})
    monkeypatch.setattr(module,'audit_groups',lambda *a,**kw:{'update':kw['update']})
    def receipt(index):
        return dict(runtime_identity={'profile':'fixture'},queries=[dict(behavior_record=dict(
            request_identity={'inputs':{'input_ids':{'shape':[1,20]},'input_features':{'shape':[1,30]}}})) for _ in range(128)])
    def publish(step):
        return dict(updates=step,contract=contract,execution_file=f'execution-{step:04d}.pt',
            execution_sha256='hash',joint_policy_directory=f'policy-{step:04d}',
            joint_policy_manifest_sha256='hash',joint_policy={'policy_id':f'p{step}'})
    def read(path):
        path=Path(path);name=path.name
        if path==package/'training.json':return c
        if path==run/'result.json':return dict(passed=True,start_update=0,stop_update=634,schedule_updates=634,
                                             learning_population=40513,engineering_prefix=False)
        if name.startswith('service-'):return {'policy':{'max_batch_queries':32}}
        if name=='terminal.json':return {'exit_code':0}
        if name=='ready.json':return dict(physical_devices=[6+int(path.parent.name[-1])],policy_id='p0',
            contract={},max_batch_queries=32,model_forward_mode='fixture')
        if name=='checkpoint-scope.json':return {'fixture_scope':True}
        if name=='result.json':return dict(passed=True,start_update=0,stop_update=634,
                                          executions={str(i):'bfloat16' for i in range(113)})
        index=int(name.split('-')[1].split('.')[0])
        if name.startswith('rollout-'):return receipt(index)
        if name.startswith('batch-'):return dict(records=[{'request_id':str(i)} for i in range(128)])
        if name.startswith('policy-'):return publish(index)
        if name.startswith('checkpoint-'):
            v=dict(publish(index),rank_shards={f'rank-{r}/checkpoint-{index:04d}.pt':'hash' for r in range(2)})
            if fault=='checkpoint_policy' and index==64:v['joint_policy']={'policy_id':'other'}
            return v
        if name.startswith('activation-'):
            ack=dict(policy_id=f'p{index}',head_marker_exact=True,native_loaded_tensors=112)
            value=dict(ack,replica_receipts=[copy.deepcopy(ack),copy.deepcopy(ack)])
            if fault=='stale_replica' and index==317:value['replica_receipts'][1]['policy_id']='p316'
            return dict(error=None,policy_id=f'p{index}',receipt=value)
        if name.startswith('update-'):
            real=65 if index in (317,634) else 128
            if fault=='tail_denominator' and index==317:real=128
            save=index in STEPS[1:]
            if fault=='checkpoint_flag' and index==1:save=True
            return dict(update=index,real_queries=real,next_policy_id=f'p{index}',rollout=receipt(index-1),
                checkpoint=save,behavior_parity_history=[[dict(passed=True,error=None)]*2 for _ in range(8)],
                total_seconds=1.,actor_seconds=.5,preparation_seconds=.4)
        raise AssertionError(path)
    def load(path,**kw):
        name=Path(path).name;step=int(name.split('-')[1].split('.')[0])
        if name.startswith('groups-'):return [{'request_id':str(i)} for rank in ranks for i in rank]
        if name.startswith('checkpoint-'):return {'parameters':{'weight':torch.tensor([1.])}}
        if name.startswith('execution-'):
            result=tensors(step)
            if fault=='wrong_export' and step==317:result['backbone.p0']+=1
            return result
        raise AssertionError(path)
    monkeypatch.setattr(module,'read',read);monkeypatch.setattr(module.torch,'load',load)
    return package,tmp_path/'collection',audits


def test_full634_output_audit_visits_every_step_and_only_ten_recovery_checkpoints(tmp_path,monkeypatch):
    package,collection,audits=fixture(tmp_path,monkeypatch)
    result=module.audit_outputs(package,collection,formal=True)
    assert result['status']=='formal_dp2_outputs_audited' and len(result['updates'])==634
    assert audits==STEPS[1:]
    assert sum(v['master_export'] is not None for v in result['updates'])==10


@pytest.mark.parametrize('fault',['extra_checkpoint','missing_policy','missing_rank','checkpoint_policy',
    'tail_denominator','checkpoint_flag','stale_replica','wrong_export'])
def test_corrupt_full_run_never_passes_formal_audit(tmp_path,monkeypatch,fault):
    package,collection,_=fixture(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):module.audit_outputs(package,collection,formal=True)
