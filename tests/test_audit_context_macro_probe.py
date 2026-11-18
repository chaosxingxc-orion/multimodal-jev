import copy
import json
from pathlib import Path

import pytest
import torch

from test_context_macro_measurement import fixture
from test_context_bulk_rpc_candidate import record
from multimodal_jev import context_macro_rollout as macro, local_checkpoint
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.baseline import file_hash
from tools.reports.context_macro_measurement import measure
from tools.reports.audit_context_macro_probe import audit_outputs


def build(tmp_path,monkeypatch):
    monkeypatch.setattr(macro,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    package=tmp_path/'package';package.mkdir();collection=tmp_path/'collected';collection.mkdir()
    rows,reference,_=fixture();torch.save(rows,package/'inputs.pt')
    reference['input_sha256']=file_hash(package/'inputs.pt')
    runtime={'profile':'fixture'};contract={'base_id':'base'}
    class Client:
        phase='ready';bundle=dict(policy_id='p',contract=contract);runtime_identity=runtime
        def score_many(self,batch,**kw):
            records=[record('p',r) for r in batch]
            for r in records:r.update(logits=torch.tensor(r['prob']).logit().tolist(),output_protocol=WIRE)
            return records
    pool=macro.MacroRolloutPool([Client(),Client()],collection/'journal/attempt',runtime_identity=runtime)
    result=measure(pool,rows,reference['passes'][0],audio_token_id=1)
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write(package/'measurement.json',dict(policy=dict(policy_id='p',contract=contract),
        macro_client=dict(runtime_identity=runtime,rollout_queries=32,attempt_id='attempt')))
    write(package/'serial-reference.json',reference)
    for i in range(2):
        write(package/f'service-{i}.json',dict(policy=dict(max_batch_queries=4)))
        write(collection/f'service-{i}/ready.json',dict(physical_devices=[6+i],policy_id='p',
            contract=contract,max_batch_queries=4,model_forward_mode='fixture'))
    for role in ('service-0','service-1','measurement'):write(collection/role/'terminal.json',dict(exit_code=0))
    write(collection/'measurement/measurement.json',result)
    return package,collection


def test_independent_audit_reconstructs_full_macro_and_ownership(tmp_path,monkeypatch):
    package,collection=build(tmp_path,monkeypatch);result=audit_outputs(package,collection)
    assert result['status']=='finite_two_replica_macro_audited'
    assert all(set(owners)=={0,1} for owners in result['rollout_replica_owners'])


@pytest.mark.parametrize('fault',['owner','record','complete','input','timing','device','terminal'])
def test_matching_final_claims_do_not_hide_corrupt_rollout_files(tmp_path,monkeypatch,fault):
    package,collection=build(tmp_path,monkeypatch)
    batch=collection/'journal/attempt/batch-00000000'
    path=batch/'rollout-0000.json'
    if fault=='complete':path=batch/'COMPLETE.json'
    if fault=='input':path=batch/'INPUT.json'
    if fault=='timing':path=collection/'measurement/measurement.json'
    if fault=='device':path=collection/'service-0/ready.json'
    if fault=='terminal':path=collection/'service-1/terminal.json'
    value=json.loads(path.read_text())
    if fault=='owner':value['replica_id']=1-value['replica_id']
    if fault=='record':value['records'].pop()
    if fault=='complete':value['result_sha256']='wrong'
    if fault=='input':value['inputs'].reverse()
    if fault=='timing':value['timed_mean_seconds']+=1
    if fault=='device':value['physical_devices']=[4]
    if fault=='terminal':value['exit_code']=1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):audit_outputs(package,collection)
