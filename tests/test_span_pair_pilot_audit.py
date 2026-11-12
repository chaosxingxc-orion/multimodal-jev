import importlib.util
import json
from pathlib import Path
import pytest
import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_pair_credit import exact_pair_credit,expected_pair_reward
from multimodal_jev.span_pair_verl import reference_pair_loss
from multimodal_jev.span_parity import compare_span_policies


@pytest.mark.parametrize('fault',[None,'loss','pair','identity','moment','step','ready','admission'])
def test_independent_audit_rejects_corrupt_evidence(tmp_path,monkeypatch,fault):
    path=Path(__file__).resolve().parents[1]/'tools/jev/verify_span_pair_pilot.py'
    monkeypatch.syspath_prepend(str(path.parent));spec=importlib.util.spec_from_file_location('pair_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=tmp_path/'protocol.json';p.write_text('{}');c=dict(protocol=str(p),pair_weight=.5)
    protocol=dict(base_id='base',initial_adapter_id='initial',algorithm='algorithm')
    rows=[dict(id=str(i),targets=[False,True,bool(i%2)]) for i in range(8)]
    pairs=torch.arange(8).reshape(4,2);gold=torch.tensor([i%2 for i in range(8)])
    old=tuple(torch.tensor([[.6,.4]]*8).log() for _ in range(3));ids=['bound-'+str(i) for i in range(8)]
    refs={r['id']:dict(candidate_logprobs=[p[i].tolist() for p in old],request_id=ids[i]) for i,r in enumerate(rows)}
    before=dict(candidate_logprobs=[p.tolist() for p in old],request_ids=ids)
    after=dict(candidate_logprobs=[p.tolist() for p in old],policy_id='base:next',expected_pair_reward=expected_pair_reward(old[-1],gold,pairs,.5).tolist())
    state=dict(schema='jev-pair-engineering-pilot-v1',protocol_sha256=file_hash(p),pair_weight=.5,completed_updates=1)
    checkpoint=dict(base_id='base',adapter_id='next',completed_updates=1,training_state=state)
    saved=dict(request_ids=ids,policy_id='base:initial',targets=gold,pairs=pairs,advantages=exact_pair_credit(old[-1],gold,pairs,.5),candidate_logprobs=old)
    audits=[compare_span_policies([p[i].tolist() for p in old],[p[i].tolist() for p in old]) for i in range(8)]
    update=dict(initial_policy_id='base:initial',exported_policy_id='base:next',optimizer_updates=1,training_requests=8,pairs=4,
                pair_weight=.5,auxiliary_weight=0.,algorithm='algorithm',checkpoint_ready=False,training_admitted=False,
                admission=audits,loss=float(reference_pair_loss(old[-1],old[-1],gold,pairs,.5)),
                expected_reward_before=float(expected_pair_reward(old[-1],gold,pairs,.5).mean()))
    adam={i:dict(step=torch.tensor(1.),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1)) for i in range(112)}
    optimizer=dict(base_id='base',adapter_id='next',completed_updates=1,training_state=state,ordered_parameter_names=[str(i) for i in range(112)],
                   optimizer=dict(optimizer=dict(state=adam),masters=[torch.zeros(1) for _ in range(112)]))
    if fault=='loss':update['loss']=1.
    elif fault=='pair':saved['pairs']=pairs.flip(-1)
    elif fault=='identity':after['policy_id']='other'
    elif fault=='moment':adam[0]['exp_avg'][0]=float('nan')
    elif fault=='step':adam[0]['step']=torch.tensor(2.)
    elif fault=='ready':update['checkpoint_ready']=True
    elif fault=='admission':update['admission']=[]
    args=(c,protocol,{},rows,pairs,before,after,update,saved,checkpoint,optimizer,refs)
    if fault:
        with pytest.raises((ValueError,AssertionError)):module.audit_payload(*args)
    else:assert module.audit_payload(*args)['passed']
