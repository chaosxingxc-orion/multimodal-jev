import copy
import pytest
from tools.reports.probe_context_attribute_parity_v2 import content_hash
from tools.reports.probe_context_attribute_parity_v2 import compare, REFERENCE_ID


def reports():
    panel=[];records=[]
    for arm,n in (('direct',60),('q_input',124)):
        for i in range(3):
            entry=dict(id=str(i),arm=arm,request_sha256='a'*64,decision_ids=[f'{arm}-{k}' for k in range(n)])
            panel.append(entry)
            identity=dict(policy_id=REFERENCE_ID,positions=list(range(n)),inputs={'tensor':str(i)})
            records.append(dict(entry,policy_id=REFERENCE_ID,request_identity=identity,
                request_id=content_hash(identity),prob=[.25]*n))
    hf=dict(backend='hf',policy_id=REFERENCE_ID,input_panel=panel,records=records)
    vl=copy.deepcopy(hf);vl['backend']='vllm'
    return hf,vl


def test_exact_difference_and_saturation_are_separate():
    hf,vl=reports();assert compare(hf,vl)['exact']
    vl['records'][0]['prob'][0]=.75
    r=compare(hf,vl);assert not r['exact'] and r['threshold_flips']==1 and r['probability_max_abs']==.5
    hf,vl=reports()
    for x in (hf,vl):x['records'][0]['prob'][0]=1.
    r=compare(hf,vl);assert r['exact'] and not r['finite_log_policy_supported']


@pytest.mark.parametrize('fault',['policy','identity','order','short','duplicate','nan','decision_order','request'])
def test_invalid_pair_rejected(fault):
    hf,vl=reports()
    if fault=='policy':vl['policy_id']='wrong'
    if fault=='identity':vl['records'][0]['request_identity']['inputs']['tensor']='changed'
    if fault=='order':vl['records'].reverse()
    if fault=='short':vl['records'][0]['prob'].pop()
    if fault=='duplicate':
        for x in (hf,vl):x['input_panel'][1]=copy.deepcopy(x['input_panel'][0])
    if fault=='nan':vl['records'][0]['prob'][0]=float('nan')
    if fault=='decision_order':vl['records'][0]['decision_ids'].reverse()
    if fault=='request':vl['records'][0]['request_sha256']='b'*64
    with pytest.raises(ValueError):compare(hf,vl)


def test_wire_hash_matches_actual_frozen_runtime_score_path():
    """Exercise actual transport identity code with only the engine substituted."""
    from types import SimpleNamespace
    import torch
    from multimodal_jev.context_resident_rollout import FixedContextRollout
    from multimodal_jev.slurp_training import content_hash as unrelated_hash
    from tools.reports.probe_context_attribute_parity_v2 import checked
    runtime = object.__new__(FixedContextRollout)
    runtime.bundle = dict(policy_id=REFERENCE_ID,contract=dict(token=dict(id=151706)))
    runtime.max_model_len = 32; runtime.max_decisions = 1; runtime.lora = None
    runtime.engine = SimpleNamespace(encode=lambda *a,**k:[SimpleNamespace(
        prompt_token_ids=[151676,151706],outputs=SimpleNamespace(data=torch.tensor([[.25]])))])
    inputs = dict(input_ids=torch.tensor([[151676,151706]]),
        attention_mask=torch.ones((1,2),dtype=torch.long),
        input_features=torch.zeros((1,128,2),dtype=torch.bfloat16),
        input_features_mask=torch.ones((1,2),dtype=torch.int32))
    record = runtime._score(inputs,torch.tensor([[1]]),151676,SimpleNamespace)
    checked(record,policy_id=REFERENCE_ID,decisions=1)
    assert record['request_id'] != unrelated_hash(record['request_identity'])
