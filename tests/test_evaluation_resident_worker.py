import copy
import hashlib
import io

import pytest
import torch

from tools.reports.evaluation_resident_worker import batches,load_input,compare_records,capacity_policy,submission_capacity
from multimodal_jev.span_request_identity import tensor_identity


def test_cache_bytes_tensors_and_shape_are_independently_bound(tmp_path):
    v=dict(inputs={'input_ids':torch.tensor([[1,2]]),'input_features':torch.ones(1,2,3)},positions=torch.tensor([[1]]))
    buf=io.BytesIO();torch.save(v,buf);raw=buf.getvalue();(tmp_path/'key.pt').write_bytes(raw)
    entry=dict(cache_key='key',cache_sha256=hashlib.sha256(raw).hexdigest(),
               tensors={k:tensor_identity(x) for k,x in dict(v['inputs'],positions=v['positions']).items()})
    assert torch.equal(load_input(entry,tmp_path)['positions'],v['positions'])
    bad=copy.deepcopy(entry);bad['tensors']['positions']['shape']=[1,2]
    with pytest.raises(ValueError):load_input(bad,tmp_path)
    (tmp_path/'key.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):load_input(entry,tmp_path)


def test_length_batches_keep_all_rows_and_budget():
    entries=[dict(id=str(i),tensors={'input_ids':{'shape':[1,n]}}) for i,n in enumerate([4,7,3,8,5])]
    groups=list(batches(entries,2,10))
    assert sorted(i for b in groups for i in b)==list(range(5))
    assert all(len(b)<=2 and sum(entries[i]['tensors']['input_ids']['shape'][-1] for i in b)<=10 for b in groups)


def test_compare_requires_complete_same_policy_real_outputs():
    r=dict(id='x',policy_id='p',request_identity={'audio':'a'},prob=[.1,.9],logits=[-2.,2.])
    assert compare_records([r],[r],'p')['exact']
    bad=dict(r,logits=[-2.,2.1]);assert not compare_records([bad],[r],'p')['exact']
    with pytest.raises(ValueError):compare_records([dict(r,policy_id='q')],[r],'p')
    with pytest.raises(ValueError):compare_records([], [r],'p')
    with pytest.raises(ValueError):compare_records([r,r],[r,r],'p')
    with pytest.raises(ValueError):compare_records([dict(r,prob=[float('nan'),.9])],[r],'p')


def test_batch128_updates_all_capacities_without_mutating_frozen_config():
    old=dict(max_model_len=4096,max_batch_queries=32,max_batch_tokens=131072,kv_cache_memory_bytes=1024**3)
    c=dict(batch_queries=128,max_batch_tokens=393216,kv_cache_memory_bytes=44*1024**3)
    result=capacity_policy(old,c)
    assert result['max_batch_queries']==128 and result['max_batch_tokens']==393216
    assert old['max_batch_queries']==32 and old['kv_cache_memory_bytes']==1024**3
    with pytest.raises(ValueError):capacity_policy(old,dict(c,kv_cache_memory_bytes=60*1024**3))


def test_submission256_does_not_claim_scheduler_or_kv_capacity256():
    policy=dict(max_model_len=4096,max_batch_queries=128,max_batch_tokens=393216,kv_cache_memory_bytes=44*1024**3)
    submitted=submission_capacity(policy,256)
    assert submitted['max_batch_queries']==256 and submitted['max_batch_tokens']==1048576
    assert policy['max_batch_queries']==128 and policy['max_batch_tokens']==393216
