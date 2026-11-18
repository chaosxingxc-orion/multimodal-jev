import pytest
from tools.reports.plan_context_kv_capacity import capacity


def model():
    return dict(num_hidden_layers=28,num_key_value_heads=8,head_dim=128,
                layer_types=['full_attention']*28,use_sliding_window=False)


def test_bf16_tp1_formula_matches_real_one_gib_capacity_and_full32():
    r=capacity(model(),[[2828]*32],block_size=128,max_queries=32,max_model_len=4096,
        max_batch_tokens=131072,proposed_bytes=16*2**30)
    assert r['kv_bytes_per_token']==114688
    assert (2**30//r['bytes_per_block'])*128==9344
    assert r['actual_batch_required_bytes']==[32*23*128*114688]
    assert r['full_query_capacity_bound_bytes']==14*2**30
    assert r['covers_full_query_capacity_bound']


def test_each_request_is_rounded_individually_and_insufficient_memory_is_explicit():
    r=capacity(model(),[[129,129]],block_size=128,max_queries=2,max_model_len=256,
        max_batch_tokens=512,proposed_bytes=128*114688)
    assert r['actual_batch_required_bytes']==[512*114688]
    assert not r['covers_observed_batches']


@pytest.mark.parametrize('fault',['sliding','layers','overflow','zero'])
def test_unsupported_models_or_batches_rejected(fault):
    m=model();batch=[[100]*32];block=128
    if fault=='sliding':m['use_sliding_window']=True
    if fault=='layers':m['layer_types'][0]='linear_attention'
    if fault=='overflow':batch=[[4097]*32]
    if fault=='zero':block=0
    with pytest.raises(ValueError):capacity(m,batch,block_size=block,max_queries=32,
        max_model_len=4096,max_batch_tokens=131072,proposed_bytes=16*2**30)
