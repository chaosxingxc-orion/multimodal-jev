import torch,pytest
from multimodal_jev.context_prefix_history import PrefixHistory,block_keys

def binding(**kw):return dict(dict(policy_id='p1',runtime_id='r1',audio_id='audio1',audio_start=4),**kw)

def test_prefix_reuse_restores_all_history_and_preserves_suffix():
 c=PrefixHistory(block_size=2,max_bytes=1024);tokens=list(range(10));x=torch.arange(30.).reshape(10,3)
 assert torch.equal(c.restore(x,tokens,0,binding()),x)
 assert torch.equal(c.restore(x[8:],tokens,8,binding()),x)
 assert c.hit_tokens==8

@pytest.mark.parametrize('key,value',[('policy_id','p2'),('runtime_id','r2'),('audio_id','audio2')])
def test_namespace_miss_fails_closed(key,value):
 c=PrefixHistory(block_size=2,max_bytes=1024);tokens=list(range(10));x=torch.randn(10,3);c.restore(x,tokens,0,binding())
 with pytest.raises(ValueError,match='no matching'):c.restore(x[8:],tokens,8,binding(**{key:value}))

def test_audio_change_only_preserves_pre_audio_prefix():
 a=block_keys(list(range(10)),block_size=2,**binding());b=block_keys(list(range(10)),block_size=2,**binding(audio_id='audio2'))
 assert a[:2]==b[:2] and all(x!=y for x,y in zip(a[2:],b[2:]))

def test_changed_suffix_reuses_shared_prefix():
 c=PrefixHistory(block_size=2,max_bytes=1024);x=torch.randn(10,3);tokens=list(range(10));c.restore(x,tokens,0,binding());tokens[-1]=99
 assert torch.equal(c.restore(x[8:],tokens,8,binding()),x)

def test_eviction_never_silently_fills_hidden():
 c=PrefixHistory(block_size=2,max_bytes=24);x=torch.randn(10,3);tokens=list(range(10));c.restore(x,tokens,0,binding())
 assert c.bytes==24
 with pytest.raises(ValueError):c.restore(x[8:],tokens,8,binding())

def test_changed_hidden_or_invalid_suffix_rejected():
 c=PrefixHistory(block_size=2,max_bytes=1024);x=torch.randn(10,3);tokens=list(range(10));c.restore(x,tokens,0,binding())
 with pytest.raises(ValueError):c.restore(x+1,tokens,0,binding())
 with pytest.raises(ValueError):c.restore(x[1:],tokens,1,binding())
