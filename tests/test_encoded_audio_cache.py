from types import SimpleNamespace
import pytest
import torch
from multimodal_jev.encoded_audio_cache import EncodedAudioCache, binding, key_for
from multimodal_jev.ordered_prefetch import OrderedPrefetch


def fixture():
    request=SimpleNamespace(model_input=lambda:dict(audio='/a.wav',version='v1',requirements=['rule']))
    identity=binding(request=request,audio_sha256='a'*64,processor_sha256='b'*64,preprocessing={'mode':'raw'})
    inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                input_features=torch.randn(1,2,5),input_features_mask=torch.ones(1,5,dtype=torch.long))
    return identity,inputs,torch.tensor([[2]]),{'waveform':'original'}


def test_exact_roundtrip_mutation_isolation_and_version_misses(tmp_path):
    identity,inputs,pos,receipt=fixture();cache=EncodedAudioCache(tmp_path)
    assert cache.read(identity) is None
    cache.write(identity,inputs,pos,receipt)
    restored=cache.read(identity)
    for k in inputs:assert torch.equal(inputs[k],restored[0][k])
    restored[0]['input_features'].zero_()
    assert torch.equal(cache.read(identity)[0]['input_features'],inputs['input_features'])
    assert cache.read(dict(identity,processor_sha256='c'*64)) is None
    assert cache.read(dict(identity,preprocessing={'mode':'gain'})) is None
    assert cache.read(dict(identity,request=dict(identity['request'],version='v2'))) is None


def test_immutable_collision_and_corruption_fail(tmp_path):
    identity,inputs,pos,receipt=fixture();cache=EncodedAudioCache(tmp_path)
    path=cache.write(identity,inputs,pos,receipt)
    cache.write(identity,inputs,pos,receipt)
    with pytest.raises(ValueError,match='collision'):cache.write(identity,inputs,pos,{'waveform':'other'})
    damaged=torch.load(path,weights_only=True);damaged['inputs']['input_ids'][0,0]=5;torch.save(damaged,path)
    with pytest.raises(ValueError,match='integrity'):cache.read(identity)
    assert not list(tmp_path.glob('.building-*'))


def test_cache_refuses_supervision(tmp_path):
    identity,inputs,pos,receipt=fixture()
    with pytest.raises(ValueError,match='label-free'):
        EncodedAudioCache(tmp_path).write(identity,dict(inputs,labels=torch.ones(1)),pos,receipt)


def test_prefetch_preserves_order_and_bounded_source_consumption():
    consumed=[]
    def source():
        for i in range(7):consumed.append(i);yield i
    with OrderedPrefetch(lambda x:x*x,source(),workers=2,ahead=2) as queue:
        assert next(queue)==0
        assert len(consumed)==3  # current result + two future batches
        assert list(queue)==[1,4,9,16,25,36]
    assert queue.closed


def test_prefetch_exception_never_skips_failed_item_and_resume_cursor():
    def prepare(x):
        if x==3:raise RuntimeError('audio broken')
        return x
    with OrderedPrefetch(prepare,range(2,6),ahead=2) as queue:
        assert next(queue)==2
        with pytest.raises(RuntimeError,match='audio broken'):next(queue)
    assert queue.closed
    with OrderedPrefetch(lambda x:x,range(4,6),ahead=1) as resumed:
        assert list(resumed)==[4,5]
