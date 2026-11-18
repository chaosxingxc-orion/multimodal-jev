from concurrent.futures import ThreadPoolExecutor
import hashlib
from types import SimpleNamespace
import pytest
import torch

import multimodal_jev.cached_span_preparation as module
from multimodal_jev.encoded_audio_cache import EncodedAudioCache


def setup(tmp_path, monkeypatch):
    audio=tmp_path/'audio';audio.write_bytes(b'fixture')
    request=SimpleNamespace(audio=str(audio),questions=['q'],
        model_input=lambda:dict(audio=str(audio),questions=['q'],version='fixture-v1'))
    counts={'decode':0,'encode':0}
    def decode(*args,**kwargs):counts['decode']+=1;return [0.],{'wave':'sha'}
    def encode(*args):
        counts['encode']+=1
        return dict(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long),
                    input_features=torch.ones(1,2,3),input_features_mask=torch.ones(1,3,dtype=torch.long)),torch.tensor([[1]])
    monkeypatch.setattr(module,'load_audio',decode)
    monkeypatch.setattr(module,'encode_span_request',encode)
    monkeypatch.setattr(module,'span_labels',lambda *args:[{'token_ids':[3,4]}])
    preparer=module.CachedSpanPreparer(lambda:SimpleNamespace(tokenizer=None),EncodedAudioCache(tmp_path/'cache'),
                                      processor_sha256='a'*64)
    return request,hashlib.sha256(b'fixture').hexdigest(),preparer,counts


def test_hit_skips_decode_encode_and_policy_binds_late(tmp_path,monkeypatch):
    request,digest,preparer,counts=setup(tmp_path,monkeypatch)
    assert not preparer((request,digest)).cache_hit
    cached=preparer((request,digest));assert cached.cache_hit and counts=={'decode':1,'encode':1}
    target=dict(id='r',targets=[1],supervised=[True],span_weights=[1.])
    slot=dict(id='r',weight=1.,padding=False)
    first=module.bind_instruction_slot(cached,request,target,slot,epoch=0,batch_index=0,index=0,policy_id='p1')
    later=module.bind_instruction_slot(cached,request,target,slot,epoch=1,batch_index=1,index=2,policy_id='p2')
    assert first['policy_id']=='p1' and later['policy_id']=='p2'
    assert first['request_id']!=later['request_id']
    assert torch.equal(first['inputs']['input_features'],later['inputs']['input_features'])
    padded=module.bind_instruction_slot(cached,request,target,dict(slot,padding=True,weight=0.),
                                       epoch=1,batch_index=1,index=3,policy_id='p2')
    assert not padded['real_request'] and padded['request_weight']==0
    with pytest.raises(ValueError,match='padding'):
        module.bind_instruction_slot(cached,request,target,dict(slot,padding=True),
                                     epoch=1,batch_index=1,index=3,policy_id='p2')


def test_changed_source_rejected_even_on_hit(tmp_path,monkeypatch):
    request,digest,preparer,_=setup(tmp_path,monkeypatch)
    preparer((request,digest))
    from pathlib import Path
    Path(request.audio).write_bytes(b'changed')
    with pytest.raises(ValueError,match='source audio'):preparer((request,digest))


def test_concurrent_identical_writers_preserve_cache(tmp_path,monkeypatch):
    request,digest,preparer,_=setup(tmp_path,monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(preparer,[(request,digest)]*4))
    assert len(results)==4 and preparer((request,digest)).cache_hit
    assert len(list((tmp_path/'cache').glob('*.pt')))==1
