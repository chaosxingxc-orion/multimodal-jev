import hashlib
import json
from types import SimpleNamespace as N

import pytest
import torch

import multimodal_jev.cached_context_preparation as module
from multimodal_jev.encoded_audio_cache import EncodedAudioCache


def setup(tmp_path,monkeypatch):
    path=tmp_path/'audio';path.write_bytes(b'audio')
    request=N(audio=str(path),questions=['q'],model_input=lambda:dict(audio=str(path),questions=['q'],version='test'))
    calls=[]
    monkeypatch.setattr(module,'load_audio',lambda *a,**k:([0.],dict(wave='audio')))
    def encode(*a,**kw):
        calls.append(kw)
        ids=torch.tensor([[1,98,kw['decision_token_id']]])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=torch.ones(1,4,8),
            input_features_mask=torch.ones(1,8,dtype=torch.long)),torch.tensor([[2]])
    monkeypatch.setattr(module,'encode_context_request',encode)
    cache=EncodedAudioCache(tmp_path/'cache')
    def preparer(marker=99,limit=32):
        return module.CachedContextPreparer(lambda:object(),cache,processor_sha256='a'*64,
            decision_token_id=marker,audio_token_id=98,max_tokens=limit)
    return request,hashlib.sha256(b'audio').hexdigest(),preparer,calls,path


def test_hit_skips_processing_and_changed_compiler_is_new_entry(tmp_path,monkeypatch):
    request,digest,factory,calls,_=setup(tmp_path,monkeypatch)
    prepare=factory();first=prepare((request,digest));second=prepare((request,digest))
    assert not first.cache_hit and second.cache_hit and len(calls)==1
    assert torch.equal(first.inputs['input_features'],second.inputs['input_features'])
    assert second.inputs['input_features'].dtype==torch.bfloat16
    assert not factory(marker=100)((request,digest)).cache_hit
    assert not factory(limit=64)((request,digest)).cache_hit
    second.inputs['input_ids'][0,0]=55
    assert prepare((request,digest)).inputs['input_ids'][0,0]==1


def test_changed_audio_rejected_on_hit(tmp_path,monkeypatch):
    request,digest,factory,_,path=setup(tmp_path,monkeypatch)
    prepare=factory();prepare((request,digest));path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source audio'):prepare((request,digest))


def test_local_replica_preserves_request_and_cache_identity(tmp_path,monkeypatch):
    request,digest,factory,calls,path=setup(tmp_path,monkeypatch)
    original=factory()((request,digest))
    replica=tmp_path/'replica';replica.write_bytes(path.read_bytes());path.unlink()
    accesses=[]
    class Mirror:
        def resolve(self,expected):
            assert expected==digest
            accesses.append(expected)
            assert hashlib.sha256(replica.read_bytes()).hexdigest()==expected
            return replica
    prepare=factory();prepare.audio_mirror=Mirror()
    hit=prepare((request,digest))
    assert hit.cache_hit and len(calls)==1 and accesses==[digest]
    assert hit.request==original.request
    assert torch.equal(hit.inputs['input_features'],original.inputs['input_features'])
    cold=factory(limit=64);cold.audio_mirror=Mirror()
    seen=[]
    monkeypatch.setattr(module,'load_audio',lambda p,*a,**k:(seen.append(p) or [0.],dict(wave='audio')))
    assert not cold((request,digest)).cache_hit
    assert seen==[replica] and len(calls)==2


def test_processor_identity_shared_across_policy_updates(tmp_path):
    from multimodal_jev.slurp_training import content_hash
    root=tmp_path/'policy';(root/'processor').mkdir(parents=True)
    assets={}
    for name in ('tokenizer.json','tokenizer_config.json'):
        p=root/'processor'/name;p.write_text('{}')
        assets['processor/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=root/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(files=dict(assets,**{'execution.pt':'a'*64}),policy_id='old')))
    first=module.processor_cache_identity(root)
    manifest.write_text(json.dumps(dict(files=dict(assets,**{'execution.pt':'b'*64}),policy_id='new')))
    assert module.processor_cache_identity(root)==first==content_hash(assets)
    (root/'processor/tokenizer.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='processor assets changed'):
        module.processor_cache_identity(root)
