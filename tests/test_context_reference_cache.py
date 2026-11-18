import copy,hashlib,importlib.util,json
from pathlib import Path
import pytest
import torch
from multimodal_jev.span_request_identity import tensor_identity
spec=importlib.util.spec_from_file_location('context_reference_cache',Path(__file__).parents[1]/'tools/jev/context_reference_cache.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(tmp_path,monkeypatch,policy='a'*64):
    monkeypatch.setattr(m,'check_paths',lambda x:None)
    cache=m.ReferenceCache(tmp_path,policy_id=policy,base_id='b'*64,contract={'base_id':'b'*64})
    inputs={'input_ids':torch.tensor([[1,2,3]]),'input_features':torch.ones(1,2,3)};pos=torch.tensor([[1,2]])
    identity=dict(policy_id=policy,positions=[1,2],inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
    record=dict(policy_id=policy,request_identity=identity,request_id=cache.digest(identity),prob=[.3,.7])
    return cache,inputs,pos,record


def test_restart_reuses_fixed_reference_without_engine(tmp_path,monkeypatch):
    cache,x,p,r=fixture(tmp_path,monkeypatch);cache.write(r,x,p)
    other,_,_,_=fixture(tmp_path,monkeypatch)
    e=m.CachedReferenceEngine(other)
    assert e.score(x,p,audio_token_id=1)==r and e.hits==1 and e.misses==0
    y=copy.deepcopy(x);y['input_features'][0,0,0]=2
    with pytest.raises(ValueError,match='cache miss'):e.score(y,p,audio_token_id=1)


@pytest.mark.parametrize('fault',['stale','labels','actions','probability','identity','nan'])
def test_rejects_wrong_distribution_or_forbidden_data(tmp_path,monkeypatch,fault):
    cache,x,p,r=fixture(tmp_path,monkeypatch);path=cache.write(r,x,p)
    if fault=='stale':r['policy_id']='c'*64
    if fault in ('labels','actions'):r[fault]=[1,0]
    if fault=='nan':r['prob'][0]=float('nan')
    if fault in ('probability','identity'):
        data=json.loads(path.read_text())
        if fault=='probability':data['record']['prob'][0]=.9
        else:data['reference']['base_id']='c'*64
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):cache.read(x,p)
    else:
        with pytest.raises(ValueError):cache.write(r,x,p)


def test_collision_does_not_overwrite_and_new_reference_isolated(tmp_path,monkeypatch):
    cache,x,p,r=fixture(tmp_path,monkeypatch);path=cache.write(r,x,p);before=path.read_bytes()
    cache.write(r,x,p)
    changed=copy.deepcopy(r);changed['prob'][0]=.4
    with pytest.raises(ValueError,match='collision'):cache.write(changed,x,p)
    assert path.read_bytes()==before and not list(cache.namespace.glob('.reference-*'))
    newer,_,_,_=fixture(tmp_path,monkeypatch,policy='c'*64)
    assert newer.read(x,p) is None and newer.namespace!=cache.namespace
