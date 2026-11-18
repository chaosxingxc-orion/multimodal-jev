import copy
import json
from pathlib import Path

import pytest
import torch

from tools.reports import context_train_parity_inputs as module


def evidence():
    return {f's{i:04d}':dict(tensors=dict(input_ids={'shape':[1,100+i]},
        input_features={'shape':[1,128,100+(i*73)%1200]})) for i in range(512)}


def test_panel_is_order_invariant_and_includes_both_extrema():
    rows=evidence();original=copy.deepcopy(rows)
    a=module.select(rows);b=module.select(dict(reversed(list(rows.items()))))
    assert a==b and rows==original and len(a['ids'])==256
    for axis in a['extremes'].values():
        assert set(axis['shortest']+axis['longest'])<=set(a['ids'])
    assert a['rule']['labels_used'] is False


@pytest.mark.parametrize('fault',['partial','text','audio','shape'])
def test_panel_rejects_incomplete_or_unbounded_inputs(fault):
    rows=evidence()
    if fault=='partial':rows={k:v for k,v in list(rows.items())[:255]}
    elif fault=='text':rows['s0000']['tensors']['input_ids']['shape']=[1,4097]
    elif fault=='audio':rows['s0000']['tensors']['input_features']['shape']=[1,128,0]
    else:rows['s0000']['tensors']['input_features']['shape']=[128,100]
    with pytest.raises(ValueError):module.select(rows)


def fixture(tmp_path,monkeypatch):
    from multimodal_jev.cached_context_preparation import CachedContextPreparer
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache,key_for
    from multimodal_jev.span_request_identity import tensor_identity
    import multimodal_jev.local_checkpoint as ckpt
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(ckpt,'check_paths',lambda _:None)
    candidates=[dict(id=f'c{i}',definition=f'candidate {i}') for i in range(60)]
    request=module.compile_candidates('/not-present/audio.wav',candidates,[],semantics='exclusive_event')
    cache=EncodedAudioCache(tmp_path/'cache')
    p=CachedContextPreparer(lambda:(_ for _ in ()).throw(AssertionError('no processor')),cache,
        processor_sha256='a'*64,decision_token_id=99,audio_token_id=98,max_tokens=4096)
    identity=p.identity(request,'b'*64)
    ids=torch.tensor([[1,98]+[99]*60]);positions=torch.arange(2,62)[None]
    inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        input_features=torch.ones(1,128,100,dtype=torch.bfloat16),
        input_features_mask=torch.ones(1,100,dtype=torch.int32))
    cache_path=cache.write(identity,inputs,positions,{'fixture':True})
    receipt=dict(id='sample',audio_sha256='b'*64,request_sha256=module.content_hash(request.model_input()),
        cache_key=key_for(identity),cache_sha256=module.file_hash(cache_path),
        tensors={k:tensor_identity(v) for k,v in dict(inputs,positions=positions).items()})
    names=[f'sample{i:03d}' for i in range(256)]
    values=dict(panel=dict(rule=module.RULE,ids=names,population=40513),
        inputs=[dict(id=name,request=json.loads(json.dumps(request.model_input())),receipt=dict(receipt,id=name)) for name in names],
        candidate_ids=[v['id'] for v in candidates],source={'processor_sha256':'a'*64,'release_sha256':'c'*64})
    monkeypatch.setattr(module,'reconstruct',lambda c:copy.deepcopy(values))
    c=dict(output=str(tmp_path/'panel'),release=str(tmp_path/'release'),input_proof=str(tmp_path/'proof'),
        input_receipts=str(tmp_path/'receipts'),cache_root=str(cache.root))
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    return c,values,inputs,cache_path


def test_real_readonly_cache_encoder_checks_tensors_without_audio_or_processor(tmp_path,monkeypatch):
    c,_,inputs,_=fixture(tmp_path,monkeypatch)
    assert module.audit(c,{})['recordings']==256
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assets,encode=module.encoder(c,processor_sha256='a'*64,decision_token_id=99,audio_token_id=98)
    value,receipt=encode(assets[0])
    assert torch.equal(value['inputs']['input_ids'],inputs['input_ids'])
    assert len(receipt['positions'])==60
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['processor','cache','tensor','request','extra','panel','label'])
def test_changed_cache_or_resealed_wrong_panel_fails(tmp_path,monkeypatch,fault):
    c,values,_,cache_path=fixture(tmp_path,monkeypatch);root=Path(c['output'])
    if fault=='cache':cache_path.write_bytes(b'changed')
    if fault in ('tensor','request','panel','label'):
        changed=copy.deepcopy(values)
        if fault=='tensor':changed['inputs'][0]['receipt']['tensors']['input_ids']['sha256']='e'*64
        elif fault=='request':changed['inputs'][0]['request']['requirements'].append('changed')
        elif fault=='label':changed['inputs'][0]['receipt']['intent']='gold'
        else:changed['panel']['ids']=['another']
        (root/'inputs.json').write_text(json.dumps(changed))
        m=json.loads((root/'MANIFEST.json').read_text());m['files']['inputs.json']=module.file_hash(root/'inputs.json')
        (root/'MANIFEST.json').write_text(json.dumps(m));c['manifest_sha256']=module.file_hash(root/'MANIFEST.json')
        with pytest.raises(ValueError):module.audit(c,{})
    if fault=='extra':(root/'extra.json').write_text('{}')
    with pytest.raises(ValueError):
        assets,encode=module.encoder(c,processor_sha256='c'*64 if fault=='processor' else 'a'*64,
            decision_token_id=99,audio_token_id=98)
        encode(assets[0])
