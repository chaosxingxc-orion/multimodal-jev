import json

import pytest
import torch

from test_cached_context_preparation import setup
from multimodal_jev import local_checkpoint
from multimodal_jev.baseline import file_hash
from multimodal_jev.encoded_audio_cache import key_for
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports import verify_selection_macro_cache as module
from tools.reports.selection_macro_cache_operations import command,audit_receipts


def test_full_population_real_tensor_read_and_positions_export(tmp_path,monkeypatch):
    request,digest,factory,_,_=setup(tmp_path,monkeypatch)
    request.questions=['q']*60
    def encode(*a,**kw):
        ids=torch.tensor([[98]+[99]*60])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=torch.ones(1,4,8),
            input_features_mask=torch.ones(1,8,dtype=torch.long)),torch.arange(1,61).reshape(1,60)
    monkeypatch.setattr('multimodal_jev.cached_context_preparation.encode_context_request',encode)
    prepare=factory(limit=4096);value=prepare((request,digest));key=key_for(prepare.identity(request,digest))
    tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()}
    population=[dict(id=str(i),audio_sha256=digest,request_sha256=content_hash(request.model_input())) for i in range(9226)]
    receipts=[dict(a,cache_key=key,cache_sha256=file_hash(prepare.cache.root/(key+'.pt')),tensors=tensors) for a in population]
    root=tmp_path/'binding';root.mkdir()
    artifacts={}
    for name,v in [('population.json',population),('candidate-ids.json',[str(i) for i in range(60)]),('cache-receipts.json',receipts)]:
        p=root/name;p.write_text(json.dumps(v));artifacts[name]=file_hash(p)
    (root/'binding.json').write_text(json.dumps(dict(recordings=9226,processor_sha256='a'*64,artifacts=artifacts)))
    monkeypatch.setattr(module,'check_paths',lambda x:None);monkeypatch.setattr(local_checkpoint,'check_paths',lambda x:None)
    result=module.verify(root,prepare.cache.root,tmp_path/'output',binding_sha256=file_hash(root/'binding.json'),decision_token_id=99,audio_token_id=98)
    assert result['recordings']==9226
    got=json.loads((tmp_path/'output/input-receipts.json').read_text())
    assert len(got)==9226 and got[-1]['positions']==list(range(1,61))
    assert audit_receipts(root,tmp_path/'output',file_hash(root/'binding.json'))['recordings']==9226
    got[0]['positions'][0]=0
    (tmp_path/'output/input-receipts.json').write_text(json.dumps(got))
    reported=json.loads((tmp_path/'output/result.json').read_text())
    reported['input_receipts_sha256']=file_hash(tmp_path/'output/input-receipts.json')
    (tmp_path/'output/result.json').write_text(json.dumps(reported))
    with pytest.raises(ValueError,match='original independent'):audit_receipts(root,tmp_path/'output',file_hash(root/'binding.json'))
    (prepare.cache.root/(key+'.pt')).write_bytes(b'wrong')
    with pytest.raises(ValueError,match='bytes'):module.verify(root,prepare.cache.root,tmp_path/'bad',binding_sha256=file_hash(root/'binding.json'),decision_token_id=99,audio_token_id=98)


def test_cpu_command_has_readonly_input_and_no_devices():
    c=dict(remote_root='/data/user/cache-verify',cache_root='/data/user/shared-cache',driver_root='/usr/local/Ascend/driver',host='npu',tag='verify')
    argv=command(c)
    assert '--device' not in argv and '--network=none' in argv
    assert 'type=bind,src=/data/user/shared-cache,dst=/data/user/shared-cache,readonly' in argv
    c['cache_root']=c['remote_root']+'/cache'
    with pytest.raises(ValueError):command(c)
