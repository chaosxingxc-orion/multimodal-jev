import json

import pytest
import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.card_acoustic_pack import encode, PackedAcoustics, SCHEMA
from multimodal_jev.intent_triplet_training import TripletStream


def test_exact_single_card_input_and_tampered_payload(tmp_path):
    prefix,suffix=torch.tensor([9,8]),torch.tensor([6,4])
    native=dict(input_ids=torch.tensor([[9,8,2,2,6,4]]),attention_mask=torch.ones(1,6),
                input_features=torch.tensor([[[1.,2.,-3.,4.]]],dtype=torch.bfloat16),
                input_features_mask=torch.tensor([[1,1,0,0]]))
    b,meta=encode(native,torch.tensor([[5]]),prefix,suffix,2,4)
    data=tmp_path/'part-0.bin'
    data.write_bytes(b)
    path=tmp_path/'index.json'
    path.write_text(json.dumps(dict(schema=SCHEMA,token=2,identity='key',entries={'a':[0,0,len(b),'sha',*meta]})))
    pin=dict(path=str(path),sha256=file_hash(path),identity='key')
    pack=PackedAcoustics(pin,{'a':{'audio_sha256':'sha'}},2)
    stream=TripletStream.__new__(TripletStream)
    stream.templates=[(torch.tensor([10]),torch.tensor([4]))]
    stream.collation=dict(audio_token_id=2,decision_token_id=4)
    old,new=(stream.render(x,0,'a') for x in (native,pack.read('a')))
    assert torch.equal(old['positions'],new['positions'])
    for k,v in old['inputs'].items():
        assert torch.equal(v,new['inputs'][k]) and v.dtype==new['inputs'][k].dtype
    with data.open('r+b') as f:
        f.write(b'xx')
    with pytest.raises(ValueError,match='changed'):
        pack.read('a')
    with pytest.raises(ValueError,match='population'):
        PackedAcoustics(pin,{'a':{'audio_sha256':'other'}},2)
