from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('verl')
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from probe_omni_pretrained_verl import diagnostic_batch


def test_real_dataproto_keeps_mixed_audio_features_and_causal_boundary():
    processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    prepared={('a','d'):dict(input_ids=torch.tensor([4,5]),input_features=torch.arange(24.).reshape(2,4,3),
                           feature_attention_mask=torch.tensor([[1,1,1],[1,1,0]])),
              ('b','d'):dict(input_ids=torch.tensor([6,7,8]),input_features=torch.ones(1,4,5),
                           feature_attention_mask=torch.ones(1,5,dtype=torch.long))}
    keys=[('a','d'),('b','d')];batch=diagnostic_batch(processor,prepared,keys,[10,11])
    assert batch.batch['input_ids'].tolist()==[[0,4,5,10,11],[6,7,8,10,11]]
    assert batch.batch['position_ids'].tolist()==[[1,0,1,2,3],[0,1,2,3,4]]
    assert batch.batch['input_ids'][:,-3:-1].tolist()==[[5,10],[8,10]]
    assert batch.meta_info['synthetic_old_log_probs'] is True
    from verl.utils.model import extract_multi_modal_inputs
    merged=extract_multi_modal_inputs(batch.non_tensor_batch['multi_modal_inputs'])
    assert merged['feature_attention_mask'].sum(-1).tolist()==[3,2,5]
    torch.testing.assert_close(merged['input_features'][:2,:,:3],prepared[keys[0]]['input_features'])
    assert not merged['input_features'][:2,:,3:].any()


def test_diagnostic_batch_refuses_truncation_or_missing_pad():
    processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    item=dict(input_ids=torch.ones(4096,dtype=torch.long),input_features=torch.ones(1,4,5),
              feature_attention_mask=torch.ones(1,5,dtype=torch.long))
    with pytest.raises(ValueError,match='overflow'):diagnostic_batch(processor,{('a','d'):item},[('a','d')],[1])
    item['input_ids']=torch.ones(2,dtype=torch.long);processor.tokenizer.pad_token_id=None
    with pytest.raises(ValueError,match='pad token'):diagnostic_batch(processor,{('a','d'):item},[('a','d')],[1])
