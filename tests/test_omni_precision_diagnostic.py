from pathlib import Path
import sys
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
from diagnose_omni_pretrained_precision import CELLS, cell_inputs


def test_factorial_chain_changes_one_factor_and_preserves_audio_and_tokens():
    item=dict(input_ids=torch.tensor([7,8,9]), input_features=torch.arange(24.).reshape(2,4,3)/7,
              feature_attention_mask=torch.tensor([[1,1,1],[1,1,0]]))
    original=item['input_features'].clone()
    previous=None
    for _, autocast, fp32, apad, tpad in CELLS:
        flags=(autocast,fp32,apad,tpad)
        if previous is not None:assert sum(a!=b for a,b in zip(previous,flags))==1
        previous=flags
        x=cell_inputs(item,[10,11],prompt_width=6,audio_width=5,
                      fp32_features=fp32,audio_padding=apad,text_padding=tpad)
        valid=x['attention_mask'].bool()
        assert x['input_ids'][valid].tolist()==[7,8,9,10,11]
        assert x['position_ids'][valid].tolist()==list(range(5))
        assert x['input_ids'][0,-3:-1].tolist()==[9,10]
        assert x['feature_attention_mask'].sum(-1).tolist()==[3,2]
        dtype=torch.float32 if fp32 else torch.bfloat16
        torch.testing.assert_close(x['input_features'][:,:,:3],original.to(dtype),rtol=0,atol=0)
        assert not x['input_features'][:,:,3:].any()
    torch.testing.assert_close(item['input_features'],original,rtol=0,atol=0)


@pytest.mark.parametrize('pwidth,awidth',[(2,5),(6,2)])
def test_rejects_truncation(pwidth,awidth):
    item=dict(input_ids=torch.tensor([7,8,9]),input_features=torch.ones(1,4,3),
              feature_attention_mask=torch.ones(1,3,dtype=torch.long))
    with pytest.raises(ValueError,match='truncate'):
        cell_inputs(item,[10,11],prompt_width=pwidth,audio_width=awidth,
                    fp32_features=True,audio_padding=True,text_padding=True)
