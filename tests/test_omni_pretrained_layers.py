from pathlib import Path
import sys
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from diagnose_omni_pretrained_layers import pair_inputs,complete_dependency


def test_pair_changes_only_masked_text_padding():
    item=dict(input_ids=torch.tensor([2,3,4]),input_features=torch.arange(12.).reshape(1,4,3),
              feature_attention_mask=torch.ones(1,3,dtype=torch.long))
    original=item['input_features'].clone()
    a,b=pair_inputs(item,[8,9],7,5,99)
    assert a['input_ids'].tolist()==[[2,3,4,8,9]]
    assert b['input_ids'].tolist()==[[99,99,99,99,2,3,4,8,9]]
    assert b['position_ids'][b['attention_mask'].bool()].tolist()==[0,1,2,3,4]
    assert torch.equal(a['input_features'],b['input_features'])
    assert torch.equal(item['input_features'],original)
    assert a['input_features'].dtype==torch.float32


def test_full_run_cannot_use_missing_terminal_evidence(tmp_path):
    with pytest.raises(ValueError,match='terminal inspection'):
        complete_dependency(tmp_path,None,None)
