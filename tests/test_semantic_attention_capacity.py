from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_semantic_attention_capacity import tensor_budget


def test_explicit_tensor_bytes_quadratic_and_not_divided_by_dp():
    torch=pytest.importorskip('torch')
    length=17;heads=16
    score=torch.empty(1,heads,length,length,dtype=torch.bfloat16)
    probability=score.softmax(-1,dtype=torch.float32)
    r=tensor_budget(length,heads)
    assert r['score_plus_softmax_bytes']==score.numel()*score.element_size()+probability.numel()*probability.element_size()
    assert tensor_budget(length*2,heads)['score_plus_softmax_bytes']==4*r['score_plus_softmax_bytes']
    assert tensor_budget(length,heads,batch=4)['score_plus_softmax_bytes']==4*r['score_plus_softmax_bytes']


def test_long_input_fails_nominal_capacity_even_without_model_allocations():
    assert tensor_budget(32768)['score_plus_softmax_gib']==96
    assert tensor_budget(30972)['exceeds_nominal_capacity']
    assert not tensor_budget(8192)['exceeds_nominal_capacity']
    for n in (0,-1,1.5):
        with pytest.raises(ValueError):tensor_budget(n)
