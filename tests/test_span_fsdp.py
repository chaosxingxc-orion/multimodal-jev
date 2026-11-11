import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_fsdp import unshared_linear_modules


def test_shared_embedding_head_stays_in_common_unit():
    model=torch.nn.Module()
    model.embedding=torch.nn.Embedding(8,4)
    model.head=torch.nn.Linear(4,8,bias=False)
    model.head.weight=model.embedding.weight
    model.adapter=torch.nn.Linear(4,2,bias=False)
    selected=unshared_linear_modules(model)
    assert selected=={model.adapter}
    assert model.head.weight is model.embedding.weight
    model.head.weight=torch.nn.Parameter(model.head.weight.detach().clone())
    assert unshared_linear_modules(model)=={model.head,model.adapter}
