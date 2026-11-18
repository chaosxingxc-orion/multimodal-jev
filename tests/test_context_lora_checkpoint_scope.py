import pytest
import torch
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope


def test_only_lora_head_and_empty_shards_allowed():
    named=[('_fsdp_wrapped_module.backbone.language_model.layers.0.q_proj.lora_A.default.weight',torch.nn.Parameter(torch.ones(2,3))),
           ('_fsdp_wrapped_module.decision_head._fsdp_wrapped_module.module.projection.weight',torch.nn.Parameter(torch.empty(0)))]
    result=validate_trainable_scope(named)
    assert result['base_model_saved'] is False and result['local_numel']==dict(lora=6,head=0)


@pytest.mark.parametrize('name',['backbone.embed_tokens.weight','backbone.layers.0.q_proj.weight','backbone.audio_tower.conv1.weight','lm_head.weight'])
def test_accidentally_unfrozen_backbone_rejected(name):
    with pytest.raises(ValueError,match='base model tensor forbidden'):
        validate_trainable_scope([(name,torch.nn.Parameter(torch.ones(2)))])


def test_frozen_tensors_cannot_be_saved_as_optimizer_parameters():
    with pytest.raises(ValueError,match='frozen tensor'):
        validate_trainable_scope([('decision_head.projection.weight',torch.nn.Parameter(torch.ones(2),requires_grad=False))])
