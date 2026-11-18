import pytest
from multimodal_jev.span_adapter_export import adapter_key, legacy_broken_key
from multimodal_jev.span_adapter_export import checkpoint_export_key


@pytest.mark.parametrize('projection', ['q_proj', 'v_proj'])
@pytest.mark.parametrize('matrix', ['A', 'B'])
def test_native_and_peft_names_converge_without_losing_weight_separator(projection, matrix):
    tail = f'language_model.layers.12.self_attn.{projection}.lora_{matrix}'
    expected = 'base_model.model.model.'+tail+'.weight'
    native = '_fsdp_wrapped_module.backbone.'+tail+'.default._fsdp_wrapped_module.weight'
    peft = 'base_model.model.model.'+tail+'.default._fsdp_wrapped_module.weight'
    assert adapter_key(native, native=True) == expected
    assert adapter_key(peft) == expected
    assert adapter_key(expected) == expected
    assert legacy_broken_key(expected).endswith(f'lora_{matrix}weight')
    with pytest.raises(ValueError):
        adapter_key(legacy_broken_key(expected))


@pytest.mark.parametrize('name', ['backbone.audio_tower.lora_A.default.weight',
    'base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_A.default.weight',
    'base_model.model.model.language_model.layers.0.self_attn.q_proj.weight'])
def test_reject_unexpected_trainable_weights(name):
    with pytest.raises(ValueError):
        adapter_key(name)


def test_current_export_verification_does_not_require_historical_defect():
    key='base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight'
    assert checkpoint_export_key(key) == key
    assert checkpoint_export_key(key, legacy_repair=True) == legacy_broken_key(key)
    with pytest.raises(ValueError):
        checkpoint_export_key(key, legacy_repair='false')
