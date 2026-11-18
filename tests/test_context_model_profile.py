import json
from pathlib import Path

import pytest

from multimodal_jev.context_model_profile import (
    GEOMETRY, SCHEMA, build_head, from_hf, record, validate_against_config)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'configs/experiments/context-two-stage-v0.0.11.json'
MODELS = Path('/mnt/cephfs/xingchao/models')
SMALL = MODELS/'Qwen__Qwen3-ASR-0.6B-hf'
LARGE = MODELS/'Qwen__Qwen3-ASR-1.7B-hf'

# Declared geometry copied from the published HF configs; the tests below build
# their own config.json files so they never depend on a mounted model.
SMALL_CONFIG = dict(
    model_type='qwen3_asr',
    text_config=dict(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
                     num_key_value_heads=8, head_dim=128, intermediate_size=3072,
                     vocab_size=151936, layer_types=['full_attention']*28),
    audio_config=dict(d_model=896, encoder_layers=18, output_dim=1024))
LARGE_CONFIG = dict(
    model_type='qwen3_asr',
    text_config=dict(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                     num_key_value_heads=8, head_dim=128, intermediate_size=6144,
                     vocab_size=151936, layer_types=['full_attention']*28),
    audio_config=dict(d_model=1024, encoder_layers=24, output_dim=2048))


def write_model(root, config):
    path = Path(root)/'model'
    path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps(config))
    return path


def frozen_targets():
    design = json.loads(DESIGN.read_text())
    return design['stage1']['lora']['target_modules']


def test_small_profile_reproduces_frozen_lora_list_and_head(tmp_path):
    profile = from_hf(write_model(tmp_path, SMALL_CONFIG))
    assert profile.lora_targets() == frozen_targets()
    assert len(profile.lora_targets()) == 56
    assert profile.decision_head() == dict(hidden_size=1024, num_heads=8, ffn_size=1024,
                                           residual_scale=0.1)
    assert (profile.audio_d_model, profile.audio_encoder_layers,
            profile.audio_output_dim) == (896, 18, 1024)


def test_large_model_same_targets_with_wide_head(tmp_path):
    profile = from_hf(write_model(tmp_path, LARGE_CONFIG))
    assert profile.lora_targets() == frozen_targets()
    assert profile.decision_head() == dict(hidden_size=2048, num_heads=8, ffn_size=2048,
                                           residual_scale=0.1)
    assert (profile.audio_d_model, profile.audio_encoder_layers,
            profile.audio_output_dim) == (1024, 24, 2048)


def test_targets_follow_layer_count_not_checkpoint_name(tmp_path):
    config = json.loads(json.dumps(SMALL_CONFIG))
    config['text_config']['num_hidden_layers'] = 2
    config['text_config']['layer_types'] = ['full_attention']*2
    assert from_hf(write_model(tmp_path, config)).lora_targets() == [
        'language_model.layers.0.self_attn.q_proj', 'language_model.layers.0.self_attn.v_proj',
        'language_model.layers.1.self_attn.q_proj', 'language_model.layers.1.self_attn.v_proj']


def test_identity_changes_with_any_geometry_or_config_byte(tmp_path):
    profile = from_hf(write_model(tmp_path, SMALL_CONFIG))
    first = profile.identity()
    assert set(first) == set(GEOMETRY) | {
        'schema', 'model_type', 'config_sha256', 'implementation_sha256', 'geometry_sha256'}
    assert first['schema'] == SCHEMA
    other = tmp_path/'other'
    other.mkdir()
    changed = json.loads(json.dumps(SMALL_CONFIG))
    changed['text_config']['intermediate_size'] = 4096
    assert from_hf(write_model(other, changed)).identity()['geometry_sha256'] != \
        first['geometry_sha256']
    again = tmp_path/'again'
    again.mkdir()
    profile2 = from_hf(write_model(again, SMALL_CONFIG))
    assert profile2.identity() == first


@pytest.mark.parametrize('mutate', [
    lambda c: c.pop('audio_config'),
    lambda c: c['text_config'].pop('hidden_size'),
    lambda c: c['text_config'].update(hidden_size='1024'),
    lambda c: c['text_config'].update(hidden_size=1023),
    lambda c: c['text_config'].update(num_hidden_layers=1.0),
    lambda c: c['text_config'].update(num_key_value_heads=7),
    lambda c: c['text_config'].update(layer_types=['full_attention']*27),
    lambda c: c['text_config'].update(layer_types=['sliding_attention']*28),
    lambda c: c['audio_config'].update(output_dim=0),
])
def test_malformed_configs_rejected(tmp_path, mutate):
    config = json.loads(json.dumps(SMALL_CONFIG))
    mutate(config)
    with pytest.raises(ValueError):
        from_hf(write_model(tmp_path, config))


def test_symlinked_model_directory_rejected(tmp_path):
    path = write_model(tmp_path, SMALL_CONFIG)
    link = tmp_path/'link'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        from_hf(link)


def test_build_head_matches_declared_dimensions(tmp_path):
    head = build_head(from_hf(write_model(tmp_path, LARGE_CONFIG)))
    assert head.hidden_size == 2048
    assert head.ffn[0].in_features == head.ffn[0].out_features == 2048
    assert head.projection.out_features == 1


def test_frozen_design_config_is_consistent(tmp_path):
    profile = from_hf(write_model(tmp_path, SMALL_CONFIG))
    design = json.loads(DESIGN.read_text())
    report = validate_against_config(profile, design)
    assert report['status'] == 'consistent'
    assert report['compared'] == {'stage1.lora.target_modules': 56,
                                  'stage2.lora.target_modules': 56}


def test_same_lora_block_valid_for_both_sizes_but_declared_profile_pins_one(tmp_path):
    # Identical q/v structure means one LoRA block serves both checkpoints; only
    # an explicit model_profile block binds a config to a single geometry.
    design = json.loads(DESIGN.read_text())
    for config in (SMALL_CONFIG, LARGE_CONFIG):
        model = tmp_path/str(config['text_config']['hidden_size'])
        model.mkdir()
        assert validate_against_config(from_hf(write_model(model, config)),
                                       design)['status'] == 'consistent'
    larger = from_hf(write_model(tmp_path/'declared', LARGE_CONFIG))
    design['model_profile'] = from_hf(write_model(tmp_path/'small', SMALL_CONFIG)).geometry()
    with pytest.raises(ValueError):
        validate_against_config(larger, design)


def test_declared_targets_from_a_different_layer_count_rejected(tmp_path):
    profile = from_hf(write_model(tmp_path, SMALL_CONFIG))
    design = json.loads(DESIGN.read_text())
    design['stage1']['lora']['target_modules'] = design['stage1']['lora']['target_modules'][:-2]
    with pytest.raises(ValueError):
        validate_against_config(profile, design)


def test_declared_profile_mismatch_and_unknown_keys_rejected(tmp_path):
    profile = from_hf(write_model(tmp_path, SMALL_CONFIG))
    design = json.loads(DESIGN.read_text())
    design['model_profile'] = dict(profile.geometry())
    assert validate_against_config(profile, design)['compared']['model_profile']
    design['model_profile']['hidden_size'] = 2048
    with pytest.raises(ValueError):
        validate_against_config(profile, design)
    design['model_profile'] = dict(profile.geometry(), num_experts=8)
    with pytest.raises(ValueError):
        validate_against_config(profile, design)


def test_record_is_json_serialisable(tmp_path):
    row = record(from_hf(write_model(tmp_path, SMALL_CONFIG)))
    assert json.loads(json.dumps(row))['decision_head']['hidden_size'] == 1024


@pytest.mark.skipif(not SMALL.is_dir() or not LARGE.is_dir(),
                    reason='published model checkpoints not mounted')
@pytest.mark.parametrize('path,hidden,audio', [(SMALL, 1024, 896), (LARGE, 2048, 1024)])
def test_published_checkpoints_match_declared_geometry(path, hidden, audio):
    profile = from_hf(path)
    assert profile.hidden_size == hidden and profile.audio_d_model == audio
    assert profile.lora_targets() == frozen_targets()
