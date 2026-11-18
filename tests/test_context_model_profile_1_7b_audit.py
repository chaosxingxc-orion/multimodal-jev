"""The 1.7B audit's own helpers, independent of whether the models are mounted.

The accepted report proves one run. These tests pin the pieces that decide that
verdict: the design diff must report a model-identity block as two paths and a
real algorithmic difference as its own path, the safetensors header reader must
derive q/v widths from geometry rather than assume one symmetric matrix, and the
snapshot roundtrip must carry the asymmetric v_proj output through to the native
payload.
"""
import json
import struct

import pytest

from multimodal_jev.context_model_profile import ModelProfile
from tools.reports import audit_context_model_profile_1_7b as audit

PAYLOAD = 2


def profile(hidden=1024, layers=2, heads=16, kv_heads=8, head_dim=64):
    return ModelProfile(hidden_size=hidden, num_hidden_layers=layers,
                        num_attention_heads=heads, num_key_value_heads=kv_heads,
                        head_dim=head_dim, intermediate_size=hidden * 3, vocab_size=151936,
                        audio_d_model=hidden - 128, audio_encoder_layers=18,
                        audio_output_dim=hidden, config_sha256='a'*64)


def safetensors(path, tensors):
    """Write a single-file header plus a placeholder body of the declared size."""
    header = {name: dict(dtype='BF16', shape=list(shape), data_offsets=[0, 0])
              for name, shape in tensors.items()}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b'\0' * 16)
    return path


def test_differences_names_each_model_identity_path_and_no_shared_field():
    small = dict(model='0.6B', stage1=dict(lora=dict(rank=4)))
    large = dict(model='1.7B', model_profile=dict(hidden_size=2048),
                 stage1=dict(lora=dict(rank=4)))
    assert audit.differences(small, large) == ['model', 'model_profile']
    assert audit.differences(dict(small), dict(small)) == []


def test_differences_descends_into_a_changed_shared_field():
    left = dict(stage1=dict(lora=dict(rank=4, alpha=8)))
    right = dict(stage1=dict(lora=dict(rank=4, alpha=16)))
    assert audit.differences(left, right) == ['stage1.lora.alpha']


def test_target_widths_take_q_and_v_from_their_own_head_counts():
    widths = audit.target_widths(profile(heads=16, kv_heads=8, head_dim=128, hidden=2048))
    assert set(widths) == set(profile().lora_targets())
    for name, shape in widths.items():
        assert shape[1] == 2048
        assert shape[0] == (2048 if '.q_proj' in name else 1024)


def test_checkpoint_targets_reports_a_width_mismatch_instead_of_skipping_it(tmp_path):
    p = profile(hidden=2048, heads=16, kv_heads=8, head_dim=128)
    targets = p.lora_targets()
    weights = safetensors(tmp_path/'model.safetensors', {
        'model.' + targets[0] + '.weight': (2048, 2048),   # q_proj, correct
        'model.' + targets[1] + '.weight': (2048, 1024),   # v_proj, transposed
        'model.audio_tower.layers.0.self_attn.q_proj.weight': (2048, 2048)})
    found = audit.checkpoint_targets(weights, p)
    assert list(found['present']) == [targets[0]]
    assert list(found['unexpected_width']) == [targets[1]]
    assert found['expected_counts'] == 4 and found['v_width'] == 1024
    # The audio tower's identically named projection is not a LoRA target.
    assert found['present'].keys().isdisjoint(
        {'audio_tower.layers.0.self_attn.q_proj'})


def test_header_rejects_symlinks_and_implausible_lengths(tmp_path):
    real = safetensors(tmp_path/'real.safetensors', {'a': (2, 2)})
    link = tmp_path/'link.safetensors'
    link.symlink_to(real)
    with pytest.raises(ValueError, match='ordinary safetensors'):
        audit.header(link)
    huge = tmp_path/'huge.safetensors'
    huge.write_bytes(struct.pack('<Q', 200 * 1024**2))
    with pytest.raises(ValueError, match='implausible'):
        audit.header(huge)


def test_parameters_counts_from_the_header_and_excludes_metadata(tmp_path):
    weights = safetensors(tmp_path/'model.safetensors', {'a': (2, 3), 'b': (4,)})
    found = audit.parameters(weights)
    assert found['tensors'] == 2 and found['parameters'] == 10
    assert found['bf16_bytes'] == 20 and found['dtypes'] == ['BF16']


def test_roundtrip_carries_the_asymmetric_v_projection_into_the_payload(tmp_path,
                                                                       monkeypatch):
    # pytest's tmp_path is tmpfs; the storage policy only admits local disks for
    # hot writes. The admission check is P2's subject, not this roundtrip's.
    from multimodal_jev import context_policy_snapshot as snapshot_module
    monkeypatch.setattr(snapshot_module, 'check_paths', lambda paths: None)
    p = profile()
    result = audit.roundtrip(tmp_path, p, rank=4, alpha=8, token_id=151706,
                             seed=20261118)
    assert result['byte_identical'] and result['differing_tensors'] == []
    assert result['restored_tensors'] == result['exported_tensors'] + 1
    # q_proj writes 1024 rows, v_proj only kv_heads*head_dim = 512.
    assert result['payload_by_projection'] == {
        'q_proj': dict(A=[4, 1024], B=[1024, 4]),
        'v_proj': dict(A=[4, 1024], B=[512, 4])}
    assert result['payload_targets'] == 4 and result['payload_tensors'] == 8
    assert result['head_width'] == 1024 and result['payload_alpha'] == 8.0
    assert {tuple(shape) for shape in result['matrix_shapes']} == {
        ((4, 1024), (1024, 4)), ((4, 1024), (512, 4))}
    assert result['snapshot_bytes'] > 0 and len(result['execution_sha256']) == 64


def test_run_refuses_a_foreign_schema_and_a_missing_role(tmp_path):
    with pytest.raises(ValueError, match='1.7B profile audit schema'):
        audit.run({'schema': 'jev-context-model-geometry-audit-v1', 'output': str(tmp_path)},
                  {'p': 1})
    c = dict(schema=audit.SCHEMA, output=str(tmp_path/'out'),
             models=[dict(role='small')], lora={})
    with pytest.raises(ValueError, match='one small and one large'):
        audit.run(c, {'p': 1})
    assert not (tmp_path/'out').exists()


def test_not_exercised_names_a_blocked_step_for_every_missing_package():
    host = dict(peft=False, transformers=False, safetensors=False, vllm=False)
    steps = audit.not_exercised(host)
    assert len(steps) == len(audit.MISSING) + 1
    assert all(step['blocked_by'] and step['covered_instead'] for step in steps)
    assert steps[-1]['blocked_by'] == 'NPU dispatch is out of scope this round'
    approved = dict.fromkeys(audit.MISSING, True)
    assert len(audit.not_exercised(approved)) == 1


def test_allowed_differences_are_only_the_model_identity():
    assert audit.ALLOWED_DESIGN_DIFFERENCES == {'model', 'model_profile'}
    assert audit.LARGE_HIDDEN == 2048
