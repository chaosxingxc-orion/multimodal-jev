#!/usr/bin/env python3
"""Execute pinned source methods on synthetic tensors, without loading weights.

AST extraction removes decorators only. This checks local method behavior, not
model forward, processor integration, distributed execution or training.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def method(path, name, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    import torch
    scope = {'torch': torch, 'NonTensorData': type('NonTensorData', (), {})}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope[name]


def run(root):
    import torch
    torch.set_num_threads(2)
    hf = root / 'transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py'
    vf = root / 'verl/utils/model.py'
    tree = ast.parse(hf.read_text())
    owner = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                 and any(isinstance(m, ast.FunctionDef) and m.name == 'get_rope_index' for m in n.body))
    rope = method(hf, 'get_rope_index', owner)
    obj = SimpleNamespace(spatial_merge_size=2, config=SimpleNamespace(
        image_token_id=10, video_token_id=11, audio_token_id=12,
        vision_start_token_id=13, audio_start_token_id=14, position_id_per_seconds=25))
    # Two audio spans in row zero; one in row one, with left and right padding.
    ids = torch.tensor([[0, 14, 12, 12, 15, 14, 12, 15, 7, 8],
                        [0, 0, 0, 14, 12, 12, 15, 7, 8, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
    positions, delta = rope(obj, input_ids=ids, attention_mask=mask)
    expected = mask.cumsum(-1) - 1
    expected.masked_fill_(mask == 0, 1)
    assert torch.equal(positions, expected.unsqueeze(0).expand(3, -1, -1))
    assert torch.count_nonzero(delta) == 0

    extract = method(vf, 'extract_multi_modal_inputs')
    a = {'input_features': torch.ones(2, 4, 5), 'feature_attention_mask': torch.ones(2, 5, dtype=torch.long)}
    b = {'input_features': torch.full((1, 4, 7), 2.), 'feature_attention_mask': torch.ones(1, 7, dtype=torch.long)}
    try:
        extract([a, b])
    except RuntimeError as exc:
        unpadded_error = str(exc)
    else:
        raise AssertionError('unequal feature widths unexpectedly accepted')
    padded_a = {k: torch.nn.functional.pad(v, (0, 2)) for k, v in a.items()}
    merged = extract([padded_a, b])
    assert merged['input_features'].shape == (3, 4, 7)
    lengths = merged['feature_attention_mask'].sum(-1)
    assert lengths.tolist() == [5, 5, 7]

    get_audio = method(hf, 'get_audio_features', 'Qwen3OmniMoeThinkerForConditionalGeneration')
    captured = {}
    def tower(features, **kwargs):
        captured.update(shape=list(features.shape), lengths=kwargs['feature_lens'].tolist())
        return 'synthetic-tower-result'
    fake = SimpleNamespace(audio_tower=tower)
    assert get_audio(fake, **merged) == 'synthetic-tower-result'
    assert captured == {'shape': [4, 17], 'lengths': [5, 5, 7]}
    try:
        get_audio(fake, merged['input_features'], audio_feature_lengths=lengths)
    except AttributeError as exc:
        lengths_only_error = str(exc)
    else:
        raise AssertionError('lengths-only source path changed; re-audit required')
    return {
        'scope': 'AST-extracted pinned methods; synthetic tensors and mocked audio tower; no model forward or training',
        'torch_version': torch.__version__,
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (hf, vf)},
        'audio_only_rope_equals_cumulative_positions': True,
        'rope_shape': list(positions.shape), 'rope_deltas': delta.tolist(),
        'unpadded_batch_error': unpadded_error, 'padded_audio_lengths': lengths.tolist(),
        'audio_tower_input': captured, 'lengths_only_error': lengths_only_error,
        'model_loaded': False, 'weight_updates': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source_root)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
