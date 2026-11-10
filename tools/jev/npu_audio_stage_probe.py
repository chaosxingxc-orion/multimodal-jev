#!/usr/bin/env python3
"""Capture real HF/vLLM acoustic stages without changing model outputs.

TRAIN-only diagnosis. Numerical differences are measurements, not a new
admission threshold. The sampler still checks exact expanded prompt IDs.
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def capture_first(module, name, destination):
    import torch
    path = Path(destination) / (name + '.pt')
    def hook(_module, _inputs, output):
        if not path.exists():
            tensor = output[0] if isinstance(output, tuple) else output
            torch.save(tensor.detach().cpu(), path)
    return module.register_forward_hook(hook)


def install_engine_hooks(model, destination):
    tower = model.audio_tower
    for name in ('conv2d1', 'conv2d2', 'conv2d3', 'conv_out', 'ln_post'):
        capture_first(getattr(tower, name), name, destination)
    capture_first(tower, 'projected', destination)
    return {'tower_class': type(tower).__module__ + '.' + type(tower).__name__}


def run(args):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import hf_prompt, load_audio
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.rollout import VllmAudioRollout
    from multimodal_jev.task_schema import JevExample
    config = json.loads(args.config.read_text())
    torch.set_num_threads(4); torch.manual_seed(17)
    wanted = set(config['example_ids'])
    examples = [JevExample.from_record(row) for row in
                map(json.loads, Path(config['manifest']).read_text().splitlines())
                if row['example_id'] in wanted]
    assert len(examples) == len(wanted) and all(ex.split == 'train' for ex in examples)
    assert len({ex.audio for ex in examples}) == 1, 'first-stage capture requires one shared TRAIN recording'
    for ex in examples:
        assert file_hash(ex.audio) == ex.metadata['audio_sha256']
    audios = [load_audio(ex.audio) for ex in examples]
    policy_id, files = checkpoint_identity(Path(config['checkpoint']))
    atomic_json(args.output/'identity.json', {'policy_id': policy_id, 'checkpoint_files': files,
        'config': config, 'manifest_sha256': file_hash(config['manifest'])})
    for name in ('hf', 'vllm'):
        (args.output/name).mkdir()
    atomic_json(args.output/'progress.json', {'phase': 'hf_audio', 'heartbeat_unix': time.time()})
    processor, model = load_runtime(config['checkpoint'], 'npu:0', config['engine']['dtype'])
    hooks = [capture_first(getattr(model.model.audio_tower, name), name, args.output/'hf')
             for name in ('conv2d1', 'conv2d2', 'conv2d3', 'conv_out', 'ln_post')]
    hooks.append(capture_first(model.model.multi_modal_projector, 'projected', args.output/'hf'))
    inputs = hf_prompt(processor, examples[0], audios[0])
    features = inputs['input_features'].to(device='npu:0', dtype=model.dtype)
    mask = inputs['input_features_mask'].to('npu:0')
    with torch.inference_mode():
        model.get_audio_features(features, mask, return_dict=True)
    atomic_json(args.output/'features.json', {'shape': list(features.shape), 'valid_frames': int(mask.sum()),
        'padding_abs_max': float(features[..., int(mask.sum()):].abs().max()) if mask.sum() < features.shape[-1] else 0.0})
    for hook in hooks: hook.remove()
    del model, features, mask, inputs
    gc.collect(); torch.npu.empty_cache()
    atomic_json(args.output/'progress.json', {'phase': 'vllm_load', 'heartbeat_unix': time.time()})
    runner = VllmAudioRollout(config['checkpoint'], processor, policy_id, config['engine'], 'actor_features')
    try:
        destination = str(args.output/'vllm')
        hook_result = runner.engine.apply_model(lambda model: install_engine_hooks(model, destination))
        atomic_json(args.output/'progress.json', {'phase': 'sampling', 'heartbeat_unix': time.time()})
        records = runner.sample(examples, audios, config['sampling'])
        atomic_json(args.output/'prompt-identity.json', runner.last_prompt_identity)
        with (args.output/'rollouts.jsonl').open('x') as handle:
            for row in records: handle.write(json.dumps(asdict(row))+'\n')
    finally:
        runner.close()
    comparisons = {}
    for name in ('conv2d1', 'conv2d2', 'conv2d3', 'conv_out', 'ln_post', 'projected'):
        hf = torch.load(args.output/'hf'/f'{name}.pt', weights_only=True).float()
        engine = torch.load(args.output/'vllm'/f'{name}.pt', weights_only=True).float()
        row = {'hf_shape': list(hf.shape), 'vllm_shape': list(engine.shape),
               'hf_sha256': file_hash(args.output/'hf'/f'{name}.pt'),
               'vllm_sha256': file_hash(args.output/'vllm'/f'{name}.pt')}
        if hf.shape == engine.shape:
            row.update(max_abs_error=float((hf-engine).abs().max()),
                       mean_abs_error=float((hf-engine).abs().mean()),
                       finite=bool(torch.isfinite(hf).all() and torch.isfinite(engine).all()))
        comparisons[name] = row
    return {'status': 'diagnostic_complete', 'stages': comparisons, 'engine_hooks': hook_result,
            'rollouts': len(records), 'rollouts_sha256': file_hash(args.output/'rollouts.jsonl'),
            'generation_budget': runner.last_generation_budget,
            'scope': 'single TRAIN recording acoustic-stage measurements and exact prompt identity; no RL admission'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args)
        atomic_json(args.output/'result.json', result)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
        raise
    atomic_json(args.output/'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})
