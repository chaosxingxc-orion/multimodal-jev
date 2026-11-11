#!/usr/bin/env python3
"""Verify actual CPU/meta loading and all Thinker checkpoint values before FSDP.

No FSDP wrapping, communication, forward pass, NPU or optimizer.
"""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import re
import resource
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_fsdp_loading import (
    build_meta_thinker, buffer_manifest, parameter_manifest, require_buffers, load_cpu_thinker,
)

PREFIX_IDENTITY = 'da5eeb4af8108237816ff5632dc95138a71208b8c200dc2cbb15bf1ab2693cea'
BUFFER_RESULT = '98d823f9dc1f82f9f9813035f58dd2c5bb04580a9a55101ef4dc642efc2df07b'
HF_SOURCE = 'a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'


def loading_report(value):
    if isinstance(value, dict):
        return {k:loading_report(v) for k,v in value.items()}
    if isinstance(value, set):
        return sorted(loading_report(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [loading_report(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError('unsupported HF loading report value')


def original_parameter_view(parameters, name, intermediate_size):
    """Read one original checkpoint tensor from the loaded fused layout."""
    expert = re.fullmatch(r'(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight', name)
    if expert is None:
        return parameters[name]
    prefix, index, projection = expert.groups()
    if projection == 'down':
        return parameters[prefix+'.down_proj'][int(index)]
    fused = parameters[prefix+'.gate_up_proj'][int(index)]
    return fused[:intermediate_size] if projection == 'gate' else fused[intermediate_size:]


def run(args):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    torch.set_num_threads(8)
    begin = time.monotonic()
    progress = lambda phase, **kw: atomic_json(args.output/'progress.json', dict(
        phase=phase, heartbeat_unix=time.time(), **kw))
    read = lambda p: json.loads(p.read_text())
    if (file_hash(args.prefix_identity) != PREFIX_IDENTITY
            or file_hash(args.buffer_reference/'result.json') != BUFFER_RESULT
            or file_hash(inspect.getfile(hf)) != HF_SOURCE):
        raise ValueError('registered actual model and reference evidence required')
    prior = read(args.prefix_identity)
    pins = {r['file']:r['sha256'] for r in prior['model_files']}
    if len(pins) != len(prior['model_files']):
        raise ValueError('unique complete original model inventory required')
    for name, pin in pins.items():
        if Path(name).name != name:
            raise ValueError('flat original model inventory required')
        progress('verify_original_assets', file=name)
        if file_hash(args.model/name) != pin:
            raise ValueError('original model file changed: '+name)
    buffer_result = read(args.buffer_reference/'result.json')
    if file_hash(args.buffer_reference/'buffers.pt') != buffer_result['artifacts']['buffers.pt']:
        raise ValueError('reference buffer tensors changed')
    reference = torch.load(args.buffer_reference/'buffers.pt', map_location='cpu', weights_only=True)['before']
    raw = read(args.model/'config.json')
    progress('build_meta_thinker')
    meta = build_meta_thinker(raw)
    expected_parameters = parameter_manifest(meta, device_type='meta')
    expected_buffers = buffer_manifest(meta)
    if sum(v['numel'] for v in expected_parameters.values()) != 31719205488:
        raise ValueError('full original Thinker required')
    if set(reference) != set(expected_buffers):
        raise ValueError('complete reference buffers required')
    for name, tensor in meta.named_buffers():
        if not torch.equal(tensor, reference[name]):
            raise ValueError('meta constructor buffer differs from accepted FP32 reference')
    atomic_json(args.output/'meta-parameters.json', expected_parameters)
    atomic_json(args.output/'buffers.json', expected_buffers)
    del meta
    progress('load_full_pretrained_cpu')
    model, loading = load_cpu_thinker(args.model)
    atomic_json(args.output/'loading-info.json', loading_report(loading))
    actual_parameters = parameter_manifest(model, device_type='cpu')
    if actual_parameters != expected_parameters:
        raise ValueError('CPU/meta complete parameter structure differs')
    require_buffers(model, expected_buffers)
    torch.save({n:v.detach().cpu().clone() for n,v in model.named_buffers()}, args.output/'loaded-buffers.pt')
    # Independently undo HF expert stacking by views and compare every original
    # tensor to safetensors payload, not merely parameter names or total counts.
    parameters = dict(model.named_parameters())
    index = read(args.model/'model.safetensors.index.json')['weight_map']
    checked = {}; counts = Counter(); elements = 0
    intermediate = model.config.text_config.moe_intermediate_size
    for shard in sorted(set(index.values())):
        progress('compare_loaded_checkpoint_values', shard=shard, checked=len(checked))
        with safe_open(args.model/shard, framework='pt', device='cpu') as handle:
            for name in sorted(handle.keys()):
                if not name.startswith('thinker.'):
                    continue
                if index.get(name) != shard or name in checked:
                    raise ValueError('checkpoint tensor index differs')
                source = handle.get_tensor(name)
                loaded = original_parameter_view(parameters, name.removeprefix('thinker.'), intermediate)
                if source.dtype != loaded.dtype or source.shape != loaded.shape or not torch.equal(source, loaded):
                    raise ValueError('loaded checkpoint tensor differs: '+name)
                checked[name] = dict(shape=list(source.shape), dtype=str(source.dtype), numel=source.numel())
                counts[shard] += 1; elements += source.numel()
    if set(checked) != {n for n in index if n.startswith('thinker.')} or elements != 31719205488:
        raise ValueError('complete original tensor payload coverage required')
    require_buffers(model, expected_buffers)
    if parameter_manifest(model, device_type='cpu') != expected_parameters:
        raise ValueError('model structure changed during payload verification')
    atomic_json(args.output/'checkpoint-values.json', dict(tensors=checked, per_shard=dict(counts),
                                                         parameters=elements, all_exact=True))
    del model, parameters, loaded, source
    for name, pin in pins.items():
        progress('verify_original_assets_after', file=name)
        if file_hash(args.model/name) != pin:
            raise ValueError('original model file changed during probe')
    result = dict(status='preflight_complete', parameters=elements, checkpoint_tensors=len(checked),
        source_sha256={inspect.getfile(hf):HF_SOURCE, 'runner':file_hash(__file__),
                      'loader':file_hash(inspect.getfile(build_meta_thinker))},
        prefix_identity_sha256=PREFIX_IDENTITY, buffer_reference_result_sha256=BUFFER_RESULT,
        model_files=pins, metadata_parameter_tensors=len(expected_parameters),
        parameter_dtype='torch.bfloat16', buffer_dtype='torch.float32',
        cpu_meta_parameter_structure_exact=True, all_checkpoint_values_exact=True,
        all_buffers_exact=True, weights_loaded=True, weight_updates=0, forward_calls=0,
        artifacts={n:file_hash(args.output/n) for n in ['meta-parameters.json','buffers.json',
                   'loading-info.json','loaded-buffers.pt','checkpoint-values.json']},
        elapsed_seconds=time.monotonic()-begin, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Complete real CPU Thinker load, meta parameter preparation, all original checkpoint value comparisons and preserved FP32 buffers only. No distributed rank/FSDP/HCCL/NPU forward, sampled probabilities, optimizer or task-quality acceptance.')
    atomic_json(args.output/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'prefix-identity', 'buffer-reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
