#!/usr/bin/env python3
"""Original BF16 decoder layer capacity probe; CPU load preflight or explicit NPU.

Consumes all nine original TRAIN6 prefix calls. No compact-attention/FP32 patch,
no decoder replay beyond layer zero, no optimizer or full-actor parity claim.
"""
import argparse
import inspect
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash

PREFIX_RESULT = 'fb10bfc904f23e4ebb408ce3a685c9eac66a55f3bfd48b50dfae95443c5e08e7'
HF_SOURCE = 'a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'


def load_original_layer(model):
    import torch
    from safetensors import safe_open
    from transformers import initialization
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import dot_natural_key
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    read = lambda p: json.loads(p.read_text())
    config = Qwen3OmniMoeTextConfig(**read(model / 'config.json')['thinker_config']['text_config'])
    config._attn_implementation = 'sdpa'
    # A standalone Experts(config) skips PreTrainedModel's default implementation
    # selection. Build the original text model on meta, then materialize layer 0 only.
    with torch.device('meta'), initialization.no_init_weights():
        skeleton = hf.Qwen3OmniMoeThinkerTextModel(config).to(torch.bfloat16)
    layer = skeleton.layers[0]
    del skeleton
    if config._experts_implementation != 'grouped_mm':
        raise ValueError('unexpected default experts dispatch; inspect before executing')
    old_prefix = 'model.layers.0.'
    conversion_info = []
    with safe_open(model / 'model-00001-of-00015.safetensors', framework='pt', device='cpu') as f:
        old = {k.removeprefix('thinker.'): f.get_tensor(k) for k in f.keys() if k.startswith('thinker.' + old_prefix)}
        converted = {}
        used = set()
        for converter in get_checkpoint_conversion_mapping('qwen3_omni_moe_thinker'):
            target = None
            for name in sorted(old, key=dot_natural_key):
                renamed, pattern = converter.rename_source_key(name)
                if pattern is not None:
                    if target is None:
                        target = renamed
                    if target != renamed:
                        raise ValueError('unexpected conversion target')
                    converter.add_tensor(renamed, name, pattern, old[name])
                    used.add(name)
            if target is None:
                raise ValueError('expected layer0 expert conversion missing')
            converted.update(converter.convert(target))
            conversion_info.append(dict(sources=converter.source_patterns, targets=converter.target_patterns,
                                        operations=[type(op).__name__ for op in converter.operations]))
        for i in range(config.num_experts):
            p = old_prefix + f'mlp.experts.{i}.'
            gate_up = converted[old_prefix + 'mlp.experts.gate_up_proj'][i]
            down = converted[old_prefix + 'mlp.experts.down_proj'][i]
            for a, b in [(gate_up[:config.moe_intermediate_size], old[p + 'gate_proj.weight']),
                         (gate_up[config.moe_intermediate_size:], old[p + 'up_proj.weight']),
                         (down, old[p + 'down_proj.weight'])]:
                if not torch.equal(a, b):
                    raise ValueError('converted expert differs from original checkpoint')
        state = {k.removeprefix(old_prefix): v for k, v in old.items() if k not in used}
        state.update({k.removeprefix(old_prefix): v for k, v in converted.items()})
        layer.load_state_dict(state, strict=True, assign=True)
        del old, converted, state
    layer.eval().requires_grad_(False)
    if any(p.is_meta or p.dtype != torch.bfloat16 or p.device.type != 'cpu' for p in layer.parameters()):
        raise ValueError('complete original BF16 CPU layer required')
    return layer, config, conversion_info


def move_inputs(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(move_inputs(v, device) for v in value)
    if isinstance(value, list):
        return [move_inputs(v, device) for v in value]
    if value is None:
        return None
    raise ValueError('unexpected non-tensor prefix value')


def run(args):
    import torch
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.integrations import moe
    torch.set_num_threads(2)
    begin = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    progress = lambda phase, **kw: atomic_json(args.output/'progress.json', dict(phase=phase, heartbeat_unix=time.time(), **kw))
    if file_hash(inspect.getfile(hf)) != HF_SOURCE or file_hash(args.prefix/'result.json') != PREFIX_RESULT:
        raise ValueError('original source and complete real TRAIN6 prefix required')
    result = read(args.prefix/'result.json')
    if file_hash(args.prefix/'identity.json') != result['identity_sha256']:
        raise ValueError('original prefix identity changed')
    identity = read(args.prefix/'identity.json')
    expected = [(s, o) for s in (1, 2) for o in range(0, 6, s)]
    if identity['order'] != [2, 0, 5, 1, 3, 4] or [(r['micro_batch_size'], r['offset']) for r in result['records']] != expected:
        raise ValueError('original complete nine-call order required')
    pins = {r['file']: r['sha256'] for r in identity['model_files']}
    names = ['config.json', 'model-00001-of-00015.safetensors']
    progress('verify_weights')
    for name in names:
        if file_hash(args.model/name) != pins[name]:
            raise ValueError('original layer weights changed')
    progress('load_original_layer_cpu')
    layer, config, conversions = load_original_layer(args.model)
    if sum(p.numel() for p in layer.parameters()) != 623120640:
        raise ValueError('complete original first decoder layer required')
    records = []
    for r in result['records']:
        p = args.prefix/r['tensor_file']
        if file_hash(p) != r['tensor_sha256']:
            raise ValueError('prefix tensor changed')
        inp = torch.load(p, map_location='cpu', weights_only=True)
        hidden = inp['hidden_states'];mask = inp['token_mask']
        if (hidden.dtype != torch.bfloat16 or hidden.shape != (r['micro_batch_size'], 1230, 2048)
                or mask.shape != hidden.shape[:2] or not torch.isfinite(hidden).all()
                or not torch.all((mask == 0) | (mask == 1))):
            raise ValueError('complete original padded TRAIN state required')
        records.append(dict(size=r['micro_batch_size'],offset=r['offset'],input_file=p.name,
            input_sha256=r['tensor_sha256'],input_shape=list(hidden.shape),valid_lengths=mask.sum(-1).tolist()))
    atomic_json(args.output/'identity.json', dict(prefix_result_sha256=PREFIX_RESULT,
        prefix_identity_sha256=result['identity_sha256'],order=identity['order'],calls=records,
        model_files={n:pins[n] for n in names},parameters=sum(p.numel() for p in layer.parameters()),
        source_sha256={inspect.getfile(hf):file_hash(inspect.getfile(hf)),inspect.getfile(moe):file_hash(inspect.getfile(moe)),'runner':file_hash(__file__)},
        conversions=conversions,expert_tensors_exact=384,experts_implementation=config._experts_implementation,
        preflight_only=args.preflight_only,attention='original SDPA with saved original mask',weight_updates=0))
    outputs = []
    if not args.preflight_only:
        import torch_npu
        if not torch.npu.is_available() or torch.npu.device_count() != 1:
            raise ValueError('coordinator must expose exactly one authorized NPU')
        torch.npu.set_device(0)
        torch.npu.reset_peak_memory_stats()
        progress('move_layer_npu')
        layer.to('npu:0')
        for record in records:
            progress('npu_forward',size=record['size'],offset=record['offset'])
            inp = torch.load(args.prefix/record['input_file'],map_location='cpu',weights_only=True)
            actual = {k:move_inputs(v,'npu:0') for k,v in inp.items() if k!='token_mask'}
            start = time.monotonic()
            with torch.inference_mode(),torch.autocast('npu',dtype=torch.bfloat16):
                out = layer(actual['hidden_states'],attention_mask=actual['attention_mask'],
                    position_ids=actual['position_ids'],position_embeddings=actual['position_embeddings'],use_cache=False)
            torch.npu.synchronize()
            if not torch.isfinite(out).all():
                raise ValueError('nonfinite NPU layer output')
            path = args.output/f"micro-{record['size']}-offset-{record['offset']}.pt"
            torch.save(dict(output=out.cpu(),token_mask=inp['token_mask']),path)
            report = dict(**record,output_file=path.name,output_sha256=file_hash(path),
                seconds=time.monotonic()-start,peak_allocated_bytes=torch.npu.max_memory_allocated(),
                peak_reserved_bytes=torch.npu.max_memory_reserved())
            atomic_json(path.with_suffix('.json'),report);outputs.append(report)
            del actual,out,inp
    progress('verify_weights_after')
    for name in names:
        if file_hash(args.model/name) != pins[name]:
            raise ValueError('weights changed during probe')
    atomic_json(args.output/'result.json',dict(status='preflight_complete' if args.preflight_only else 'diagnostic_complete',
        identity_sha256=file_hash(args.output/'identity.json'),outputs=outputs,calls=len(records),
        preflight_only=args.preflight_only,npu_forward_calls=len(outputs),weight_updates=0,
        elapsed_seconds=time.monotonic()-begin,
        scope='Original first-layer materialization and prefix validation; optional real NPU compatibility/capacity only. No full actor, FSDP, on-policy probability, optimizer or training acceptance.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('model','prefix','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
