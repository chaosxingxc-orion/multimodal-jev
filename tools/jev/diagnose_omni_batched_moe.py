#!/usr/bin/env python3
"""Trace one real text decoder layer from saved actual verl prefix inputs."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_compact_attention import compact_text_attention


def run(args):
    import torch
    from safetensors import safe_open
    from transformers import initialization
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import dot_natural_key
    from transformers.integrations import moe
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

    torch.set_num_threads(16)
    start = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    if file_hash(inspect.getfile(hf)) != 'a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340':
        raise ValueError('original HF model source required')
    for path, pin in [
        (args.prefix / 'result.json', 'fb10bfc904f23e4ebb408ce3a685c9eac66a55f3bfd48b50dfae95443c5e08e7'),
        (args.trace / 'result.json', '23ee087e133334428c0e0fcc85b728a4f7459350954b32ef7a22b1b6a284a161'),
    ]:
        if file_hash(path) != pin:
            raise ValueError('accepted original captures required')
    prefix = read(args.prefix / 'result.json')
    if prefix['identity_sha256'] != file_hash(args.prefix / 'identity.json'):
        raise ValueError('prefix identity changed')
    identity = read(args.prefix / 'identity.json')
    if file_hash(inspect.getfile(inspect.unwrap(compact_text_attention))) != identity['source_files']['src/multimodal_jev/omni_compact_attention.py']:
        raise ValueError('original attention adapter required')
    trace_reports = read(args.trace / 'result.json')['reports']
    weights = {r['file']: r['sha256'] for r in identity['model_files']}
    for name in ('config.json', 'model-00001-of-00015.safetensors'):
        if file_hash(args.model / name) != weights[name]:
            raise ValueError('original published weights required')

    config = Qwen3OmniMoeTextConfig(**read(args.model / 'config.json')['thinker_config']['text_config'])
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
    with safe_open(args.model / 'model-00001-of-00015.safetensors', framework='pt', device='cpu') as f:
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
    atomic_json(args.output / 'identity.json', dict(
        prefix_result_sha256=file_hash(args.prefix / 'result.json'), trace_result_sha256=file_hash(args.trace / 'result.json'),
        order=identity['order'], experts_implementation=config._experts_implementation,
        source_sha256={inspect.getfile(obj): file_hash(inspect.getfile(obj)) for obj in
                       (hf, moe, get_checkpoint_conversion_mapping, dot_natural_key)},
        conversions=conversion_info, original_expert_tensors_verified=config.num_experts * 3,
        config=config.to_dict(), weight_updates=0,
    ))
    reports = []
    paths = {}
    reference_stages = ['input', 'attention', 'router.logits', 'router.weights', 'router.indices', 'mlp', 'output']

    def difference(a, b):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('incompatible or nonfinite comparison tensors')
        return dict(equal=bool(torch.equal(a, b)), max_abs_error=float((a.float() - b.float()).abs().max()),
                    differing_elements=int((a != b).sum()))

    for record in prefix['records']:
        size, offset = record['micro_batch_size'], record['offset']
        begin = time.monotonic()
        atomic_json(args.output / 'progress.json', dict(phase='first_layer', size=size, offset=offset, heartbeat_unix=time.time()))
        path = args.prefix / record['tensor_file']
        if file_hash(path) != record['tensor_sha256']:
            raise ValueError('original prefix tensor changed')
        inp = torch.load(path, map_location='cpu', weights_only=True)
        width = inp['hidden_states'].shape[1]
        values = {}
        routing = {}
        hooks = []

        def save(name, value):
            if name in values:
                raise ValueError('stage executed more than once')
            values[name] = value.detach().clone()

        def after(name):
            def hook(module, a, out):
                save(name, out)
            return hook

        def attention_hook(module, a, out):
            save('attention', out[0])

        def norm_input_hook(module, a):
            save('post_attention_residual', a[0])

        def router_hook(module, a, out):
            for name, value in zip(('logits', 'weights', 'indices'), out, strict=True):
                save('router.' + name, value.reshape(size, width, -1))
            _, perm = torch.sort(out[2].reshape(-1))
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.numel())
            routing.update(perm=perm, inv=inv)

        hooks.append(layer.self_attn.register_forward_hook(attention_hook))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(norm_input_hook))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(after('post_attention_norm')))
        hooks.append(layer.mlp.gate.register_forward_hook(router_hook))
        hooks.append(layer.mlp.register_forward_hook(after('mlp')))
        original_linear = moe._grouped_linear
        linear_calls = []

        def grouped_linear(x, weight, offsets, *a, **kw):
            index = len(linear_calls)
            if index not in (0, 1):
                raise ValueError('unexpected extra expert projection')
            expected_weight = layer.mlp.experts.gate_up_proj if index == 0 else layer.mlp.experts.down_proj
            if weight is not expected_weight:
                raise ValueError('unexpected expert weight')
            if index == 0:
                expected_input = values['post_attention_norm'].reshape(-1, config.hidden_size)[routing['perm'] // config.num_experts_per_tok]
                if not torch.equal(x, expected_input):
                    raise ValueError('grouped sort does not match router capture')
            else:
                save('experts.activation', x[routing['inv']].reshape(size, width, config.num_experts_per_tok, -1))
            out = original_linear(x, weight, offsets, *a, **kw)
            save('experts.gate_up' if index == 0 else 'experts.down',
                 out[routing['inv']].reshape(size, width, config.num_experts_per_tok, -1))
            linear_calls.append(dict(stage='gate_up' if index == 0 else 'down', offsets=offsets.tolist(),
                                     input_shape=list(x.shape), output_shape=list(out.shape)))
            return out

        try:
            moe._grouped_linear = grouped_linear
            with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16), compact_text_attention(hf) as events:
                save('input', inp['hidden_states'])
                out = layer(inp['hidden_states'], attention_mask=inp['attention_mask'], position_ids=inp['position_ids'],
                            position_embeddings=inp['position_embeddings'], use_cache=False)
                save('output', out)
        finally:
            moe._grouped_linear = original_linear
            for hook in hooks:
                hook.remove()
        if len(linear_calls) != 2 or len(events) != 1:
            raise ValueError('expected exactly one attention and two grouped expert projections')
        saved = []
        tensor = args.output / f'micro-{size}-offset-{offset}.pt'
        torch.save(dict(token_mask=inp['token_mask'], stages=values), tensor)
        for j in range(size):
            row = identity['order'][offset + j]
            mask = inp['token_mask'][j].bool()
            old = read(args.trace / f'row-{row:02d}.json')
            if old != trace_reports[row]:
                raise ValueError('reference row changed')
            comparisons = []
            for name in reference_stages:
                ref = args.trace / f'row-{row:02d}-trace/layer-00.{name}.pt'
                if file_hash(ref) != old['reference']['layer-00.' + name]['sha256']:
                    raise ValueError('original reference tensor changed')
                comparisons.append(dict(stage=name, reference_sha256=file_hash(ref),
                                        **difference(values[name][j, mask], torch.load(ref, map_location='cpu', weights_only=True))))
            saved.append(dict(row=row, comparisons=comparisons))
            paths[size, row] = (tensor, j)
        report = dict(size=size, offset=offset, tensor_file=tensor.name, tensor_sha256=file_hash(tensor),
                      attention_records=events, linear_calls=linear_calls, rows=saved, seconds=time.monotonic()-begin)
        reports.append(report)
        atomic_json(args.output / f'micro-{size}-offset-{offset}.json', report)
        del values, inp, out
    differences = []
    for row in identity['order']:
        p1, i1 = paths[1, row]
        p2, i2 = paths[2, row]
        a = torch.load(p1, map_location='cpu', weights_only=True)
        b = torch.load(p2, map_location='cpu', weights_only=True)
        if not torch.equal(a['token_mask'][i1], b['token_mask'][i2]):
            raise ValueError('token masks differ')
        mask = a['token_mask'][i1].bool()
        comparisons = [dict(stage=k, valid=difference(v[i1, mask], b['stages'][k][i2, mask]),
                            full_padded=difference(v[i1], b['stages'][k][i2])) for k, v in a['stages'].items()]
        differences.append(dict(row=row, stages=comparisons,
                                first_valid_difference=next((c['stage'] for c in comparisons if not c['valid']['equal']), None)))
        del a, b
    for name in ('config.json', 'model-00001-of-00015.safetensors'):
        if file_hash(args.model / name) != weights[name]:
            raise ValueError('model asset changed during diagnostic')
    atomic_json(args.output / 'result.json', dict(status='diagnostic_complete', reports=reports, differences=differences,
        elapsed_seconds=time.monotonic()-start, weight_updates=0, source_sha256=file_hash(__file__),
        identity_sha256=file_hash(args.output / 'identity.json'),
        scope='One original HF decoder layer with original default grouped_mm dispatch, real padded actor prefix, original weights and compact attention. Full padded tensors persisted, grouped expert outputs restored to token/top-k order. No later layers, actor probabilities, NPU, optimizer, generation, or training.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('model', 'prefix', 'trace', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output / 'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output / 'terminal.json', dict(exit_code=0))
