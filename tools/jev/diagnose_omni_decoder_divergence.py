#!/usr/bin/env python3
"""Continue accepted real layer-0 states; stop at first valid decoder difference."""
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
    from multimodal_jev.omni_cpu_expert_projection import fp32_grouped_linear

    torch.set_num_threads(16)
    start = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    for p, pin in [
        (args.first / 'result.json', 'a4cd6a0c2938268fe5c455b3513879f85856aa0be260f0d5282d844e1801567d'),
        (args.prefix / 'result.json', 'fb10bfc904f23e4ebb408ce3a685c9eac66a55f3bfd48b50dfae95443c5e08e7'),
        (args.trace / 'result.json', '23ee087e133334428c0e0fcc85b728a4f7459350954b32ef7a22b1b6a284a161'),
    ]:
        if file_hash(p) != pin:
            raise ValueError('accepted original diagnostic sources required')
    first = read(args.first / 'result.json')
    prior_identity = read(args.first / 'identity.json')
    if file_hash(args.first / 'identity.json') != first['identity_sha256']:
        raise ValueError('first layer identity changed')
    if any(file_hash(Path(p)) != pin for p, pin in prior_identity['source_sha256'].items()):
        raise ValueError('original HF model, conversion and expert implementations required')
    prefix = read(args.prefix / 'result.json')
    identity = read(args.prefix / 'identity.json')
    if file_hash(args.prefix / 'identity.json') != prefix['identity_sha256']:
        raise ValueError('prefix identity changed')
    if file_hash(inspect.getfile(inspect.unwrap(compact_text_attention))) != identity['source_files']['src/multimodal_jev/omni_compact_attention.py']:
        raise ValueError('original compact attention adapter required')
    order = identity['order']
    if order != [2, 0, 5, 1, 3, 4]:
        raise ValueError('complete original six-row pairing required')
    model_pins = {r['file']: r['sha256'] for r in identity['model_files']}
    verified = set()

    def verify_file(name):
        if name not in verified:
            if file_hash(args.model / name) != model_pins[name]:
                raise ValueError('original model file changed: ' + name)
            verified.add(name)

    for name in ('config.json', 'model.safetensors.index.json'):
        verify_file(name)
    index = read(args.model / 'model.safetensors.index.json')['weight_map']
    config = Qwen3OmniMoeTextConfig(**read(args.model / 'config.json')['thinker_config']['text_config'])
    config._attn_implementation = 'sdpa'
    with torch.device('meta'), initialization.no_init_weights():
        skeleton = hf.Qwen3OmniMoeThinkerTextModel(config).to(torch.bfloat16)
    if config._experts_implementation != prior_identity['experts_implementation'] or config.num_hidden_layers != 48:
        raise ValueError('original decoder configuration required')

    def load_layer(layer_index):
        checkpoint_prefix = f'thinker.model.layers.{layer_index}.'
        old_prefix = f'model.layers.{layer_index}.'
        names = [n for n in index if n.startswith(checkpoint_prefix)]
        if not names:
            raise ValueError('missing original layer keys')
        old = {}
        for filename in sorted({index[n] for n in names}):
            verify_file(filename)
            with safe_open(args.model / filename, framework='pt', device='cpu') as f:
                old.update({n.removeprefix('thinker.'): f.get_tensor(n) for n in names if index[n] == filename})
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
                        raise ValueError('unexpected expert conversion target')
                    converter.add_tensor(renamed, name, pattern, old[name])
                    used.add(name)
            if target is None:
                raise ValueError('missing expert conversion')
            converted.update(converter.convert(target))
        for i in range(config.num_experts):
            p = old_prefix + f'mlp.experts.{i}.'
            gate_up = converted[old_prefix + 'mlp.experts.gate_up_proj'][i]
            down = converted[old_prefix + 'mlp.experts.down_proj'][i]
            for a, b in [(gate_up[:config.moe_intermediate_size], old[p+'gate_proj.weight']),
                         (gate_up[config.moe_intermediate_size:], old[p+'up_proj.weight']),
                         (down, old[p+'down_proj.weight'])]:
                if not torch.equal(a, b):
                    raise ValueError('converted expert differs from checkpoint')
        state = {k.removeprefix(old_prefix): v for k, v in old.items() if k not in used}
        state.update({k.removeprefix(old_prefix): v for k, v in converted.items()})
        layer = skeleton.layers[layer_index]
        layer.load_state_dict(state, strict=True, assign=True)
        layer.eval().requires_grad_(False)
        if any(p.is_meta or p.dtype != torch.bfloat16 or p.device.type != 'cpu' for p in layer.parameters()):
            raise ValueError('complete original BF16 CPU layer required')
        return layer

    inputs = {}
    states = {}
    initial_files = []
    expected = [(s, o) for s in (1, 2) for o in range(0, 6, s)]
    if [(r['size'], r['offset']) for r in first['reports']] != expected:
        raise ValueError('complete first layer calls required')
    prefix_map = {(r['micro_batch_size'], r['offset']): r for r in prefix['records']}
    for record in first['reports']:
        key = record['size'], record['offset']
        p = args.first / record['tensor_file']
        if file_hash(p) != record['tensor_sha256']:
            raise ValueError('accepted first layer tensor changed')
        v = torch.load(p, map_location='cpu', weights_only=True)
        pp = args.prefix / prefix_map[key]['tensor_file']
        if file_hash(pp) != prefix_map[key]['tensor_sha256']:
            raise ValueError('actual actor prefix tensor changed')
        inp = torch.load(pp, map_location='cpu', weights_only=True)
        if not torch.equal(v['token_mask'], inp['token_mask']):
            raise ValueError('prefix mask changed')
        inputs[key] = {k: t for k, t in inp.items() if k != 'hidden_states'}
        states[key] = inp['hidden_states'] if args.fp32_projections else v['stages']['output']
        initial_files.append(dict(size=key[0], offset=key[1], sha256=file_hash(p)))
        del v, inp
    refs = read(args.trace / 'result.json')['reports']
    refstages = ('input', 'attention', 'router.logits', 'router.weights', 'router.indices', 'mlp', 'output')

    def difference(a, b):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('incompatible/nonfinite stage tensors')
        return dict(equal=bool(torch.equal(a, b)), max_abs_error=float((a.float()-b.float()).abs().max()),
                    differing_elements=int((a != b).sum()))

    start_layer = 0 if args.fp32_projections else 1
    atomic_json(args.output / 'identity.json', dict(order=order, start_layer=start_layer, maximum_layer=47,
        fp32_projections=args.fp32_projections,
        projection_source_sha256=file_hash(inspect.getfile(fp32_grouped_linear)) if args.fp32_projections else None,
        stop_rule=('Stop after any valid-token batch1/batch2 stage difference; original BF16 reference deviations are recorded separately.' if args.fp32_projections else
                   'Stop after any valid-token batch1/batch2 stage difference or original-reference difference; no output head or actor probabilities.'),
        initial_files=initial_files, first_result_sha256=file_hash(args.first/'result.json'),
        prefix_result_sha256=file_hash(args.prefix/'result.json'), trace_result_sha256=file_hash(args.trace/'result.json'),
        experts_implementation=config._experts_implementation, weight_updates=0))
    summaries = []
    found = False
    for layer_index in range(start_layer, config.num_hidden_layers):
        begin = time.monotonic()
        atomic_json(args.output / 'progress.json', dict(phase='load_layer', layer=layer_index, heartbeat_unix=time.time()))
        layer = load_layer(layer_index)
        captures = {}
        calls = []
        for size, offset in expected:
            atomic_json(args.output / 'progress.json', dict(phase='forward_layer', layer=layer_index, size=size, offset=offset, heartbeat_unix=time.time()))
            key = size, offset
            inp = inputs[key]
            values = dict(input=states[key].clone())
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

            def norm_pre(module, a):
                save('post_attention_residual', a[0])

            def router_hook(module, a, out):
                for name, t in zip(('logits', 'weights', 'indices'), out, strict=True):
                    save('router.'+name, t.reshape(size, states[key].shape[1], -1))

            hooks.append(layer.input_layernorm.register_forward_hook(after('input_norm')))
            hooks.append(layer.self_attn.register_forward_hook(attention_hook))
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(norm_pre))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(after('post_attention_norm')))
            hooks.append(layer.mlp.gate.register_forward_hook(router_hook))
            hooks.append(layer.mlp.register_forward_hook(after('mlp')))
            original_projection = moe._grouped_linear
            try:
                if args.fp32_projections:
                    moe._grouped_linear = fp32_grouped_linear
                with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16), compact_text_attention(hf) as events:
                    out = layer(states[key], attention_mask=inp['attention_mask'], position_ids=inp['position_ids'],
                                position_embeddings=inp['position_embeddings'], use_cache=False)
                    save('output', out)
            finally:
                moe._grouped_linear = original_projection
                for hook in hooks:
                    hook.remove()
            if len(events) != 1 or events[0]['layer'] != layer_index:
                raise ValueError('unexpected attention call coverage')
            captures[key] = values
            calls.append(dict(size=size, offset=offset, attention_records=events))
        between = []
        reference = []
        for i, row in enumerate(order):
            single = captures[1, i]
            pair = captures[2, (i//2)*2]
            j = i % 2
            mask = inputs[1, i]['token_mask'][0].bool()
            comparisons = [dict(stage=k, valid=difference(v[0, mask], pair[k][j, mask]),
                                full_padded=difference(v[0], pair[k][j])) for k, v in single.items()]
            between.append(dict(row=row, comparisons=comparisons,
                                first_valid_difference=next((c['stage'] for c in comparisons if not c['valid']['equal']), None)))
            for stage in refstages:
                event = f'layer-{layer_index:02d}.{stage}'
                rp = args.trace / f'row-{row:02d}-trace/{event}.pt'
                if file_hash(rp) != refs[row]['reference'][event]['sha256']:
                    raise ValueError('original layer reference changed')
                ref = torch.load(rp, map_location='cpu', weights_only=True)
                reference.append(dict(row=row, stage=stage, reference_sha256=file_hash(rp),
                                      micro1=difference(single[stage][0, mask], ref), micro2=difference(pair[stage][j, mask], ref)))
        found = (any(d['first_valid_difference'] is not None for d in between)
                 or (not args.fp32_projections and any(not r[s]['equal'] for r in reference for s in ('micro1', 'micro2'))))
        # Persist every stage, including equal layers. External audits can locate
        # the first difference from tensors rather than trusting summary flags.
        saved = []
        for key, values in captures.items():
            size, offset = key
            p = args.output / f'layer-{layer_index:02d}-micro-{size}-offset-{offset}.pt'
            torch.save(dict(token_mask=inputs[key]['token_mask'], stages=values), p)
            saved.append(dict(size=size, offset=offset, file=p.name, sha256=file_hash(p)))
            states[key] = values['output']
        report = dict(layer=layer_index, between=between, reference=reference, calls=calls, saved=saved,
                      valid_difference_found=found, seconds=time.monotonic()-begin)
        atomic_json(args.output/f'layer-{layer_index:02d}.json', report)
        summaries.append(dict(layer=layer_index, report_sha256=file_hash(args.output/f'layer-{layer_index:02d}.json'),
                              valid_difference_found=found, seconds=report['seconds']))
        skeleton.layers[layer_index] = torch.nn.Identity()
        del layer, captures, values, single, pair, out
        if found:
            break
    for name in sorted(verified):
        if file_hash(args.model/name) != model_pins[name]:
            raise ValueError('model file changed during diagnostic')
    atomic_json(args.output/'result.json', dict(status='diagnostic_complete', valid_difference_found=found,
        fp32_projections=args.fp32_projections,
        layers=summaries, first_differing_layer=layer_index if found else None,
        model_files_verified={name:model_pins[name] for name in sorted(verified)},
        identity_sha256=file_hash(args.output/'identity.json'), source_sha256=file_hash(__file__),
        elapsed_seconds=time.monotonic()-start, weight_updates=0,
        scope=('Uniform FP32 expert projection candidate from actual layer0 prefix; original BF16 reference deviations recorded, stop at first valid inter-batch stage difference. ' if args.fp32_projections else
               'Sequential original decoder continuation from accepted real first-layer outputs; stop at first valid-stage/reference difference. ')
              +'All six original rows and both microbatch sizes. No audio encoding, output head, full actor probability replay, NPU, optimizer or training.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('model', 'first', 'prefix', 'trace', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fp32-projections',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))
