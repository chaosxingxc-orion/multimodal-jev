"""Finite TRAIN parity diagnosis using the audited input cache and TP1 owner."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import require_training_storage
from tools.reports.context_train_macro_parity import panel


def read(path):
    return json.loads(Path(path).read_text())


def layer_trace(layers, indices, backend):
    from multimodal_jev.span_trace import SpanTrace

    trace = SpanTrace()
    for i in indices:
        layer = layers[i]
        attention = layer.self_attn
        label = f'decoder.{i}.'
        trace.watch(layer.input_layernorm, label + 'input_norm')
        trace.watch(layer.post_attention_layernorm, label + 'post_norm')
        if backend == 'hf':
            trace.watch_input(layer.post_attention_layernorm, label + 'post_norm_input')
        else:
            def residual(module, args, result, *, prefix=label):
                trace.capture(prefix + 'post_norm_input', result[1])
            trace.handles.append(layer.post_attention_layernorm.register_forward_hook(residual))
        trace.watch_input(attention.o_proj, label + 'attention_kernel')
        trace.watch(attention.o_proj, label + 'attention_output')
        trace.watch(layer.mlp, label + 'mlp')
        for kind in ('q', 'k'):
            def norm(module, args, result, *, name=label + kind + '_norm'):
                if result.ndim == 4 and result.shape[0] == 1:
                    result = result[0]
                trace.capture(name, result.reshape(result.shape[0], -1))
            trace.handles.append(getattr(attention, kind + '_norm').register_forward_hook(norm))
        if backend == 'hf':
            for kind in ('q', 'k', 'v'):
                trace.watch(getattr(attention, kind + '_proj'), label + kind + '_projection')
        else:
            def capture_qkv(module, args, result, *, attn=attention, prefix=label):
                qkv = result[0] if isinstance(result, tuple) else result
                for key, part in zip(('q', 'k', 'v'), qkv.split(
                        [attn.q_size, attn.kv_size, attn.kv_size], dim=-1), strict=True):
                    trace.capture(prefix + key + '_projection', part)
            trace.handles.append(attention.qkv_proj.register_forward_hook(capture_qkv))
            active = [False]
            def begin(module, args, *, flag=active):
                flag[0] = True
            def end(module, args, result, *, flag=active):
                flag[0] = False
            def rotary(module, args, result, *, flag=active, prefix=label):
                if flag[0]:
                    trace.capture(prefix + 'q_rotary', result[0])
                    trace.capture(prefix + 'k_rotary', result[1])
            trace.handles.extend((attention.register_forward_pre_hook(begin),
                                  attention.register_forward_hook(end, always_call=True),
                                  attention.rotary_emb.register_forward_hook(rotary)))
    if backend == 'hf':
        from transformers.models.qwen3 import modeling_qwen3

        original = modeling_qwen3.eager_attention_forward
        names = {id(layers[i].self_attn): f'decoder.{i}.' for i in indices}

        def eager(module, q, k, v, *args, **kwargs):
            name = names.get(id(module))
            if name is not None:
                for key, value in (('q', q), ('k', k)):
                    trace.capture(name + key + '_rotary', value[0].transpose(0, 1).flatten(1))
            return original(module, q, k, v, *args, **kwargs)

        modeling_qwen3.eager_attention_forward = eager

        class Restore:
            def remove(self):
                modeling_qwen3.eager_attention_forward = original

        trace.handles.append(Restore())
    return trace


def worker_trace_start(worker, indices):
    model = worker.get_model()
    if getattr(model, '_jev_endpoint_layer_trace', None) is not None:
        raise ValueError('an endpoint trace is already attached')
    model._jev_endpoint_layer_trace = layer_trace(model.language_model.model.layers, indices, 'vllm')
    return True


def worker_trace_end(worker, destination):
    model = worker.get_model()
    trace = model._jev_endpoint_layer_trace
    trace.close()
    with Path(destination).open('xb') as handle:
        torch.save(trace.tensors, handle)
    model._jev_endpoint_layer_trace = None
    return True


@contextmanager
def fresh_audio_pass(owner, pass_id):
    engine = owner.engine

    class FreshAudioView:
        def encode(self, prompts, **kwargs):
            fresh = [dict(prompt, multi_modal_uuids={
                'audio': [f'diagnosis-{pass_id}:' + prompt['multi_modal_uuids']['audio'][0]],
            }) for prompt in prompts]
            return engine.encode(fresh, **kwargs)

    owner.engine = FreshAudioView()
    try:
        yield
    finally:
        owner.engine = engine


def hf_rows(config, bundle, rows, trace_rows, trace_layers, output):
    import torch_npu  # noqa: F401
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_model_profile import build_head, from_hf
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.context_runtime_profile import PROFILE, install_actor
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_trace import attach_hf_trace
    from tools.jev.context_configurable_device_scope import validate_tp1

    validate_tp1(config)
    torch.set_num_threads(2)
    torch.manual_seed(317)
    torch.npu.manual_seed_all(317)
    device = torch.device('npu', 0)
    profile = from_hf(config['model'])
    contract = bundle['contract']
    if (set(contract['lora']) != set(profile.lora_targets())
            or any(contract['branch'].get(k) != v for k, v in profile.decision_head().items())):
        raise ValueError('checkpoint differs from model profile')
    [(rank, alpha)] = {(s['rank'], s['alpha']) for s in contract['lora'].values()}
    _, base = load_runtime(config['model'], device, 'bfloat16')
    adapted = initialize_adapter(base, {
        'lora_rank': rank, 'lora_alpha': alpha, 'lora_execution_dtype': 'bfloat16',
    }).eval()
    head = build_head(profile).to(device=device, dtype=torch.bfloat16).eval()
    model = ContextDecisionModel(adapted.get_base_model().model, head, contract['token']['id']).eval()
    load_policy(config['source_policy'], model, expected_policy_id=bundle['policy_id'],
                expected_contract=contract, expected_base_id=config['base_id'])
    handle, runtime = install_actor(adapted, model, PROFILE)
    commit_json(output / 'runtime.json', runtime)
    records = {}
    try:
        for index, row in rows.items():
            trace = None
            if index in trace_rows:
                trace = (layer_trace(base.model.language_model.layers, trace_layers, 'hf')
                         if trace_layers else attach_hf_trace(base))
            try:
                with torch.no_grad():
                    logits = model({k: v.to(device) for k, v in row['inputs'].items()},
                                   row['positions'].to(device), return_logits=True).float()[0]
                    records[index] = {'logits': logits.cpu().tolist(), 'prob': logits.sigmoid().cpu().tolist()}
                if trace is not None:
                    torch.save(trace.tensors, output / f'trace-{index}.pt')
            finally:
                if trace is not None:
                    trace.close()
    finally:
        handle.remove()
    return records


def service_rows(config, binding, rows, groups, trace_rows, trace_layers, output, intervention=None):
    from tools.reports.context_replica_batched_service import score_rows
    from tools.reports.context_replica_runtime import service_namespace

    owner = service_namespace(binding)['owner_class'](binding['source_root'])(binding['policy'])
    rpc = owner.engine.llm_engine.collective_rpc
    records = {}
    try:
        runtime = owner.segmented_receipt
        if intervention is not None:
            runtime = dict(runtime, intervention=rpc('install_jev_endpoint_actor_order', kwargs=intervention))
        commit_json(output / 'runtime.json', runtime)
        for pass_id, group in enumerate(groups):
            trace = len(group) == 1 and group[0] in trace_rows
            with fresh_audio_pass(owner, pass_id):
                if trace:
                    if trace_layers:
                        rpc('begin_jev_endpoint_layer_trace', args=(trace_layers,))
                    else:
                        rpc('begin_jev_span_trace')
                values, receipt = score_rows(owner, [rows[i] for i in group],
                    policy_id=owner.bundle['policy_id'], audio_token_id=config['audio_token_id'])
            commit_json(output / f'batch-{pass_id}.json', receipt)
            records.update({i: {'logits': row['logits'], 'prob': row['prob']}
                            for i, row in zip(group, values, strict=True)})
            if trace:
                trace_path = str(output / f'trace-{group[0]}.pt')
                method = 'finish_jev_endpoint_layer_trace' if trace_layers else 'finish_jev_span_trace'
                rpc(method, args=(trace_path,))
        return records
    finally:
        owner.close()


def run(spec, arm, output):
    require_training_storage(output)
    output.mkdir(exist_ok=False)
    config = read(spec['hf_config'])
    for name, sha256 in spec['pins'].items():
        if file_hash(name) != sha256:
            raise ValueError(f'changed diagnosis input: {name}')
    bundle = read_execution_bundle(config['source_policy'], expected_policy_id=config['source_policy_id'],
                                  expected_contract=config['source_contract'], expected_base_id=config['base_id'])
    verify_native_adapter(config['initial_adapter'], bundle)
    selected, encode = panel(config, bundle)
    groups = arm['groups']
    indices = [i for group in groups for i in group]
    if len(set(indices)) != len(indices) or any(not 0 <= i < len(selected['assets']) for i in indices):
        raise ValueError('require unique indices from the frozen TRAIN panel')
    rows = {}
    for index in indices:
        row, receipt = encode(selected['assets'][index])
        if receipt != selected['input_receipts'][index]:
            raise ValueError('diagnostic cache differs from frozen input')
        rows[index] = row
    config.update(physical_devices=[arm['device']], process_device_scope=arm['scope'])
    if arm['mode'] == 'hf':
        records = hf_rows(config, bundle, rows, arm['trace'], arm.get('trace_layers'), output)
    else:
        binding = read(arm['service'])
        if 'service_runtime' in spec:
            binding.update(spec['service_runtime'])
        binding['policy'].update(physical_devices=[arm['device']], process_device_scope=arm['scope'])
        if arm['mode'] == 'serial-acoustic':
            del binding['attention_execution']
        elif arm['mode'] != 'bounded':
            raise ValueError('unknown diagnostic arm')
        records = service_rows(config, binding, rows, groups, arm['trace'], arm.get('trace_layers'), output,
                               arm.get('intervention'))
    commit_json(output / 'predictions.json', dict(policy_id=bundle['policy_id'], records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    run(spec, spec['arms'][args.arm], args.output)
