"""Check a configured Jev provider using real audio and an updated BF16 policy."""
import argparse
import gc
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.audio_policy import load_audio
from multimodal_jev.context_choice import distribution
from multimodal_jev.context_decision_contract import (
    ApplicabilityQuestion, ContextDecisionRequest, PREFIX_LAYOUT,
)
from multimodal_jev.context_policy_snapshot import export_policy
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.model_provider import resolve_provider


def compare(left, right, *, choice=False):
    if left.shape != right.shape or not (left.isfinite().all() and right.isfinite().all()):
        raise ValueError('finite aligned probe logits required')
    p, q = left.sigmoid(), right.sigmoid()
    decisions = (distribution(left, 'open', 1.).exp(), distribution(right, 'open', 1.).exp()) if choice else (p, q)
    return dict(max_logit_error=(left-right).abs().max().item(),
                max_probability_error=(p-q).abs().max().item(),
                max_decision_probability_error=(decisions[0]-decisions[1]).abs().max().item(),
                top1_flips=(decisions[0].argmax(-1) != decisions[1].argmax(-1)).sum().item(),
                threshold_flips=((p >= .5) != (q >= .5)).sum().item())


def run(config):
    provider = resolve_provider(config)
    out = Path(config['output'])
    check_paths({'provider probe': out})
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config['cpu_workers'])
    torch.use_deterministic_algorithms(True)
    if config['probe_device'] == 'npu:0':
        import torch_npu  # noqa: F401
    device = torch.device(config['probe_device'])
    mounted = {int(p.name.removeprefix('davinci')) for p in Path('/dev').glob('davinci[0-9]*')}
    if device.type == 'npu':
        if mounted != set(config['physical_devices']) or len(mounted) != 1 or device.index != 0:
            raise ValueError('one isolated probe NPU required')
        torch.npu.config.allow_internal_format = False
        torch.npu.set_device(device)
    elif device.type != 'cpu' or mounted:
        raise ValueError('CPU probe must have no mounted NPU')
    if 'initial_snapshot' in config:
        from multimodal_jev.policy_identity import checkpoint_identity
        snapshot = Path(config['initial_snapshot']['path'])
        if (file_hash(snapshot/'MANIFEST.json') != config['initial_snapshot']['sha256']
                or checkpoint_identity(config['model'], io_workers=config['io_workers'])[0] != config['base_id']):
            raise ValueError('probe snapshot or base changed')
        manifest = json.loads((snapshot/'MANIFEST.json').read_bytes())
        initial = dict(base_id=config['base_id'], policy_id=manifest['policy_id'])
    else:
        initial = provider.initialize_policy(config, out)
        snapshot = out/'policy'
        manifest = json.loads((snapshot/'MANIFEST.json').read_bytes())
    contract = manifest['contract']
    app = dict(config, snapshot=str(snapshot), policy_id=initial['policy_id'], contract=contract)
    model = provider.load_snapshot(app, device)
    commit_json(out/'loaded.json', dict(policy_id=app['policy_id'], device=str(device)))
    processor = provider.processor(snapshot/'processor')
    collation = provider.collation(model.backbone, processor, model)
    compiler = provider.choice_compiler(processor, collation, config['max_tokens'])
    native, choice = [], []
    questions = tuple(ApplicabilityQuestion(**q) for q in config['questions'])
    for source in config['audio']:
        path = source['path']
        if file_hash(path) != source['sha256']:
            raise ValueError('probe audio changed')
        waveform = load_audio(path)
        request = ContextDecisionRequest(path, questions, tuple(config['requirements']),
                                         version=contract['request_version'])
        inputs, positions = provider.encode(processor, request, waveform,
            decision_token_id=collation['decision_token_id'], audio_token_id=collation['audio_token_id'],
            max_tokens=config['max_tokens'], prompt_layout=PREFIX_LAYOUT)
        item = dict(request_id=source['sha256'], policy_id=app['policy_id'], inputs=inputs, positions=positions)
        native.append(item)
        choice.append((item, request))
    choices = {}
    for width in config.get('choice_widths', [len(config['choice_indices'])]):
        slate = config['choice_indices'][:width]
        if type(width) is not int or not 0 < width == len(slate) or width in choices:
            raise ValueError('distinct Choice probe widths within the configured slate required')
        variants = []
        for item, request in choice:
            variant = dict(item)
            compiler.render(variant, request, 'open', slate)
            if variant['positions'].numel() != width+1:
                raise ValueError('Choice marker count must include NONE')
            variants.append(variant)
        choices[width] = variants
    if not choices:
        raise ValueError('a Choice probe is required')
    panels = {'noul': native, **{f'choice_none_{w}': items for w, items in choices.items()}}
    choice = choices[max(choices)]
    if len({item['inputs']['input_features'].shape[-1] for item in native}) < 2:
        raise ValueError('probe must include a variable-length audio batch')

    def logits(items, policy_id):
        return provider.forward(model, items, device=device, dtype=torch.bfloat16,
                                policy_id=policy_id, **collation).float()

    def score(items, policy_id):
        with torch.no_grad():
            return logits(items, policy_id).cpu()

    initial_metrics = {}
    for name, items in panels.items():
        batch = score(items, app['policy_id'])
        singleton = torch.cat([score([item], app['policy_id']) for item in items])
        initial_metrics[name] = compare(batch, singleton, choice=name != 'noul')
    commit_json(out/'initial-parity.json', initial_metrics)
    params = {key: p for key, p in model.named_parameters() if p.requires_grad}
    model.train()
    optimizer = torch.optim.SGD(params.values(), lr=config['probe_lr'])
    logits(choice, app['policy_id']).square().mean().backward()
    gradient = {'lora': False, 'head': False}
    for name, param in params.items():
        grad = param.grad
        if (param.dtype != torch.bfloat16 or grad is None or not grad.isfinite().all()
                or not (name.startswith('decision_head.') or '.lora_' in name)):
            raise ValueError('invalid BF16 trainable parameter: ' + name)
        group = 'head' if name.startswith('decision_head.') else 'lora'
        gradient[group] |= bool(grad.count_nonzero())
    if not all(gradient.values()):
        raise ValueError('missing LoRA/head gradient signal')
    optimizer.step()
    updated_train = score(choice, app['policy_id'])
    model.eval()
    expected = {name: score(items, app['policy_id']) for name, items in panels.items()}
    updated_eval = expected[f'choice_none_{max(choices)}']
    updated = out/'updated'
    new_id = export_policy(updated, model, snapshot/'processor', contract)
    del optimizer, params, param, grad, model
    gc.collect()
    if device.type == 'npu':
        torch.npu.empty_cache()
    updated_app = dict(app, snapshot=str(updated), policy_id=new_id)
    model = provider.load_snapshot(updated_app, device)
    reloaded = {name: score([dict(item, policy_id=new_id) for item in items], new_id)
                for name, items in panels.items()}
    comparisons = dict(
        singleton_vs_batch=initial_metrics,
        updated_train_vs_eval=compare(updated_train, updated_eval, choice=True),
        **{f'updated_reload_{name}': compare(expected[name], reloaded[name], choice=name != 'noul')
           for name in panels},
    )
    metrics = list(initial_metrics.values()) + [v for k, v in comparisons.items() if k != 'singleton_vs_batch']
    def passes(measures):
        return all(max(m['max_probability_error'], m['max_decision_probability_error']) <= config['max_probability_error']
                   and not (m['top1_flips'] or m['threshold_flips']) for m in measures)

    result = dict(schema='jev-model-provider-probe-v1', engine='HF eager', dtype='bfloat16',
                  device=str(device), base_id=initial['base_id'], geometry=contract['branch'],
                  policy_id=initial['policy_id'], updated_policy_id=new_id,
                  audio_sha256=[row['sha256'] for row in config['audio']], gradient=gradient,
                  comparisons=comparisons, max_probability_error=config['max_probability_error'],
                  passed=passes(metrics))
    if 'vllm' in config:
        commit_json(out/'hf-result.json', result)
        if device.type != 'npu':
            raise ValueError('vLLM provider probe requires an isolated NPU')
        del model
        gc.collect()
        torch.npu.empty_cache()
        from multimodal_jev.context_policy_artifacts import read_execution_bundle
        bundle = read_execution_bundle(updated, expected_policy_id=new_id, expected_contract=contract,
                                       expected_base_id=config['base_id'])
        vllm_config = dict(updated_app, adapter=str(out/'vllm-adapter'), tensor_parallel_size=1,
                           max_model_len=config['vllm']['context_tokens'],
                           max_decisions=max(item['positions'].numel() for items in panels.values() for item in items))
        engine = provider.load_vllm(vllm_config, bundle)
        try:
            comparison = {}
            for name, items in panels.items():
                outputs = [engine.score(item['inputs'], item['positions'], audio_token_id=collation['audio_token_id'])
                           for item in items]
                if any(record['policy_id'] != new_id for record in outputs):
                    raise ValueError('serving probe selected a different policy')
                served = torch.tensor([record['logits'] for record in outputs], dtype=torch.float32)
                comparison[name] = compare(reloaded[name], served, choice=name != 'noul')
            result['vllm'] = dict(comparisons=comparison, runtime=engine.runtime, passed=passes(comparison.values()))
            result['passed'] &= result['vllm']['passed']
        finally:
            engine.close()
    commit_json(out/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    arguments = parser.parse_args()
    result = run(load_config(arguments.config, arguments.local)[0])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
