#!/usr/bin/env python3
"""Actual Qwen LoRA replay through pinned native compute_log_prob/update_policy."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from peft import PeftModel
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.paired_rollout import compose_rewards
    from multimodal_jev.policy_identity import validate_adapter_export
    from multimodal_jev.rl_run import load_rollout_run
    from multimodal_jev.verl_audio import make_batch, compute_advantages
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device, native_actor_cpu_logprobs

    config = json.loads(args.config.read_text())
    dependency = args.reference_run
    if json.loads((dependency/'terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('successful reference dependency required')
    reference_result = json.loads((dependency/'result.json').read_text())
    if (reference_result['status'] != 'passed'
            or reference_result['comparison_sha256'] != file_hash(dependency/'comparison.pt')
            or reference_result['config_sha256'] != file_hash(args.config)
            or reference_result['actor_source_sha256'] != args.actor_sha256):
        raise ValueError('reference identity changed')
    if file_hash(inspect.getfile(DataParallelPPOActor)) != args.actor_sha256:
        raise ValueError('pinned native actor source changed')
    if config.get('parent_run') or config['device'] != 'cpu' or config['dtype'] != 'float32':
        raise ValueError('fresh CPU FP32 initializer replay required')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    identity, examples, records = load_rollout_run(config['rollout_run'])
    validate_adapter_export(config['model_path'], identity['config']['checkpoint'], config['initial_adapter'])
    if len(records) != 12:
        raise ValueError('registered twelve-response gate required')
    reference = torch.load(dependency/'comparison.pt', map_location='cpu', weights_only=True)
    for key in ('gradients', 'parameters', 'exp_avg', 'exp_avg_sq'):
        if len(reference['reference'][key]) != len(reference['native'][key]):
            raise ValueError('incomplete dependency tensors')
        for left, right in zip(reference['reference'][key], reference['native'][key]):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-4)
    expected = reference['reference']
    del reference
    processor, base = load_runtime(config['model_path'], 'cpu', 'float32')
    model = PeftModel.from_pretrained(base, config['initial_adapter'], is_trainable=True)
    # Native train() must not introduce dropout absent in reference eval().
    from multimodal_jev.model_admission import config_dropout_values
    dropout_values = config_dropout_values(model.config.to_dict())
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            dropout_values['module.'+name] = module.p
    if any(value != 0 for value in dropout_values.values()):
        raise ValueError('nonzero dropout invalidates train/eval comparison')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if len(parameters) != len(expected['parameters']):
        raise ValueError('reference parameter count changed')
    initial = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=0., foreach=False, fused=False)
    lookup = {ex.example_id: ex for ex in examples}
    waves = {ex.example_id: load_audio(ex.audio) for ex in examples}
    ordered = [lookup[r.example_id] for r in records]
    rewards, _ = compose_rewards(ordered, records, config['reward_mode'], config['pair_bonus_weight'])
    batch = compute_advantages(make_batch(processor, ordered, [waves[r.example_id] for r in records],
        [r.token_ids for r in records], [r.log_probs for r in records], rewards,
        identity['config']['sampling']['n']))
    if not batch.batch['advantages'].any():
        raise ValueError('registered gate must exercise nonzero advantages')
    batch.meta_info.update(temperature=1., micro_batch_size=len(records), use_dynamic_bsz=False,
                           pad_token_id=processor.tokenizer.pad_token_id)
    actor_config = OmegaConf.create({
        'use_remove_padding': False, 'use_fused_kernels': False,
        'ulysses_sequence_parallel_size': 1, 'use_dynamic_bsz': False, 'use_prefix_grouper': False,
        'entropy_from_logits_with_chunking': False, 'use_torch_compile': False,
        'fsdp_config': {'dtype': 'float32'}, 'calculate_sum_pi_squared': False,
        'entropy_checkpointing': False, 'ppo_mini_batch_size': len(records),
        'ppo_micro_batch_size_per_gpu': len(records), 'ppo_epochs': 1,
        'use_kl_loss': False, 'calculate_entropy': False, 'entropy_coeff': 0.,
        'loss_agg_mode': 'token-mean', 'use_rollout_log_probs': True,
        'policy_loss': {'loss_mode': 'vanilla'}, 'clip_ratio': .2, 'clip_ratio_low': None,
        'clip_ratio_high': None, 'clip_ratio_c': 3., 'global_batch_info': {}, 'grad_clip': 1.,
    })
    def progress(phase):
        atomic_json(args.output/'progress.json', {'phase': phase, 'heartbeat_unix': time.time()})
    progress('initializing_native_actor')
    dist.init_process_group('gloo', init_method=(args.output/'gloo-init').resolve().as_uri(), rank=0, world_size=1)
    try:
        actor = DataParallelPPOActor(actor_config, model, optimizer)
        captured = {}
        def capture(index, grad):
            if index in captured: raise ValueError('unexpected multiple backwards')
            captured[index] = grad.detach().clone()
            return grad
        with native_actor_cpu_device(actor), native_actor_cpu_logprobs():
            progress('native_compute_log_prob')
            logp = actor.compute_log_prob(batch)['log_probs']
            mask = batch.batch['response_mask'].bool()
            parity_error = float((logp-batch.batch['old_log_probs'])[mask].abs().max())
            if not torch.isfinite(logp).all() or parity_error > config['max_logprob_error']:
                raise ValueError('native wrapper rollout parity failed')
            del logp
            handles = [p.register_hook(lambda grad, i=i: capture(i, grad)) for i, p in enumerate(parameters)]
            try:
                progress('native_update_policy')
                metrics = actor.update_policy(batch)
            finally:
                for handle in handles: handle.remove()
        if len(captured) != len(parameters): raise ValueError('incomplete native gradients')
        actual = {'gradients': [captured[i] for i in range(len(parameters))],
                  'parameters': [p.detach().clone() for p in parameters],
                  'exp_avg': [optimizer.state[p]['exp_avg'].clone() for p in parameters],
                  'exp_avg_sq': [optimizer.state[p]['exp_avg_sq'].clone() for p in parameters]}
        steps = [int(optimizer.state[p]['step']) for p in parameters]
        changed = sum(not torch.equal(before, after) for before, after in zip(initial, actual['parameters']))
        errors, failures = {}, []
        for key in actual:
            errors[key] = max(float((a-b).abs().max()) for a, b in zip(expected[key], actual[key]))
            for i, (left, right) in enumerate(zip(expected[key], actual[key])):
                try: torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-4)
                except AssertionError as exc: failures.append({'field': key, 'parameter': i, 'error': str(exc)})
        comparison = {'max_errors': errors, 'failures': failures, 'optimizer_steps': steps,
                      'changed_parameters': changed, 'parity_error': parity_error, 'metrics': metrics,
                      'atol': 2e-6, 'rtol': 2e-4, 'dropout_values': dropout_values,
                      'actor_config': OmegaConf.to_container(actor_config)}
        atomic_json(args.output/'comparison.json', comparison)
        torch.save({'reference': expected, 'native_update_policy': actual}, args.output/'comparison.pt')
        if failures or set(steps) != {1} or changed == 0:
            raise AssertionError('native update_policy equivalence failed')
        atomic_json(args.output/'result.json', {'status': 'passed', **comparison,
            'comparison_sha256': file_hash(args.output/'comparison.pt'),
            'reference_result_sha256': file_hash(dependency/'result.json'),
            'rollout_sha256': file_hash(Path(config['rollout_run'])/'rollouts.jsonl'),
            'actor_source_sha256': args.actor_sha256,
            'scope': 'real Qwen single full-batch native compute_log_prob/update_policy with explicit CPU device/logprob adapters; '
                     'not multi-microbatch, zero-gradient no-op, NPU, Ray/FSDP, distributed model training or efficacy'})
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--actor-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
        raise
    atomic_json(args.output/'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
