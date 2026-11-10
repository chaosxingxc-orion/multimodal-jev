#!/usr/bin/env python3
"""Real native actor loop with an explicit log-prob perturbation forward fixture."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    import types
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
    from multimodal_jev.rl_run import load_rollout_run
    from multimodal_jev.paired_rollout import compose_rewards
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    from multimodal_jev.verl_native_loss import register_native_token_mean_loss

    torch.set_num_threads(1)
    if file_hash(inspect.getfile(DataParallelPPOActor)) != args.actor_sha256:
        raise ValueError('native actor source changed')
    if file_hash(inspect.getfile(compute_policy_loss_vanilla)) != args.core_sha256:
        raise ValueError('native loss source changed')
    identity, examples, records = load_rollout_run(args.rollout)
    if len(records) != 48: raise ValueError('registered48-response fixture required')
    lookup = {ex.example_id: ex for ex in examples}
    rewards, _ = compose_rewards([lookup[r.example_id] for r in records], records, 'pair-grpo', .2)
    width = max(len(r.token_ids) for r in records)
    mask = torch.zeros(48, width)
    old = torch.zeros_like(mask)
    token_rewards = torch.zeros_like(mask)
    for i, record in enumerate(records):
        length = len(record.token_ids)
        mask[i, :length] = 1
        old[i, :length] = torch.tensor(record.log_probs)
        token_rewards[i, length-1] = rewards[i]
    advantages, _ = compute_grpo_outcome_advantage(token_rewards, mask,
        np.array([r.example_id for r in records], dtype=object), norm_adv_by_std_in_grpo=True)
    index = torch.arange(48*width, dtype=torch.float32).reshape(48, width)
    features = torch.stack([torch.sin(index*.17), torch.cos(index*.31), (index.remainder(7)-3)/7], -1)
    class Perturbation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.012, -.008, .004]))
        def forward(self, selected): return (selected*self.weight).sum(-1)
    def optimizer(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False, fused=False)
    def config(micro, mode):
        return OmegaConf.create({'use_remove_padding': False, 'use_fused_kernels': False,
            'ulysses_sequence_parallel_size': 1, 'use_dynamic_bsz': False, 'use_prefix_grouper': False,
            'entropy_from_logits_with_chunking': False, 'use_torch_compile': False,
            'fsdp_config': {'dtype': 'float32'}, 'calculate_sum_pi_squared': False,
            'ppo_mini_batch_size': 48, 'ppo_micro_batch_size_per_gpu': micro, 'ppo_epochs': 1,
            'use_kl_loss': False, 'calculate_entropy': False, 'entropy_coeff': 0.,
            'loss_agg_mode': 'token-mean', 'use_rollout_log_probs': True,
            'policy_loss': {'loss_mode': mode}, 'clip_ratio': .2, 'clip_ratio_low': None,
            'clip_ratio_high': None, 'clip_ratio_c': 3., 'global_batch_info': {}, 'grad_clip': 1.,
            'jev_batch_num_tokens': int(mask.sum()), 'jev_batch_num_responses': 48})
    def state(model, opt, grad):
        return {'parameters': model.weight.detach().clone(), 'gradients': grad.clone(),
                'exp_avg': opt.state[model.weight]['exp_avg'].clone(),
                'exp_avg_sq': opt.state[model.weight]['exp_avg_sq'].clone()}
    reference = Perturbation()
    opt = optimizer(reference)
    loss, _ = compute_policy_loss_vanilla(old, old+reference(features), advantages, mask, config=config(48, 'vanilla'))
    loss.backward()
    grad = reference.weight.grad.detach().clone()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    expected = state(reference, opt, grad)
    # input_ids encode fixture row indices, explicitly not ASR tokens.
    batch = DataProto.from_dict(tensors={'responses': torch.zeros(48, width, dtype=torch.long),
        'response_mask': mask, 'input_ids': torch.arange(48).reshape(48, 1),
        'attention_mask': torch.ones(48, 1), 'position_ids': torch.zeros(48, 1, dtype=torch.long),
        'old_log_probs': old, 'advantages': advantages}, meta_info={'temperature': 1., 'pad_token_id': 0})
    dist.init_process_group('gloo', init_method=(args.output/'gloo-init').resolve().as_uri(), rank=0, world_size=1)
    try:
        registered = register_native_token_mean_loss()
        assert register_native_token_mean_loss() == registered
        outputs, tensors, failures = {}, {'reference': expected}, []
        for label, micro, mode in [('full', 48, 'vanilla'), ('micro8', 8, registered),
                                  ('micro12', 12, registered), ('wrong_local_mean8', 8, 'vanilla')]:
            model = Perturbation()
            opt = optimizer(model)
            actor = DataParallelPPOActor(config(micro, mode), model, opt)
            captured = []
            handle = model.weight.register_hook(lambda g: captured.append(g.detach().clone()))
            def fixture_forward(self, inputs, temperature, calculate_entropy=False):
                if temperature != 1. or calculate_entropy: raise ValueError('fixture contract changed')
                ids = inputs['input_ids'][:, 0].long()
                return {'log_probs': old[ids]+self.actor_module(features[ids])}
            actor._forward_micro_batch = types.MethodType(fixture_forward, actor)
            with native_actor_cpu_device(actor):
                metrics = actor.update_policy(batch)
            handle.remove()
            assert len(captured) == 48//micro
            actual = state(model, opt, torch.stack(captured).sum(0))
            assert int(opt.state[model.weight]['step']) == 1
            tensors[label] = actual
            errors = {key: float((actual[key]-expected[key]).abs().max()) for key in actual}
            if label != 'wrong_local_mean8':
                for key in actual:
                    try: torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-4)
                    except AssertionError as exc: failures.append({'case': label, 'field': key, 'error': str(exc)})
            elif errors['gradients'] <= 1e-5:
                failures.append({'case': label, 'error': 'negative control failed to distinguish local averaging'})
            outputs[label] = {'max_errors': errors, 'native_metrics': metrics,
                              'microbatch_tokens': [int(mask[i:i+micro].sum()) for i in range(0,48,micro)]}
        torch.save(tensors, args.output/'comparison.pt')
        report = {'cases': outputs, 'failures': failures, 'responses': 48, 'response_tokens': int(mask.sum()),
                  'atol': 2e-6, 'rtol': 2e-4, 'actor_source_sha256': args.actor_sha256,
                  'core_source_sha256': args.core_sha256, 'policy_id': identity['policy_id'],
                  'rollout_sha256': file_hash(args.rollout/'rollouts.jsonl'),
                  'comparison_sha256': file_hash(args.output/'comparison.pt'),
                  'scope': 'native update_policy loop/loss/AdamW with explicit three-parameter forward fixture and saved real rollout rewards; no ASR model, audio-forward, distributed or efficacy claim'}
        atomic_json(args.output/'comparison.json', report)
        if failures: raise AssertionError('native microbatch scaling failed')
        atomic_json(args.output/'result.json', {'status': 'passed', **report})
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--actor-sha256', required=True)
    p.add_argument('--core-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try: run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
        raise
    atomic_json(args.output/'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})


if __name__ == '__main__': main()
