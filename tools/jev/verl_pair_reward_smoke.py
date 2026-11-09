#!/usr/bin/env python3
"""Real verl numerical checks for pair rewards, using explicit response fixtures."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen3ASRProcessor
from multimodal_jev.audio_policy import load_audio, answer_ids
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.paired_rollout import compose_rewards
from multimodal_jev.task_schema import JevExample
from multimodal_jev.verl_audio import make_batch, compute_advantages, policy_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    rows = []
    with args.manifest.open() as handle:
        first = json.loads(next(handle))
        group = first['metadata']['source_group']
        rows.append(JevExample.from_record(first))
        for line in handle:
            row = json.loads(line)
            if row['metadata']['source_group'] == group:
                rows.append(JevExample.from_record(row))
    if any(ex.split != 'train' for ex in rows):
        raise ValueError('TRAIN fixtures required')
    processor = Qwen3ASRProcessor.from_pretrained(args.model, local_files_only=True)
    if file_hash(rows[0].audio) != rows[0].metadata['audio_sha256']:
        raise ValueError('audio changed')
    audio = load_audio(rows[0].audio)
    examples = rows + rows
    records, ids = [], []
    for replicate in (0, 1):
        for ex in rows:
            value = ex.target
            if replicate == 1 and ex.metadata['variant'] == 'rebind':
                value = next(c.id for c in ex.schema.candidates if c.id != ex.target)
            records.append(SimpleNamespace(example_id=ex.example_id, replicate=replicate,
                                           text=json.dumps({'value': value})))
            ids.append(answer_ids(processor.tokenizer, value))
    old = [[-2.0] * len(tokens) for tokens in ids]
    rewards, details = compose_rewards(examples, records, 'pair-grpo')
    batch = compute_advantages(make_batch(processor, examples, [audio] * len(examples),
                                         ids, old, rewards, 2))
    by_variant = {ex.metadata['variant']: i for i, ex in enumerate(rows)}
    native = by_variant['native']
    permutation = by_variant['permutation']
    adv = batch.batch['advantages']
    assert details['base_rewards'][native] == details['base_rewards'][native + len(rows)]
    assert adv[native, 0] > 0 and adv[native + len(rows), 0] < 0
    assert not adv[permutation].any() and not adv[permutation + len(rows)].any()
    logp = batch.batch['old_log_probs'].clone().requires_grad_(True)
    loss, _ = policy_loss(batch, logp)
    loss.backward()
    assert logp.grad[native, 0] < 0 and logp.grad[native + len(rows), 0] > 0
    assert not logp.grad[batch.batch['response_mask'] == 0].any()
    # Every malformed sample receives -1: no null credit or artificial variance.
    for record in records:
        record.text = 'not JSON'
    bad, bad_details = compose_rewards(examples, records, 'pair-grpo')
    assert bad == [-1.0] * len(examples) and not any(bad_details['pair_bonuses'])
    batch.batch['token_level_rewards'].zero_()
    for i in range(len(examples)):
        batch.batch['token_level_rewards'][i, int(batch.batch['response_mask'][i].sum()) - 1] = -1
    assert not compute_advantages(batch).batch['advantages'].any()
    report = {'status': 'passed', 'source_group': group, 'prompts': len(rows), 'replicates': 2,
              'manifest_sha256': file_hash(args.manifest),
              'same_base_different_pair_bonus_changes_native_advantage': True,
              'equal_pair_rewards_have_zero_advantage': True,
              'malformed_zero_bonus_zero_advantage': True, 'padding_gradient_zero': True,
              'source_sha256': {name: file_hash(Path(__file__).parents[2] / 'src/multimodal_jev' / name)
                                for name in ('paired_rollout.py', 'pair_reward.py', 'verl_audio.py', 'reward.py', 'answers.py')},
              'scope': 'Real verl DataProto/GRPO/PPO and audio preprocessing; controlled fixtures, no policy rollout or efficacy claim'}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
