"""Read-only replay of archived Bernoulli actions; never a training entrypoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from tools.reports.analyze_context_live_reward_support import run as authenticate


def identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def replay(query, gold):
    receipt = query['rollout']
    p = torch.tensor(query['behavior_record']['prob'], dtype=torch.float32)
    if (p.shape != (60,) or not torch.isfinite(p).all() or ((p <= 0) | (p >= 1)).any()
            or type(gold) is not int or not 0 <= gold < 60
            or receipt['group_size'] != 16 or type(receipt['seed']) is not int
            or receipt['seed'] < 0):
        raise ValueError('Complete interior Final60/G16 contract required')
    if identity(p) != receipt['probability']:
        raise ValueError('Probability identity mismatch')
    generator = torch.Generator(device='cpu').manual_seed(receipt['seed'])
    actions = (torch.rand((1, 16, 60), generator=generator) < p).long()
    if identity(actions) != receipt['actions']:
        raise ValueError('Actual archived action identity mismatch')
    a = actions[0]
    # Complete singleton gold: set F1 is 2*gold_present/(predicted_count+1).
    rewards = 2 * a[:, gold].float() / (a.sum(-1) + 1)
    centered = rewards - rewards.mean()
    grpo = centered / (rewards.std(correction=1) + 1e-6)
    rloo = centered * (16 / 15)
    constant = bool((rewards == rewards[0]).all())
    kind = ('all_zero' if rewards[0] == 0 else 'all_one' if rewards[0] == 1
            else 'constant_partial') if constant else 'varying'
    # Independent output-logit score direction at ratio=1; not a parameter gradient.
    directions = {}
    for name, advantage in [('grpo_formula', grpo), ('rloo_formula', rloo)]:
        direction = (advantage[:, None].double() * (a.double() - p.double())).mean(0)
        directions[name] = dict(l2=float(direction.norm()), gold_coordinate=float(direction[gold]))
    return dict(request_id=query['request_id'], gold_index=gold, action_identity=identity(actions),
                reward_kind=kind, rewards=rewards.tolist(), reward_mean=float(rewards.mean()),
                reward_sample_std=float(rewards.std(correction=1)),
                unique_actions=int(torch.unique(a, dim=0).shape[0]),
                sampled_exact_sets=int((rewards == 1).sum()),
                sampled_positive_rewards=int((rewards > 0).sum()),
                score_directions=directions)


def run(observation, release, schedule, receipts, cache_proof):
    # Recompute TRAIN/audio/tensor/slot/label binding, not just trust an earlier report.
    bound = authenticate(observation, release, schedule, receipts, cache_proof)
    live = json.loads(Path(observation).read_text())
    queries = live['progress']['latest'][0]['data']['rollout']['queries']
    rows = []
    for query, truth in zip(queries, bound['queries'], strict=True):
        if query['request_id'].split('::epoch')[0] != truth['id']:
            raise ValueError('Gold/request order mismatch')
        row = replay(query, truth['gold_index'])
        row.update(source_group=truth['source_group'],
                   threshold_exact_set=truth['behavior']['decision']['exact_set'],
                   theoretical_constant_reward_probability=truth['behavior']['signal']['group_all_equal_reward_probability'])
        rows.append(row)
    counts = Counter(r['reward_kind'] for r in rows)
    strata = {}
    for correct in (True, False):
        subset = [r for r in rows if r['threshold_exact_set'] == correct]
        strata[str(correct)] = dict(queries=len(subset), reward_groups=dict(Counter(r['reward_kind'] for r in subset)))
    sources = dict(bound['sources'])
    for p in [Path(__file__), Path('src/multimodal_jev/context_rollout_batch.py'),
              Path('src/multimodal_jev/span_group_advantage.py')]:
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(schema='context-actual-reward-group-replay-v1', sources=sources,
                torch_version=torch.__version__, actor_update=bound['actor_update'],
                behavior_update=bound['behavior_update'], queries=rows,
                summary=dict(queries=len(rows), verified_action_hashes=len(rows),
                             reward_groups=dict(counts),
                             varying_group_fraction=counts['varying']/len(rows),
                             sampled_actions=len(rows)*16,
                             sampled_exact_sets=sum(r['sampled_exact_sets'] for r in rows),
                             sampled_positive_rewards=sum(r['sampled_positive_rewards'] for r in rows),
                             mean_sampled_reward=sum(r['reward_mean'] for r in rows)/len(rows),
                             identical_action_groups=sum(r['unique_actions'] == 1 for r in rows),
                             threshold_correctness_strata=strata),
                limits=['Single TRAIN batch, not heldout efficacy or a learning trend.',
                        'All four rank copies authenticated and counted once.',
                        'Actions match archived hashes exactly; no new sampling experiment or model execution.',
                        'GRPO/RLOO directions use independent local formulas at ratio=1, not native verl or parameter-gradient replay.',
                        'A zero group advantage removes only the sampled policy term; auxiliary and KL gradients may remain.',
                        'No modification of running configurations or dispatch criteria.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('observation', 'release', 'schedule', 'receipts', 'cache-proof', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = run(args.observation, args.release, args.schedule, args.receipts, args.cache_proof)
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['summary'], indent=2))
