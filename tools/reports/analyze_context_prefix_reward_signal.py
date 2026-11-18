"""Exact reward-signal diagnostics over previously audited real audio rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jev'))
from context_reward_signal import singleton_signal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, proof_path, release):
    proof = json.loads(proof_path.read_text())
    if (not proof['passed'] or proof['state']['ExitCode'] != 0 or proof['state']['OOMKilled']
            or proof['start_update'] != 0 or proof['stop_update'] != 2):
        raise ValueError('Successful independently audited two-update prefix required')
    sources = {str(proof_path): sha(proof_path)}
    def bound(name):
        p = root / name
        if sha(p) != proof['files'][name]:
            raise ValueError('Archived artifact identity changed: ' + name)
        sources[str(p)] = sha(p)
        return json.loads(p.read_text())
    learning_path = release / 'learning.jsonl'
    tasks_path = release / 'tasks.json'
    for p in (learning_path, tasks_path):
        sources[str(p)] = sha(p)
    learning = [json.loads(x) for x in learning_path.read_text().splitlines()]
    labels = {r['id']: r for r in learning}
    if len(labels) != len(learning) or len(labels) != 40513:
        raise ValueError('Complete unique learning population required')
    candidates = sorted(json.loads(tasks_path.read_text())['intent']['candidate_pairs'])
    details = []
    for update in range(2):
        batch = bound(f'run/batch-{update:04d}.json')
        rollout = bound(f'run/rollout-{update:04d}.json')
        records = batch['records']
        queries = rollout['queries']
        if len(records) != 128 or len(queries) != 128:
            raise ValueError('Full global batch required')
        if rollout['fixed_reference_policy_id'] != proof['source_policy_id']:
            raise ValueError('Fixed Stage1 reference mismatch')
        for b, q in zip(records, queries):
            target = b['target']
            if b['request_id'] != q['request_id'] or b['weight'] != 1:
                raise ValueError('Request pairing or real query weight mismatch')
            native = labels[b['id']]
            gold = candidates.index(native['intent'])
            if (native['role'] != 'learning' or native['official_split'] != 'train'
                    or target['semantics'] != 'exclusive_event'
                    or target['candidate_ids'] != candidates
                    or target['supervised'] != [True] * 60
                    or target['labels'] != [i == gold for i in range(60)]):
                raise ValueError('Complete source truth or candidate binding mismatch')
            br, rr = q['behavior_record'], q['reference_record']
            if (br['policy_id'] != rollout['behavior_policy_id']
                    or br['policy_id'] != batch['policy_id']
                    or rr['policy_id'] != proof['source_policy_id']):
                raise ValueError('Policy binding mismatch')
            for key in ('inputs', 'positions'):
                if br['request_identity'][key] != rr['request_identity'][key]:
                    raise ValueError('Reference/behavior did not receive identical input')
            if len(br['prob']) != 60 or len(rr['prob']) != 60 or q['rollout']['group_size'] != 16:
                raise ValueError('Full candidate distribution and G16 required')
            raw = struct.pack('<60f', *br['prob'])
            if hashlib.sha256(raw).hexdigest() != q['rollout']['probability']['sha256']:
                raise ValueError('Behavior FP32 probability tensor identity mismatch')
            if update == 0 and br['prob'] != rr['prob']:
                raise ValueError('Initial policy/reference differ')
            details.append(dict(update=update, id=b['id'], source_group=native['source_group'],
                                behavior=singleton_signal(br['prob'], gold),
                                reference=singleton_signal(rr['prob'], gold)))
    summaries = []
    keys = ['expected_set_f1', 'reward_variance', 'group_all_equal_reward_probability',
            'group_positive_reward_probability', 'exact_correct_set_probability',
            'empty_set_probability', 'multiple_selection_probability',
            'expected_selected_count', 'gold_probability']
    for update in range(2):
        rows = [r for r in details if r['update'] == update]
        summary = dict(update=update, recordings=len(rows),
                       source_groups=len({r['source_group'] for r in rows}))
        for side in ('behavior', 'reference'):
            summary[side] = {k: sum(r[side][k] for r in rows) / len(rows) for k in keys}
            summary[side]['expected_equal_reward_groups_of_128'] = sum(
                r[side]['group_all_equal_reward_probability'] for r in rows)
            summary[side]['queries_with_equal_reward_probability_above_0_95'] = sum(
                r[side]['group_all_equal_reward_probability'] > .95 for r in rows)
        summary['paired_expected_f1_delta_behavior_minus_reference'] = sum(
            r['behavior']['expected_set_f1'] - r['reference']['expected_set_f1'] for r in rows) / len(rows)
        summaries.append(summary)
    return dict(schema='context-prefix-exact-reward-signal-v1', sources=sources,
                source_policy_id=proof['source_policy_id'], summaries=summaries, queries=details,
                scope='Archived engineering prefix, TRAIN only, not formal curve or heldout efficacy',
                limitations=['Exact equal-reward probability is not the numerical advantage cutoff.',
                             'Sampled actions are not replayed; values integrate over all possible actions.',
                             'Successive updates contain different queries; compare reference within each batch.',
                             'No actor gradients, network variance, clipping effects or calibration improvement inferred.',
                             'No training configuration or dispatch change.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.proof, args.release)
    report['sources'][str(Path(__file__))] = sha(Path(__file__))
    helper = Path(__file__).resolve().parents[1] / 'jev/context_reward_signal.py'
    report['sources'][str(helper)] = sha(helper)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report['summaries'], indent=2))
