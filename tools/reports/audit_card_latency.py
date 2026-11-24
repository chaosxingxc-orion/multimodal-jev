"""Verify a balanced latency replay against the archived aligned probability vectors."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json


def percentile(values, p):
    """Nearest rank, with equal weight per timed block traversal."""
    return sorted(values)[math.ceil(len(values) * p) - 1]


def validate_trial(trial, pairs):
    if set(trial) != {'standard', 'candidate'}:
        raise ValueError('paired trial missing one arm')
    for times in trial.values():
        first, decision, total = (times[k] for k in ('first_score_seconds', 'first_decision_seconds', 'scoring_seconds'))
        if not (0 < first <= decision <= total and math.isfinite(total)):
            raise ValueError('invalid host timing')
        if type(times['differences']) is not int or not 0 <= times['differences'] <= pairs:
            raise ValueError('invalid comparison count')


def audit(root, archive, archive_plan_digest, limit):
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('invalid latency threshold')
    ops = root / 'ops'
    read = blocks.read
    if read(ops / 'terminal.json') != dict(exit_code=0, status='declared_stages_finished'):
        raise ValueError('incomplete workers')
    spec = read(root / 'package/entry/block-plan.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    probe = spec['single_card_probe']
    manifest = blocks.checked_plan(root, spec['queue']['namespace'])
    plan = manifest['plan']
    original = read(archive / 'plan.json')
    if not content_key(original['plan']) == original['sha256'] == archive_plan_digest:
        raise ValueError('archived plan differs from audited receipt')
    for k in ('inputs', 'policies', 'choice'):
        if plan[k] != original['plan'][k]:
            raise ValueError('archived input or policy changed')
    wanted = set(probe['trace_blocks'])
    if plan['blocks'] != [b for b in original['plan']['blocks'] if b['key'] in wanted]:
        raise ValueError('timing population differs from registration')
    if read(ops / 'block-plan-collected.json')['plan_sha256'] != manifest['sha256']:
        raise ValueError('unexpected collected plan')
    group_trials = defaultdict(list)
    counts = defaultdict(lambda: dict(blocks=0, pairs=0, recordings=0))
    receipts = {}
    visited = set()
    expected = {b['key']: b for b in plan['blocks']}
    for f in sorted(ops.glob('block-worker-*-collected.json')):
        receipt = read(f)
        if (receipt['plan_sha256'] != manifest['sha256'] or receipt['npu_executed'] is not True
                or receipt['labels_used'] is not False):
            raise ValueError('incomplete device receipt')
        receipts[f.name] = file_hash(f)
        for row in receipt['diagnostic']:
            key = row['block']
            b = expected[key]
            group = (b['dataset'], b['arm'])
            if key in visited or (row['dataset'], row['arm']) != group:
                raise ValueError('duplicated or misassigned block')
            visited.add(key)
            pairs, recordings = sum(b['widths']), len(b['ids'])
            if (row['pairs'] != pairs or row['recordings'] != recordings
                    or row['policy_id'] != plan['policies'][b['arm']]
                    or row['reference'] != probe['latency_reference'][b['dataset']]
                    or row['mode'] != spec['endpoints'][b['dataset']]['single_card_mode']):
                raise ValueError('unexpected replay or population')
            archived = blocks.read_block(archive, original, b)['prob']
            serialized = json.dumps(archived, separators=(',', ':'), allow_nan=False).encode()
            if row['reference_sha256'] != hashlib.sha256(serialized).hexdigest():
                raise ValueError('replayed reference differs from archived probabilities')
            trials = row['trials']
            if len(trials) != probe['latency_rounds'] or len(trials) % 2:
                raise ValueError('unbalanced timing rounds')
            counts[group]['pairs'] += pairs
            counts[group]['recordings'] += recordings
            counts[group]['blocks'] += 1
            for trial in trials:
                validate_trial(trial, pairs)
                group_trials[group].append(trial)
    if visited != wanted or not group_trials:
        raise ValueError('incomplete registered replay')
    endpoints = {}
    for (dataset, arm), trials in group_trials.items():
        summary = dict(counts[dataset, arm], measured_block_traversals=len(trials))
        for kind in ('standard', 'candidate'):
            times = [t[kind] for t in trials]
            decisions = [t['first_decision_seconds'] for t in times]
            summary[kind] = dict(first_decision_p50=percentile(decisions, .50),
                                 first_decision_p95=percentile(decisions, .95),
                                 scoring_seconds=sum(t['scoring_seconds'] for t in times),
                                 probability_differences=sum(t['differences'] for t in times))
        ratios = {k: summary['candidate'][k] / summary['standard'][k]
                  for k in ('first_decision_p50', 'first_decision_p95', 'scoring_seconds')}
        summary.update(ratios=ratios, passed=(summary['candidate']['probability_differences'] == 0
                                              and max(ratios.values()) <= limit))
        endpoints.setdefault(dataset, {})[arm] = summary
    return dict(scope='finite warm block replay', max_time_ratio=limit, plan_sha256=manifest['sha256'],
                archive_plan_sha256=archive_plan_digest, receipts=receipts, endpoints=endpoints,
                passed=all(s['passed'] for arms in endpoints.values() for s in arms.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'archive', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--archive-plan-digest', required=True)
    p.add_argument('--limit', type=float, required=True)
    args = p.parse_args()
    result = audit(args.root, args.archive, args.archive_plan_digest, args.limit)
    commit_json(args.output, result)
    raise SystemExit(0 if result['passed'] else 1)
