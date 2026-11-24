"""Reconcile full-evaluation worker timings with both arms' sealed probability population."""
import argparse
from collections import defaultdict
from datetime import datetime
import math
from pathlib import Path

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.audit_card_latency import percentile


def metrics(rows):
    # Claim and commit are on the same server, and workers can overlap.
    # Within-block host latency is monotonic; it begins at the scoring timer.
    return dict(blocks=len(rows), recordings=sum(r['recordings'] for r in rows),
                claim_commit_span_seconds=max(r['committed_unix'] for r in rows) -
                                          min(r['claimed_unix'] for r in rows),
                **{name: percentile([r['first_decision_seconds'] for r in rows], p)
                   for name, p in [('first_decision_p50', .50), ('first_decision_p95', .95)]})


def collect(root, role, registration, wall):
    ops = root / 'ops'
    if (blocks.read(ops/'terminal.json') != dict(exit_code=0, status='declared_stages_finished')
            or file_hash(ops/'score-collected.json') != wall['score_sha256']):
        raise ValueError('incomplete full evaluation')
    score = blocks.read(ops/'score-collected.json')
    ref = score['comparison']
    if (score['schema'] != 'jev-performance-replay-v1' or ref['role'] != role
            or ref['archive_sha256'] != wall['archive_sha256'] or
            (role == 'candidate' and ref['differences'] != 0)):
        raise ValueError('full population was not audited against the archive')
    entry = root/'package/entry/block-plan.json'
    spec = blocks.read(entry)['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    plan = blocks.read(ops/'block-plan-collected.json')
    if (spec['performance_comparison']['role'] != role or plan['reused_blocks']
            or plan['blocks'] != len(registration)):
        raise ValueError('full, newly timed work is required')
    seen = set()
    panels = defaultdict(list)
    arms = defaultdict(list)
    sources = {}
    for path in sorted(ops.glob('block-worker-*-collected.json')):
        row = blocks.read(path)
        if row['performance_role'] != role or row['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('foreign worker result')
        sources[path.name] = file_hash(path)
        for block in row['blocks']:
            key = block['block']
            expected = registration[key]
            arm, dataset = block['arm'], block['dataset']
            if (key in seen or (arm, dataset, block['recordings']) !=
                    (expected['arm'], expected['dataset'], len(expected['ids']))):
                raise ValueError('foreign, replayed, or duplicate timing')
            claim, first_input, commit = (block[k] for k in ('claimed_unix','first_input_unix','committed_unix'))
            first, decision, total = (block[k] for k in ('first_score_seconds','first_decision_seconds','scoring_seconds'))
            if (not all(math.isfinite(v) for v in (claim, first_input, commit, first, decision, total))
                    or not claim <= first_input <= commit or not 0 < first <= decision <= total):
                raise ValueError('invalid timing boundaries')
            seen.add(key)
            arms[arm].append(block)
            panels[dataset,arm].append(block)
    if seen != registration.keys():
        raise ValueError('missing registered block timing')
    populations = {(d,a):v['recordings'] for d, by_arm in ref['panels'].items() for a,v in by_arm.items()}
    if populations != {k:sum(r['recordings'] for r in v) for k,v in panels.items()}:
        raise ValueError('worker and scoring populations differ')
    return dict(endpoints={arm:metrics(v) for arm,v in arms.items()},
                panels={d:{a:metrics(panels[d,a]) for a in ref['panels'][d]} for d in ref['panels']},
                receipts=sources)


def execution_lower_bound(root):
    """Exclude all inter-group waiting and setup; never add overlapping workers."""
    dep = blocks.read(root/'package/operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    chain, = dep['chains']
    groups = []
    last_finish = None
    for group in chain:
        intervals = []
        for job in group:
            name = job['name']
            run = blocks.read(root/'ops'/(name+'-inspect.json'))
            launch = blocks.read(root/'ops'/(name+'-launch.json'))
            state = run['State']
            if (run['Id'] != launch['container_id'] or run['Name'] != '/'+dep['tag']+'-'+name
                    or state['ExitCode'] != 0 or state['Status'] != 'exited'):
                raise ValueError('missing successful timed container: '+name)
            start, end = [datetime.fromisoformat(state[k].replace('Z','+00:00'))
                          for k in ('StartedAt','FinishedAt')]
            if start >= end or (last_finish is not None and start < last_finish):
                raise ValueError('invalid serial group boundaries: '+name)
            intervals.append((start,end))
        last_finish = max(end for _,end in intervals)
        groups.append(max((end-start).total_seconds() for start,end in intervals))
    return sum(groups)


def audit(settings, archive):
    root = settings.parent
    cfg = blocks.read(settings)
    manifest = blocks.read(archive)
    if manifest['sha256'] != blocks.content_key(manifest['plan']) or manifest['sha256'] != cfg['archive_plan_digest']:
        raise ValueError('wrong archive plan')
    registration = {b['key']: b for b in manifest['plan']['blocks']}
    if len(registration) != len(manifest['plan']['blocks']):
        raise ValueError('duplicate archived block')
    ab = blocks.read(root/'comparison.json')
    output = {role:collect(Path(cfg[role]), role, registration, ab['runs'][role])
              for role in ('standard','candidate')}
    ratios = {a:{key:v[key]/output['standard']['endpoints'][a][key] for key in
                 ('first_decision_p50','first_decision_p95','claim_commit_span_seconds')}
              for a,v in output['candidate']['endpoints'].items()}
    lower = execution_lower_bound(Path(cfg['standard']))
    if not 0 < lower <= ab['runs']['standard']['seconds']:
        raise ValueError('container lower bound exceeds total native wall')
    upper_ratio = ab['runs']['candidate']['seconds'] / lower
    return dict(arms=output, endpoint_ratios=ratios, full_wall_ratio=ab['wall_ratio'],
                standard_execution_lower_bound_seconds=lower, conservative_wall_ratio_upper_bound=upper_ratio,
                passed=ab['passed'] and upper_ratio <= cfg['limit'] and all(max(v.values()) <= cfg['limit'] for v in ratios.values()),
                scope='host-visible complete candidate row; per-endpoint overlapping claim-to-commit span; full wall includes preparation and audit')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('settings','archive','output'):
        p.add_argument(k, type=Path)
    p.add_argument('--wait-for-driver', action='store_true')
    c = p.parse_args()
    if c.wait_for_driver:
        from tools.reports.finalize_context_infonce import wait_exact
        root = c.settings.parent
        wait_exact(blocks.read(root/'started.json'), root/'terminal.json')
        if blocks.read(root/'terminal.json')['exit_code'] != 0 and not (root/'comparison.json').exists():
            p.exit(1, 'full A/B not completed; endpoint timing unavailable\n')
    result = audit(c.settings, c.archive)
    commit_json(c.output, result)
    raise SystemExit(0 if result['passed'] else 1)
