"""Read-only matched timing comparison of a committed prefix; never select a winner.

Explicit invocation only. This is not a full training/transport correctness audit.
Paths are supplied by local macros; no periodic polling or model calls.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

from multimodal_jev.local_config import load_config
from tools.reports.audit_context_macro_training import activation
from tools.reports.compare_context_service_batch import controls


def snapshot(root, steps):
    root = Path(root)
    rows = []
    for step in range(1, steps + 1):
        hashes = {}
        def read(name):
            raw = (root / name).read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)
        commit = read(f'checkpoint-{step:04d}-committed.json')
        ranks = [read(f'rank-{r}/update-{step:04d}.json') for r in range(2)]
        ack = read(f'activation-{step:04d}.json')
        batch = read(f'batch-{step-1:04d}.json')
        pid = ranks[0]['next_policy_id']
        if (commit['updates'] != step or commit['joint_policy']['policy_id'] != pid
                or ack['policy_id'] != pid or ack['error'] is not None):
            raise ValueError('committed policy/activation mismatch')
        rollout = ranks[0]['rollout']
        if ranks[1]['rollout'] != rollout:
            raise ValueError('rank rollout mismatch')
        compact = []
        for r in ranks:
            if r['update'] != step or r['real_queries'] != 128 or r['next_policy_id'] != pid:
                raise ValueError('incomplete matching DP2 step')
            compact.append({k: r[k] for k in ('total_seconds', 'preparation_seconds', 'actor_seconds')})
        rows.append(dict(update=step, ranks=compact, activation=ack,
            records_sha256=hashlib.sha256(json.dumps(batch['records'], sort_keys=True).encode()).hexdigest(),
            rollout={k: rollout[k] for k in ('update', 'reference_misses', 'total_seconds',
                'reference_seconds', 'behavior_and_credit_seconds')}, files=hashes))
    return rows


def compare_rows(a, b):
    if not a or len(a) != len(b):
        raise ValueError('same nonempty prefix required')
    def summarize(rows):
        values = []
        for step, r in enumerate(rows, 1):
            activation(r['activation']['receipt'], r['activation']['policy_id'])
            if r['update'] != step or r['rollout']['update'] != step - 1 or len(r['ranks']) != 2:
                raise ValueError('gap in matched prefix')
            roll = r['rollout']; ranks = r['ranks']; rank = ranks[0]
            timers = [v for p in ranks for v in p.values()] + [roll[k] for k in
                ('total_seconds', 'reference_seconds', 'behavior_and_credit_seconds')]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in timers):
                raise ValueError('invalid timer')
            if any(p['total_seconds'] <= 0 or p['preparation_seconds'] + p['actor_seconds'] > p['total_seconds'] + 1e-6 for p in ranks):
                raise ValueError('invalid phase partition')
            if (roll['reference_seconds'] + roll['behavior_and_credit_seconds'] > roll['total_seconds'] + 1e-6
                    or roll['total_seconds'] > rank['preparation_seconds'] + 1e-6
                    or type(roll['reference_misses']) != int or not 0 <= roll['reference_misses'] <= 128):
                raise ValueError('invalid rollout partition')
            values.append(dict(total=max(p['total_seconds'] for p in ranks), actor=rank['actor_seconds'],
                rollout=roll['total_seconds'], reference=roll['reference_seconds'],
                behavior_credit=roll['behavior_and_credit_seconds'],
                other_preparation=rank['preparation_seconds']-roll['total_seconds'],
                export_sync_residual=rank['total_seconds']-rank['preparation_seconds']-rank['actor_seconds']))
        total = sum(v['total'] for v in values)
        return dict(steps=len(rows), exposures=128*len(rows), seconds=total,
            queries_per_second=128*len(rows)/total,
            mean_seconds={k: statistics.mean(v[k] for v in values) for k in values[0]})
    for old, new in zip(a, b, strict=True):
        if (old['records_sha256'] != new['records_sha256']
                or old['rollout']['reference_misses'] != new['rollout']['reference_misses']):
            raise ValueError('sample order or reference cache condition differs')
    old, new = summarize(a), summarize(b)
    return dict(baseline=old, candidate=new, speedup=old['seconds']/new['seconds'],
        time_reduction_fraction=1-new['seconds']/old['seconds'],
        full_100step_result=False, winner_selection_allowed=False,
        scope='Matched committed prefix timers only; no full runtime, correctness, efficacy or transfer admission.')


def run(c):
    steps = c['steps']
    if type(steps) != int or not 0 < steps < 100:
        raise ValueError('explicit prefix length 1..99 required')
    ops = [json.loads(Path(c[k]).read_text()) for k in ('baseline_operations', 'candidate_operations')]
    (a, ac), (b, bc) = [controls(o) for o in ops]
    if a != b or ac != [(4, 16384)]*2 or bc != [(32, 131072)]*2:
        raise ValueError('only service capacity may change')
    out = Path(c['output']); out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(c, indent=2)+'\n')
    # Send only the standard-library snapshot function, without importing project code remotely.
    import inspect
    code = ('import json, hashlib\nfrom pathlib import Path\n' + inspect.getsource(snapshot)
        + '\nprint(json.dumps(snapshot(' + repr(ops[1]['remote_root']+'/run') + ', '+str(steps)+')))')
    r = subprocess.run(['ssh', '-o', 'ConnectTimeout=10', ops[1]['host'],
        shlex.join(['python3', '-B', '-c', code])], capture_output=True, text=True, timeout=55)
    (out/'remote-stderr.txt').write_text(r.stderr)
    r.check_returncode()
    new = json.loads(r.stdout)
    (out/'candidate-snapshot.json').write_text(r.stdout)
    old = snapshot(Path(ops[0]['operations_root'])/'collected/run', steps)
    (out/'baseline-snapshot.json').write_text(json.dumps(old)+'\n')
    result = compare_rows(old, new)
    result['operations'] = ops
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); config, _ = load_config(args.config, args.local)
    print(json.dumps(run(config), indent=2))
