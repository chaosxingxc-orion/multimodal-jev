"""Read a complete archived run; report phase costs without remote polling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from multimodal_jev.local_config import load_config


def summarize(root, updates, epoch_updates):
    root = Path(root)
    if type(updates) is not int or type(epoch_updates) is not int or min(updates, epoch_updates) < 1 or updates % epoch_updates:
        raise ValueError('complete equal-length epochs required')
    rows = []; sources = {}
    for step in range(1, updates + 1):
        ranks = []
        for rank in range(4):
            p = root / f'rank-{rank}' / f'update-{step:04d}.json'
            raw = p.read_bytes(); d = json.loads(raw)
            sources[str(p.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            if d['update'] != step:
                raise ValueError('update identity mismatch')
            for key in ('total_seconds', 'preparation_seconds', 'actor_seconds'):
                if not isinstance(d[key], (int, float)) or not math.isfinite(d[key]) or d[key] < 0:
                    raise ValueError('invalid timer')
            if d['preparation_seconds'] + d['actor_seconds'] > d['total_seconds'] + 1e-6:
                raise ValueError('phase times exceed step')
            ranks.append(d)
        d = ranks[0]; r = d['rollout']
        if any(x['real_queries'] != d['real_queries'] for x in ranks):
            raise ValueError('global exposure differs across ranks')
        if not 0 <= r['reference_seconds'] <= r['total_seconds'] <= d['preparation_seconds'] + 1e-6:
            raise ValueError('invalid nested rollout timers')
        rows.append(dict(step=step, real_queries=d['real_queries'],
            total=max(x['total_seconds'] for x in ranks),
            preparation=d['preparation_seconds'], actor=d['actor_seconds'],
            rollout=r['total_seconds'], reference=r['reference_seconds'],
            reference_misses=r['reference_misses'],
            residual=d['total_seconds']-d['preparation_seconds']-d['actor_seconds']))
    windows = []
    for start in range(0, updates, epoch_updates):
        rs = rows[start:start+epoch_updates]
        windows.append(dict(epoch=start//epoch_updates+1, updates=len(rs),
            real_query_exposures=sum(x['real_queries'] for x in rs),
            reference_misses=sum(x['reference_misses'] for x in rs),
            mean_seconds={k:statistics.mean(x[k] for x in rs) for k in
                          ('total','preparation','actor','rollout','reference','residual')}))
    return dict(schema='context-rollout-efficiency-summary-v1', epochs=windows,
        source_sha256=sources, steps=updates,
        scope='Archived timers: total is slowest rank; other phases rank0. Nested timers must not be added. Not kernel-only time or new acceleration evidence.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); config, provenance = load_config(args.config, args.local)
    report = summarize(config['run_root'], config['updates'], config['epoch_updates'])
    report['config_provenance'] = provenance
    report['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(config['output']); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
