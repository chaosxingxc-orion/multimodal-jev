"""After the finite two-arm gate, measure two isolated full evaluations sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.audit_card_latency import audit
from tools.reports.finalize_context_infonce import wait_exact


def run(settings):
    cfg = json.loads(settings.read_bytes())
    output = settings.parent
    finite = Path(cfg['finite'])
    wait_exact(json.loads((finite/'ops.started.json').read_bytes()), finite/'ops/terminal.json')
    gate = audit(finite, finite/'aligned-sample', cfg['archive_plan_digest'], cfg['limit'])
    commit_json(output/'finite-latency-audit.json', gate)
    if not gate['passed']:
        raise ValueError('finite latency or probability gate failed; full jobs not dispatched')
    results = {}
    for role in ('standard', 'candidate'):
        root = Path(cfg[role])
        template, local = root/'template.json', root/'dispatch.json'
        resolved, _ = load_config(template, local)
        if (list(resolved['sample_blocks']['devices']) != cfg['devices']
                or resolved['performance_comparison']['role'] != role
                or not cfg['devices'] or any(type(i) is not int or i < 0 for i in cfg['devices'])
                or len(set(cfg['devices'])) != len(cfg['devices'])):
            raise ValueError('unexpected devices or role')
        start = time.monotonic()
        subprocess.run([sys.executable, '-B', '-m', 'tools.reports.prepare_paired_card_evaluation',
                        str(template), str(local)], check=True)
        subprocess.run([sys.executable, '-B', '-m', 'tools.reports.native_seal_operations',
                        '--config', 'configs/portable/operations/native-seal-operations.json', '--local',
                        str(root/'package/operations-local.json')], check=True)
        ops = root/'ops'
        score = json.loads((ops/'score-collected.json').read_bytes())
        if (json.loads((ops/'terminal.json').read_bytes()) != dict(exit_code=0, status='declared_stages_finished')
                or score['schema'] != 'jev-performance-replay-v1' or score['comparison']['role'] != role
                or (role == 'candidate' and score['comparison']['differences'] != 0)):
            raise ValueError('unexpected collection or parity result')
        results[role] = dict(seconds=time.monotonic()-start, score_sha256=file_hash(ops/'score-collected.json'),
                             archive_sha256=score['comparison']['archive_sha256'])
        commit_json(output/(role+'-timing.json'), results[role])
    if results['standard']['archive_sha256'] != results['candidate']['archive_sha256']:
        raise ValueError('different reference archives')
    ratio = results['candidate']['seconds']/results['standard']['seconds']
    commit_json(output/'comparison.json', dict(runs=results, wall_ratio=ratio, passed=ratio <= cfg['limit']))
    if ratio > cfg['limit']:
        raise ValueError('full wall clock exceeded the registered limit')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('settings', type=Path)
    c = p.parse_args()
    try:
        run(c.settings)
    except Exception as exc:
        commit_json(c.settings.parent/'terminal.json', dict(exit_code=1, error=str(exc)))
        raise
    commit_json(c.settings.parent/'terminal.json', dict(exit_code=0))
