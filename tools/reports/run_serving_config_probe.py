"""Run the declared serving-config arms sequentially and compare throughput.

Dev-state ``v0.0.1`` driver. Each arm runs in its own subprocess so NPU memory is
released between configurations and an over-budget arm fails without poisoning
the next. Every arm must return bit-identical records; the comparison then ranks
arms by serving-stage wall time. No optimizer, no training admission.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import sys

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import require_training_storage


def compare(reports):
    arms = list(reports)
    first = reports[arms[0]]
    for name, report in reports.items():
        if (report['input_sha256'] != first['input_sha256'] or report['rows'] != first['rows']
                or report['records_sha256'] != first['records_sha256']):
            raise ValueError('arm differs in inputs, rows or exact records: ' + name)
    baseline = first['mean_seconds']
    rows = []
    for name, report in reports.items():
        rows.append(dict(arm=name, mean_seconds=report['mean_seconds'],
            rows_per_second=report['rows_per_second'],
            speedup_vs_first=baseline / report['mean_seconds'],
            executed=dict(gpu_memory_utilization=report['executed_policy']['gpu_memory_utilization'],
                kv_cache_memory_bytes=report['executed_policy']['kv_cache_memory_bytes'],
                max_batch_queries=report['executed_policy']['max_batch_queries'],
                max_batch_tokens=report['executed_policy']['max_batch_tokens'],
                max_model_len=report['executed_policy']['max_model_len'])))
    rows.sort(key=lambda r: r['mean_seconds'])
    return dict(schema='jev-serving-config-comparison-v1', first_arm=arms[0],
        rows=first['rows'], records_sha256=first['records_sha256'], arms=rows,
        exact_records_across_arms=True, best_arm=rows[0]['arm'],
        best_speedup_vs_first=rows[0]['speedup_vs_first'],
        training_dispatch_allowed=False,
        scope='serving-stage wall time over fixed records; engine startup excluded; no training')


def run(config_path, output):
    output = Path(output)
    require_training_storage(output)
    output.mkdir(exist_ok=False)
    config = json.loads(Path(config_path).read_text())
    commit_json(output / 'config.json', config)
    reports = OrderedDict()
    failed = {}
    for arm in config['arm_order']:
        if arm not in config['arms']:
            raise ValueError('unknown declared arm')
        arm_out = output / arm
        argv = [sys.executable, '-B', str(Path(__file__).with_name('measure_serving_config.py')),
                '--config', str(config_path), '--output', str(arm_out), '--arm', arm]
        commit_json(output / (arm + '-command.json'), dict(argv=argv))
        with (output / (arm + '.log')).open('x') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        commit_json(output / (arm + '-exit.json'), dict(exit_code=result.returncode))
        if result.returncode:
            # An over-budget arm is a measured boundary, not a reason to discard
            # the successful arms. Preserve its log and continue.
            terminal = arm_out / 'terminal.json'
            failed[arm] = json.loads(terminal.read_text()) if terminal.exists() else dict(exit_code=result.returncode)
            continue
        reports[arm] = json.loads((arm_out / 'measurement.json').read_text())
    if len(reports) < 1 or ('first_arm' in config and config['arm_order'][0] not in reports):
        raise RuntimeError('baseline arm must succeed; see per-arm logs')
    result = compare(reports)
    result['failed_arms'] = failed
    commit_json(output / 'comparison.json', result)
    commit_json(output / 'terminal.json', dict(exit_code=0, training_dispatch_allowed=False))
    return result


def compare_existing(config_path, output):
    """Compare arms that were run independently (e.g. in parallel containers)."""
    output = Path(output)
    config = json.loads(Path(config_path).read_text())
    reports = OrderedDict()
    failed = {}
    for arm in config['arm_order']:
        measurement = output / arm / 'measurement.json'
        if measurement.exists():
            reports[arm] = json.loads(measurement.read_text())
            continue
        terminal = output / arm / 'terminal.json'
        failed[arm] = json.loads(terminal.read_text()) if terminal.exists() else dict(missing=True)
    if not reports:
        raise RuntimeError('no successful arm measurement to compare')
    result = compare(reports)
    result['failed_arms'] = failed
    commit_json(output / 'comparison.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--compare-only', action='store_true',
                   help='compare already-produced per-arm measurement.json files')
    a = p.parse_args()
    result = (compare_existing(a.config, a.output) if a.compare_only
              else run(a.config, a.output))
    print(json.dumps(result, indent=2))
