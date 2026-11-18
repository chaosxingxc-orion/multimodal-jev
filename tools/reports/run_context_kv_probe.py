"""Isolate KV capacity with the admitted acoustic implementation in both arms."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import require_training_storage
from tools.reports.measure_context_acoustic_candidate import compare as compare_acoustic, run as measure
from tools.reports.run_context_batch_probe_candidate import verify


def arm_config(config, arm):
    if arm not in ('baseline', 'candidate'):
        raise ValueError('explicit KV arm required')
    probe = config['kv_probe']
    base, candidate = probe['baseline_bytes'], probe['candidate_bytes']
    if (type(base) is not int or type(candidate) is not int or not 0 < base < candidate
            or config['policy']['kv_cache_memory_bytes'] != base
            or config['policy']['max_batch_queries'] != 32
            or config['policy']['max_batch_tokens'] != 131072):
        raise ValueError('fixed service32 and increasing explicit KV capacities required')
    result = deepcopy(config)
    result['policy']['kv_cache_memory_bytes'] = base if arm == 'baseline' else candidate
    return result


def compare(reports, rows, config):
    if set(reports) != {'baseline', 'candidate'}:
        raise ValueError('two KV arms required')
    for arm, report in reports.items():
        if (report['executed_policy'] != arm_config(config, arm)['policy']
                or report['mode'] != 'acoustic'):
            raise ValueError('both arms must execute identical acoustic path and declared capacity')
    # Reuse full input, actual forward coverage, cold-audio, logits and fixed-G16 audit.
    shared = compare_acoustic({'baseline': reports['baseline'], 'acoustic': reports['candidate']}, rows)
    means = shared.pop('mean_seconds')
    shared['mean_seconds'] = dict(baseline=means['baseline'], candidate=means['acoustic'])
    calls = shared.pop('acoustic_calls')
    shared['acoustic_calls'] = dict(baseline=calls['baseline'], candidate=calls['acoustic'])
    details = {}
    for arm, report in reports.items():
        details[arm] = dict(kv_cache_memory_bytes=report['executed_policy']['kv_cache_memory_bytes'],
            timed_passes=[dict(seconds=p['seconds'],
                actual_forward_sizes={str(k): v for k, v in sorted(Counter(map(len, p['telemetry']['counts']['language'])).items())},
                submitted_batch_queries=p['plan']['batch_queries'],
                submitted_batch_tokens=p['plan']['batch_tokens'],
                peak_allocated_bytes=p['telemetry']['peak_allocated_bytes'],
                peak_reserved_bytes=p['telemetry']['peak_reserved_bytes']) for p in report['passes'][1:3]])
    return dict(shared, arms=details, schema='jev-kv-capacity-comparison-v1',
        scope='fixed-policy TRAIN128 cold-audio service32 KV-only comparison; no training admission')


def run(package, output):
    import torch
    package, output = Path(package), Path(output)
    require_training_storage(output); output.mkdir(exist_ok=False)
    config = json.loads((package/'config.json').read_text())
    try:
        commit_json(output/'source.json', dict(files_sha256=verify(package)))
        reports = {}
        for arm in ('baseline', 'candidate'):
            arm_config(config, arm)
            argv = [sys.executable, '-B', str(package/'tools/reports/run_context_kv_probe.py'),
                    '--package', str(package), '--output', str(output/arm), '--arm', arm]
            commit_json(output/(arm+'-command.json'), dict(argv=argv))
            with (output/(arm+'.log')).open('x') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            commit_json(output/(arm+'-exit.json'), dict(exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError(arm+' failed; preserve logs and do not retry')
            reports[arm] = json.loads((output/arm/'kv-measurement.json').read_text())
        rows = torch.load(package/'inputs.pt', map_location='cpu', weights_only=True)
        result = compare(reports, rows, config)
        commit_json(output/'comparison.json', result)
        commit_json(output/'terminal.json', dict(exit_code=0, training_dispatch_allowed=False))
        return result
    except BaseException as exc:
        commit_json(output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True); p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=['baseline', 'candidate'])
    args = p.parse_args()
    if args.arm:
        config = json.loads((Path(args.package)/'config.json').read_text())
        config = arm_config(config, args.arm)
        report = measure(config, args.output, 'acoustic')
        report['executed_policy'] = config['policy']
        commit_json(Path(args.output)/'kv-measurement.json', report)
    else:
        run(args.package, args.output)
