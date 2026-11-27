"""Apply a verified scoring configuration to the complete heldout evaluation."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.evaluation_blocks import read
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.prepare_paired_card_evaluation import run as prepare, settings as container_spec
from tools.reports.screen_card_evaluation_replicas import bound, screen


def config(root):
    return read(root/'package/operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']


def timings(root, plan, replicas, *, reused_policy=None):
    ops = root/'ops'
    if read(ops/'terminal.json') != dict(exit_code=0, status='declared_stages_finished'):
        raise ValueError('incomplete evaluation operations')
    dep = config(root)
    chain, = dep['chains']
    group, = (group for group in chain if all(j['name'].startswith('block-worker-') for j in group))
    registered = {row['key']: row for row in plan['blocks']}
    if len(registered) != len(plan['blocks']):
        raise ValueError('duplicate registered evaluation blocks')
    source = root/'package/entry/block-plan.json'
    if file_hash(source) != dep['entry_pins']['block-plan.json']:
        raise ValueError('installed block plan configuration changed')
    queue = read(source)['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['queue']
    if 'resume_from' in queue or queue.get('reuse_policy') != reused_policy:
        raise ValueError('inference comparison requires the declared completed-policy reuse only')
    reused_arm = reused_policy['arm'] if reused_policy else None
    expected = {key: row for key, row in registered.items() if row['arm'] != reused_arm}
    witness = read(ops/'block-plan-collected.json')
    if (not expected or witness['blocks'] != len(registered)
            or witness['reused_blocks'] != len(registered)-len(expected)
            or (reused_arm is not None and len(expected) == len(registered))):
        raise ValueError('reused and inferred blocks must match the registered workload')
    starts, ends, first_inputs, committed = [], [], [], []
    devices, seen = set(), set()
    for job in group:
        name = job['name']
        launch = read(ops/(name+'-launch.json'))
        record = read(ops/(name+'-inspect.json'))
        state = record['State']
        worker = read(ops/(name+'-collected.json'))
        device, = job['devices']
        if (device in devices or worker['devices'] != [device]
                or worker['plan_sha256'] != witness['plan_sha256']
                or worker.get('replicas_per_device', 1) != replicas
                or record['Id'] != launch['container_id'] or record['Image'] != dep['image']
                or record['Name'] != '/'+dep['tag']+'-'+name
                or state['Status'] != 'exited' or state['ExitCode']):
            raise ValueError('worker receipt differs from its physical device or container')
        devices.add(device)
        start, end = (datetime.fromisoformat(state[key].replace('Z', '+00:00')).timestamp()
                      for key in ('StartedAt', 'FinishedAt'))
        starts.append(start)
        ends.append(end)
        for row in worker['blocks']:
            key = row['block']
            block = expected.get(key)
            first, last = row['first_input_unix'], row['committed_unix']
            if (key in seen or block is None or (row['dataset'], row['arm'], row['recordings']) !=
                    (block['dataset'], block['arm'], len(block['ids']))
                    or not start <= row['claimed_unix'] <= first < last <= end):
                raise ValueError('invalid, foreign or duplicate evaluation block timing')
            seen.add(key)
            first_inputs.append(first)
            committed.append(last)
    if seen != expected.keys():
        raise ValueError('timing omits required inference blocks')
    return dict(devices=sorted(devices), scoring_start=min(first_inputs), scoring_end=max(committed),
                worker_wall_seconds=max(ends)-min(starts), host=dep['host'], image=dep['image'])


def utilization(path, interval, devices, max_gap):
    start, end = interval
    keys = tuple(map(str, devices))
    if not keys or len(set(keys)) != len(keys) or not start < end or not max_gap > 0:
        raise ValueError('valid scoring interval, physical device scope and sampling gap required')
    indexes = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if 'devices' in row and all(device in row['devices'] for device in keys):
                stamp = row['unix']
                if not math.isfinite(stamp) or stamp in indexes:
                    raise ValueError('invalid device sampling chronology')
                indexes[stamp] = row['devices']
    usable = sorted(t for t in indexes if start-max_gap <= t <= end+max_gap)
    if not usable or usable[0] > start or usable[-1] < end:
        raise ValueError('device readings must cover the full scoring window')
    totals = {key: dict(aicore=0., hbm=0.) for key in keys}
    for left, right in zip(usable, usable[1:]):
        overlap_start, overlap_end = max(left, start), min(right, end)
        seconds = overlap_end - overlap_start
        if seconds <= 0:
            continue
        if right-left > max_gap:
            raise ValueError('gap in device readings inside the scoring window')
        for device, by_metric in totals.items():
            for metric in by_metric:
                a, b = indexes[left][device][metric], indexes[right][device][metric]
                if not 0 <= a <= 100 or not 0 <= b <= 100:
                    raise ValueError('invalid device utilization')
                fraction = ((overlap_start-left)+(overlap_end-left))/(2*(right-left))
                by_metric[metric] += seconds*(a+(b-a)*fraction)
    by_device = {k: {m: v/(end-start) for m,v in stats.items()} for k,stats in totals.items()}
    return dict(by_device=by_device, **{key:sum(stats[key] for stats in by_device.values())/len(keys)
                                       for key in ('aicore', 'hbm')})


def worker_contract(root):
    dep = config(root)
    path = root/'package/entry/block-worker-0.json'
    opts = container_spec(dict(path=path, sha256=dep['entry_pins'][path.name]))
    job, = (j for group in dep['chains'][0] for j in group if j['name'] == 'block-worker-0')
    return dict(replicas=opts.get('replicas_per_device', 1),
                head_check=opts.get('eval_single_validation', False),
                container={k: job[k] for k in ('cpus', 'memory', 'shm')})


def fixed_baseline(comparison, pair):
    pin = comparison['screen_audit']
    gate = bound(pin)
    screen_root = Path(pin['path']).parent.parent
    screen_spec_path = screen_root/'package/operations-local.json'
    screened = bound(dict(path=screen_spec_path, sha256=gate['deployment_sha256']))
    screened = screened['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    requested = dict(replicas=pair['sample_blocks']['replicas_per_device'],
                     head_check=pair['eval_single_validation'], container=pair['eval_container'])
    minimums = comparison['minimums']
    count = str(requested['replicas'])
    result = gate['comparison'][count]
    if (gate['scope'] != 'matched fixed samples, physical cards per arm'
            or gate['recordings'] != comparison['screen_recordings']
            or result['passed'] is not True
            or result['aicore'] < minimums['aicore'] or result['hbm'] < minimums['hbm']
            or min(result['speedup'], result['container_speedup']) <= minimums['speedup']
            or any(screened[k] != pair['deployment'][k] for k in ('host', 'image'))):
        raise ValueError('formal evaluation requires a passing matched-sample baseline')
    jobs = [j for j in screened['chains'][0][0] if j['name'].startswith(f'replay-{count}-')]
    lanes = gate['lanes'][count]
    if (not jobs or {d for j in jobs for d in j['devices']} != {row['device'] for row in lanes}
            or any(row['differences'] or not row['collected'] for row in lanes)):
        raise ValueError('matched-sample baseline has incomplete candidate lanes')
    for job in jobs:
        name = job['name']+'.json'
        entry = screen_root/'package/entry'/name
        opts = container_spec(dict(path=entry, sha256=screened['entry_pins'][name]))
        contract = dict(replicas=opts.get('replicas_per_device', 1),
                        head_check=opts.get('eval_single_validation', False),
                        container={k: job[k] for k in ('cpus', 'memory', 'shm')})
        if contract != requested:
            raise ValueError('requested worker allocation differs from the matched-sample baseline')
    return requested, gate, result


def run_formal(pair):
    """Score the complete holdouts and audit a fixed subset against prior probabilities."""
    deployment = dict(pair['deployment'])
    comparison = deployment.pop('throughput_comparison', None)
    pair = dict(pair, deployment=deployment)
    root = Path(pair['output']).parent
    if Path(pair['output']) != root/'package' or Path(deployment['operations']) != root/'ops':
        raise ValueError('formal evaluation package and operations must share an isolated run root')
    if not isinstance(comparison, dict) or set(comparison) != {
            'screen_audit', 'screen_recordings', 'archive', 'sampling_seconds',
            'max_sampling_gap_seconds', 'minimums'}:
        raise ValueError('incomplete formal throughput comparison')
    approved, gate, speed = fixed_baseline(comparison, pair)
    pin = comparison['archive']
    manifest = bound(pin['local'])
    plan = manifest['plan']
    reuse = deployment['evaluation_options']['reuse_stage1_evaluation']
    expected_reuse = dict(arm='stage1', plan=reuse['remote_plan'])
    selected = set(pin['blocks'])
    population = Counter()
    sampled = Counter()
    for block in plan['blocks']:
        key = (block['dataset'], block['arm'])
        pairs = sum(block['widths'])
        population[key] += pairs
        if block['key'] in selected and block['arm'] != 'stage1':
            sampled[key] += pairs
    if (manifest['sha256'] != content_key(plan) or not sampled
            or len(selected) != len(pin['blocks'])
            or sum(b['key'] in selected and b['arm'] != 'stage1' for b in plan['blocks']) != len(selected)
            or pair['fit_harness_thresholds'] is not False):
        raise ValueError('invalid pinned Stage2 archive subset or unexpected threshold fit')
    pair['exact_probability_archive'] = dict(path=pin['remote'], sha256=pin['local']['sha256'],
                                             blocks=pin['blocks'])
    started = time.monotonic()
    prepare(pair)
    frozen = config(root)
    if worker_contract(root) != approved:
        raise ValueError('frozen worker settings differ from the approved allocation')
    roles = {j['name'] for group in frozen['chains'][0] for j in group if j['name'].startswith('block-worker-')}
    if screen(comparison, frozen, observed_roles=roles):
        raise ValueError('formal evaluation failed')
    elapsed = time.monotonic()-started
    candidate = timings(root, plan, approved['replicas'], reused_policy=expected_reuse)
    observed = read(root/'ops/score-collected.json')
    parity = observed['archive_parity']
    actual = {(ds, arm): row['pairs'] for ds, arms in parity['panels'].items() for arm, row in arms.items()}
    if (observed['schema'] != 'jev-audited-probability-panels-v1' or 'thresholds_sha256' in observed
            or parity['differences'] or parity['blocks'] != len(selected) or actual != sampled
            or parity['archive_sha256'] != pin['local']['sha256']):
        raise ValueError('formal metrics or archived sample probabilities differ')
    window = (candidate['scoring_start'], candidate['scoring_end'])
    device = utilization(root/'ops/telemetry.jsonl', window, candidate['devices'],
                         comparison['max_sampling_gap_seconds'])
    minimums = comparison['minimums']
    passed = device['aicore'] >= minimums['aicore'] and device['hbm'] >= minimums['hbm']
    inferred = sum(n for (_, arm), n in population.items() if arm != 'stage1')
    commit_json(root/'formal-throughput-audit.json', dict(
        screen_audit_sha256=comparison['screen_audit']['sha256'], screen_recordings=gate['recordings'],
        screen_probabilities=gate['pairs'], screen_scoring_speedup=speed['speedup'],
        screen_container_speedup=speed['container_speedup'], archive_sha256=pin['local']['sha256'],
        replicas=approved['replicas'], inferred_blocks=sum(b['arm'] != 'stage1' for b in plan['blocks']),
        reused_blocks=sum(b['arm'] == 'stage1' for b in plan['blocks']),
        compared_blocks=len(selected), compared_probabilities=sum(sampled.values()),
        differences=parity['differences'], inferred_probabilities=inferred,
        scoring_probabilities_per_second=inferred/(window[1]-window[0]),
        candidate=candidate, full_candidate_wall_seconds=elapsed, **device, passed=passed))
    if not passed:
        raise ValueError('formal evaluation did not meet the device utilization targets')
    return 0



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    parser.add_argument('--formal', action='store_true', required=True)
    args = parser.parse_args()
    settings, _ = load_config(args.config, args.local)
    raise SystemExit(run_formal(settings))


if __name__ == '__main__':
    main()
