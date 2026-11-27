"""Run the screened configuration on the complete, previously audited evaluation."""
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
from tools.reports.finalize_context_infonce import wait_exact
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


def wait_predecessor(predecessor):
    proof = bound(predecessor['driver'])
    identity = dict(pid=proof['pid'], process_start_ticks=proof['start_ticks'],
                    argv=proof.get('argv') or proof['args'])
    wait_exact(identity, predecessor['terminal'])


def worker_contract(root):
    dep = config(root)
    path = root/'package/entry/block-worker-0.json'
    opts = container_spec(dict(path=path, sha256=dep['entry_pins'][path.name]))
    job, = (j for group in dep['chains'][0] for j in group if j['name'] == 'block-worker-0')
    return dict(replicas=opts.get('replicas_per_device', 1),
                head_check=opts.get('eval_single_validation', False),
                container={k: job[k] for k in ('cpus', 'memory', 'shm')})


def run_formal(pair):
    """Dispatch the matched cutover through the experiment ledger."""
    deployment = dict(pair['deployment'])
    comparison = deployment.pop('throughput_comparison', None)
    pair = dict(pair, deployment=deployment)
    root = Path(pair['output']).parent
    if Path(pair['output']) != root/'package' or Path(deployment['operations']) != root/'ops':
        raise ValueError('formal evaluation package and operations must share an isolated run root')
    if not isinstance(comparison, dict) or set(comparison) != {
            'verification', 'baseline', 'baseline_score', 'archive', 'sampling_seconds',
            'max_sampling_gap_seconds', 'minimums'}:
        raise ValueError('incomplete formal throughput comparison')
    gate_pin = comparison['verification']
    gate = bound(gate_pin)
    approved = worker_contract(Path(gate_pin['path']).parent)
    minimums = comparison['minimums']
    if (gate['passed'] is not True or gate['differences'] != 0 or gate['replicas'] != approved['replicas']
            or gate['aicore'] < minimums['aicore'] or gate['hbm'] < minimums['hbm']
            or min(gate['scoring_speedup'], gate['full_wall_speedup_at_least']) <= minimums['speedup']):
        raise ValueError('formal cutover requires passing full-workload evidence')
    baseline = Path(comparison['baseline'])
    pin = comparison['archive']
    manifest = bound(pin['local'])
    plan = manifest['plan']
    prior = config(baseline)
    reuse = pair['deployment']['evaluation_options']['reuse_stage1_evaluation']
    expected_reuse = dict(arm='stage1', plan=reuse['remote_plan'])
    baseline_score_pin = comparison['baseline_score']
    baseline_score = bound(baseline_score_pin)
    if (manifest['sha256'] != content_key(plan)
            or manifest['sha256'] != read(baseline/'ops/block-plan-collected.json')['plan_sha256']
            or pin['remote'] != prior['remote_root']+'/sample-blocks/plan.json'
            or Path(baseline_score_pin['path']) != baseline/'ops/score-collected.json'
            or baseline_score['schema'] != 'jev-audited-probability-panels-v1'
            or 'thresholds_sha256' in baseline_score):
        raise ValueError('formal cutover must compare the same complete heldout run without threshold fits')
    original = timings(baseline, plan, 1, reused_policy=expected_reuse)
    if any(original[k] != gate['candidate'][k] for k in ('host', 'image', 'devices')):
        raise ValueError('formal launch and verification must use the same devices and runtime')
    requested = dict(replicas=pair['sample_blocks']['replicas_per_device'],
                     head_check=pair['eval_single_validation'], container=pair['eval_container'])
    if (requested != approved or pair['fit_harness_thresholds'] is not False
            or sorted(pair['sample_blocks']['devices']) != original['devices']
            or any(deployment[k] != original[k] for k in ('host', 'image'))):
        raise ValueError('formal runtime or workload does not match the verified configuration')
    pair['exact_probability_archive'] = dict(path=pin['remote'], sha256=pin['local']['sha256'])
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
    parity = observed.pop('archive_parity')
    populations = Counter()
    for block in plan['blocks']:
        populations[block['dataset'], block['arm']] += sum(block['widths'])
    actual = {(ds, arm): row['pairs'] for ds, arms in parity['panels'].items() for arm, row in arms.items()}
    if (observed != baseline_score or parity['differences'] or actual != populations
            or parity['archive_sha256'] != pin['local']['sha256']):
        raise ValueError('formal metrics or per-candidate probabilities differ')
    device = utilization(root/'ops/telemetry.jsonl', (candidate['scoring_start'], candidate['scoring_end']),
                         candidate['devices'], comparison['max_sampling_gap_seconds'])
    scoring_speed = ((original['scoring_end']-original['scoring_start'])
                     / (candidate['scoring_end']-candidate['scoring_start']))
    full_wall_floor = original['worker_wall_seconds']/elapsed
    passed = (device['aicore'] >= minimums['aicore'] and device['hbm'] >= minimums['hbm']
              and min(scoring_speed, full_wall_floor) > minimums['speedup'])
    commit_json(root/'formal-throughput-audit.json', dict(
        verification_sha256=gate_pin['sha256'], archive_sha256=pin['local']['sha256'],
        baseline_score_sha256=baseline_score_pin['sha256'],
        replicas=approved['replicas'], inferred_blocks=sum(b['arm'] != 'stage1' for b in plan['blocks']),
        reused_blocks=sum(b['arm'] == 'stage1' for b in plan['blocks']),
        compared_probabilities=sum(populations.values()), differences=parity['differences'],
        inferred_probabilities=sum(n for (_, arm), n in populations.items() if arm != 'stage1'),
        baseline=original, candidate=candidate, full_candidate_wall_seconds=elapsed,
        scoring_speedup=scoring_speed, full_wall_speedup_at_least=full_wall_floor,
        **device, passed=passed))
    if not passed:
        raise ValueError('formal evaluation did not reproduce the throughput targets')
    return 0


def run(settings, pair):
    root = Path(settings['output'])
    baseline = Path(settings['baseline'])
    pin = settings['archive']
    manifest = bound(pin['local'])
    plan = manifest['plan']
    if (manifest['sha256'] != content_key(plan) or
            manifest['sha256'] != read(baseline/'ops/block-plan-collected.json')['plan_sha256']):
        raise ValueError('archive population differs from the completed baseline')
    baseline_metrics = bound(settings['baseline_score'])
    if (Path(settings['baseline_score']['path']) != baseline/'ops/score-collected.json'
            or baseline_metrics['schema'] != 'jev-audited-probability-panels-v1'):
        raise ValueError('independently collected baseline score required')
    original = timings(baseline, plan, 1)
    for predecessor in settings['predecessors']:
        wait_predecessor(predecessor)
    screen_gate = settings['screen_gate']
    gate = read(Path(screen_gate))
    minimums = settings['minimums']
    passing = [int(replicas) for replicas, result in gate['comparison'].items()
               if result['passed'] and result['aicore'] >= minimums['aicore']
               and result['hbm'] >= minimums['hbm']
               and min(result['speedup'], result['container_speedup']) > minimums['speedup']]
    if not passing:
        raise ValueError('device concurrency screen did not qualify a candidate')
    # Select the highest measured end-to-end speed, preserving its resource envelope.
    count = max(passing, key=lambda n: gate['comparison'][str(n)]['container_speedup'])
    target = dict(pair, output=str(root/'package'), fit_harness_thresholds=False,
                  exact_probability_archive=dict(path=pin['remote'], sha256=pin['local']['sha256']),
                  eval_single_validation=settings['single_validation'],
                  sample_blocks=dict(pair['sample_blocks'], replicas_per_device=count),
                  eval_container=dict(pair['eval_container'], **settings['eval_container']))
    dep = target['deployment']
    if (dep['host'] != original['host'] or dep['image'] != original['image']
            or sorted(target['sample_blocks']['devices']) != original['devices']
            or Path(dep['operations']) != root/'ops'):
        raise ValueError('candidate must reuse the exact baseline host, image and physical cards')
    commit_json(root/'admission.json', dict(replicas=count, screen_sha256=file_hash(screen_gate),
                                           archive_sha256=pin['local']['sha256']))
    started = time.monotonic()
    prepare(target)
    deployment = config(root)
    roles = {job['name'] for chain in deployment['chains'] for group in chain for job in group
             if job['name'].startswith('block-worker-')}
    if screen(settings, deployment, observed_roles=roles):
        raise ValueError('full evaluation failed')
    elapsed = time.monotonic()-started
    candidate = timings(root, plan, count)
    scores = read(root/'ops/score-collected.json')
    parity = scores['archive_parity']
    populations = Counter()
    for block in plan['blocks']:
        populations[block['dataset'],block['arm']] += sum(block['widths'])
    actual = {(ds,arm):row['pairs'] for ds,arms in parity['panels'].items() for arm,row in arms.items()}
    if (parity['differences'] or parity['archive_sha256'] != pin['local']['sha256']
            or actual != populations or scores['seals'] != baseline_metrics['seals']):
        raise ValueError('full-run probability parity or scope differs')
    device = utilization(root/'ops/telemetry.jsonl',
                         (candidate['scoring_start'], candidate['scoring_end']),
                         candidate['devices'], settings['max_sampling_gap_seconds'])
    # Published baseline worker wall is a lower bound for all preparation and review.
    scoring_speed = ((original['scoring_end']-original['scoring_start'])
                     / (candidate['scoring_end']-candidate['scoring_start']))
    full_wall_floor = original['worker_wall_seconds']/elapsed
    result = dict(
        scope='two original endpoints, all heldout probability blocks, same eight physical cards',
        replicas=count, probabilities=sum(populations.values()), differences=parity['differences'],
        baseline=original, candidate=candidate, scoring_speedup=scoring_speed,
        full_candidate_wall_seconds=elapsed, full_wall_speedup_at_least=full_wall_floor,
        **device, passed=(device['aicore'] >= minimums['aicore'] and device['hbm'] >= minimums['hbm']
                          and min(scoring_speed, full_wall_floor) > minimums['speedup']),
    )
    commit_json(root/'full-throughput-audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    parser.add_argument('--formal', action='store_true', help='Run the registered formal paired evaluation')
    args = parser.parse_args()
    settings, _ = load_config(args.config, args.local)
    if args.formal:
        raise SystemExit(run_formal(settings))
    pair, _ = load_config(settings['pair_config'], args.local)
    root = Path(settings['output'])
    try:
        result = run(settings, pair)
    except Exception as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    commit_json(root/'terminal.json', dict(exit_code=0, passed=result['passed']))


if __name__ == '__main__':
    main()
