"""Freeze a device-utilization experiment on already registered evaluation blocks."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import threading

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.evaluation_blocks import read
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.native_seal_operations import run as operate
from tools.reports.prepare_tpo_windows import macros


def bound(pin):
    if file_hash(pin['path']) != pin['sha256']:
        raise ValueError('registered evaluation input changed')
    return read(Path(pin['path']))


def panel(plan, quotas):
    groups = defaultdict(list)
    for block in plan['blocks']:
        groups[(block['dataset'], block['arm'])].append(block)
    if set(quotas) != {dataset for dataset, _ in groups}:
        raise ValueError('a replay quota is required for every active dataset')
    chosen = set()
    for (dataset, _), rows in groups.items():
        n = quotas[dataset]
        if type(n) is not int or not 1 <= n <= len(rows):
            raise ValueError('invalid replay quota')
        # Original plan sorts by padded work. Include both ends and sample evenly.
        indices = [len(rows) // 2] if n == 1 else [i * (len(rows) - 1) // (n - 1) for i in range(n)]
        chosen.update(rows[i]['key'] for i in indices)
    return dict(plan, blocks=[b for b in plan['blocks'] if b['key'] in chosen])


def telemetry(raw):
    rows = {}
    device = None
    for line in raw.splitlines():
        identity = re.match(r'\|\s*(\d+)\s+\S+\s*\|\s*(?:OK|Warning|Critical)\s*\|', line)
        use = re.match(r'\|\s*\d+\s*\|\s*[0-9a-fA-F:.]+\s*\|\s*(\d+)\s.*?\s(\d+)\s*/\s*(\d+)\s*\|', line)
        if identity:
            device = int(identity[1])
        elif use and device is not None:
            core, used, capacity = map(int, use.groups())
            if device in rows or capacity <= 0 or not 0 <= used <= capacity or not 0 <= core <= 100:
                raise ValueError('invalid NPU telemetry')
            rows[device] = dict(aicore=core, hbm=100 * used / capacity)
            device = None
    return rows


def prepare(settings):
    base = Path(settings['baseline'])
    baseline = read(base/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    for key in ('block-worker-0.json', 'native-seal-container.json'):
        if file_hash(base/'entry'/key) != baseline['entry_pins'][key]:
            raise ValueError('baseline scoring entry changed')
    starter = read(base/'entry/block-worker-0.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    original = bound(settings['plan'])
    if (original['sha256'] != content_key(original['plan'])
            or original['plan']['namespace'] != starter['queue']['namespace']):
        raise ValueError('replay does not match the frozen baseline worker')
    plan = original['plan']
    paired = settings.get('paired_panels')
    if paired is None:
        selected = {'default': panel(plan, settings['blocks_per_panel'])}
    else:
        lanes = settings['devices_by_replica_count']
        if (len(lanes) != 2 or '1' not in lanes or not paired
                or any(len(devices) != len(paired) for devices in lanes.values())):
            raise ValueError('each comparison lane requires a matched panel')
        selected, seen, recordings = {}, set(), set()
        for i, keys in enumerate(paired):
            wanted = set(keys)
            rows = [b for b in plan['blocks'] if b['key'] in wanted]
            identities = {(b['dataset'], uid) for b in rows for uid in b['ids']}
            if (not wanted or len(wanted) != len(keys) or len(rows) != len(wanted)
                    or wanted & seen or identities & recordings
                    or len(identities) != sum(len(b['ids']) for b in rows)):
                raise ValueError('comparison panels must contain disjoint, complete recorded blocks')
            seen.update(wanted)
            recordings.update(identities)
            selected[str(i)] = dict(plan, blocks=rows)
        if len(recordings) != settings['recordings']:
            raise ValueError('comparison sample count differs from the frozen budget')
    work = settings['deployment']
    target = Path(work['package'])
    entry = target/'entry'
    entry.mkdir(parents=True)
    for path in (base/'entry').rglob('*.json'):
        rel = path.relative_to(base/'entry')
        if rel.parts[0] == 'source':
            continue
        if rel.parts[0] == 'official-supplement' or rel.name in ('parent.json', 'cards.json',
                            'native-seal-container.json'):
            (entry/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, entry/rel)
    snapshot = freeze(Path(__file__).resolve().parents[2], entry/'source', patterns=[
        'src/multimodal_jev/*.py', 'tools/reports/*.py', 'tools/jev/*.py',
        'tools/__init__.py', 'tools/reports/__init__.py',
    ], io_workers=starter['io_workers'])
    panels = {}
    for name, selection in selected.items():
        path = entry/('screen-panel.json' if name == 'default' else f'screen-panel-{name}.json')
        commit_json(path, dict(plan=selection, sha256=content_key(selection)))
        panels[name] = path
    archive_root = baseline['remote_root'] + '/sample-blocks'
    jobs = []
    for count, devices in settings['devices_by_replica_count'].items():
        if not count.isdecimal() or not devices:
            raise ValueError('replica groups must be explicit')
        from tools.reports.card_evaluation_replicas import replica_count
        replicas = replica_count(int(count))
        for i, device in enumerate(devices):
            if type(device) is not int or device < 0:
                raise ValueError('invalid physical device')
            name = f'replay-{count}-{device}'
            replay = panels['default' if paired is None else str(i)]
            spec = dict(starter, devices=[device], replicas_per_device=replicas,
                        source=dict(root='/jev-entry/source', sha256=snapshot['manifest_sha256']),
                        queue=dict(starter['queue'], root=f'/jev-work/{name}-blocks'),
                        output=f'/jev-work/{name}', result_path=f'/jev-work/{name}.json',
                        replica_replay=dict(plan='/jev-entry/'+replay.name, archive=archive_root,
                                            sha256=file_hash(replay)))
            if count in settings.get('single_validation_lanes', []):
                spec['eval_single_validation'] = True
            commit_json(entry/(name+'.json'), macros('NATIVE_SEAL_CONTAINER_CONFIG', spec))
            jobs.append(dict(name=name, lane=name, devices=[device], visible_ordinal='0',
                             collect=name+'.json', **work['container']))
    devices = [j['devices'][0] for j in jobs]
    if len(set(devices)) != len(devices) or '1' not in settings['devices_by_replica_count']:
        raise ValueError('screen needs disjoint physical devices and a one-replica control')
    with tarfile.open(target/'entry.tar', 'x') as package:
        package.add(entry, arcname='entry')
    deployment = {k: baseline[k] for k in ('host', 'image', 'lease', 'shared_root', 'driver_root',
                  'entrypoint', 'transfer_timeout')}
    deployment.update({k:work[k] for k in ('tag', 'remote_root', 'operations')})
    deployment.update(chains=[[jobs]], device_group_lease=True, wait_for_lease=True,
                      archive=str(target/'entry.tar'), entry_archive_sha256=file_hash(target/'entry.tar'),
                      entry_pins={str(path.relative_to(entry)):file_hash(path)
                                  for path in [*entry.glob('*.json'), entry/'source/FILES.json']},
                      read_only_mounts=[*baseline['read_only_mounts'], archive_root+':'+archive_root])
    if work.get('device_predecessors'):
        deployment['device_predecessors'] = work['device_predecessors']
    commit_json(target/'operations-local.json', macros('NATIVE_SEAL_OPERATIONS_CONFIG', deployment))
    return deployment


def screen(settings, deployment, *, observed_roles=None):
    root = Path(deployment['operations'])
    first, stopped = threading.Event(), threading.Event()

    def measure():
        while not first.wait(1):
            if stopped.is_set():
                return
        with (root/'telemetry.jsonl').open('x', buffering=1) as out:
            while True:
                try:
                    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o',
                          'ConnectTimeout=10', deployment['host'],
                          'date +%s.%N && npu-smi info'], timeout=15, text=True)
                    timestamp, _, body = raw.partition('\n')
                    row = dict(unix=float(timestamp), devices=telemetry(body))
                except (ValueError, subprocess.SubprocessError) as error:
                    row = dict(error=str(error))
                out.write(json.dumps(row) + '\n')
                if stopped.is_set():
                    break
                stopped.wait(settings['sampling_seconds'])

    with ThreadPoolExecutor(max_workers=1) as pool:
        sampling = pool.submit(measure)
        try:
            def started(role):
                if observed_roles is None or role in observed_roles:
                    first.set()

            return operate(deployment, on_started=started)
        finally:
            stopped.set()
            sampling.result()


def summarize(settings, deployment):
    root = Path(deployment['operations'])
    with (root/'telemetry.jsonl').open() as stream:
        samples = [json.loads(line) for line in stream]
    detail = defaultdict(list)
    for job in deployment['chains'][0][0]:
        name = job['name']
        count, device = (int(x) for x in name.rsplit('-', 2)[1:])
        path = root/(name+'-collected.json')
        if not path.exists():
            detail[count].append(dict(device=device, collected=False))
            continue
        receipt = read(path)
        launch = read(root/(name+'-launch.json'))
        inspected = read(root/(name+'-inspect.json'))
        if (inspected['Id'] != launch['container_id'] or inspected['State']['ExitCode']
                or receipt['devices'] != [device] or receipt['replicas_per_device'] != count):
            raise ValueError('inconsistent physical lane or exit identity')
        start = min(row['first_input_unix'] for row in receipt['blocks'])
        end = max(row['committed_unix'] for row in receipt['blocks'])
        # Include real device idle time between blocks; exclude model boot and post-run collection.
        observed = [row['devices'][str(device)] for row in samples
                    if str(device) in row.get('devices', {}) and start <= row['unix'] <= end]
        averages = {key: sum(row[key] for row in observed) / len(observed) if observed else None
                    for key in ('aicore', 'hbm')}
        state = inspected['State']
        dates = [datetime.fromisoformat(state[k].replace('Z', '+00:00')) for k in ('StartedAt','FinishedAt')]
        entry = Path(deployment['archive']).parent/'entry'
        spec_path = entry/(name+'.json')
        spec_pin = dict(path=spec_path, sha256=deployment['entry_pins'][spec_path.name])
        replay = bound(spec_pin)['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['replica_replay']
        p = bound(dict(path=entry/Path(replay['plan']).name, sha256=replay['sha256']))
        if (receipt['plan_sha256'] != p['sha256']
                or receipt['pairs'] != sum(sum(b['widths']) for b in p['plan']['blocks'])):
            raise ValueError('lane output differs from the frozen sample')
        detail[count].append(dict(device=device, collected=True, seconds=end-start,
             scoring_start=start, scoring_end=end, container_started=dates[0].timestamp(),
             container_ended=dates[1].timestamp(), recordings=sum(len(b['ids']) for b in p['plan']['blocks']),
             container_wall_seconds=(dates[1]-dates[0]).total_seconds(), **averages,
             pairs=receipt['pairs'], readings=len(observed), differences=receipt['probability_differences'],
             panel_sha256=receipt['plan_sha256']))
    base = detail[1]
    comparison = {}
    usable = lambda row: row['collected'] and not row['differences'] and row['readings'] >= settings['min_samples']
    paired = settings.get('paired_panels') is not None

    def throughput(rows, key):
        if paired:
            start, end = (('scoring_start', 'scoring_end') if key == 'seconds'
                          else ('container_started', 'container_ended'))
            duration = max(r[end] for r in rows) - min(r[start] for r in rows)
        else:
            duration = sum(r[key] for r in rows)
        return sum(r['pairs'] for r in rows) / duration
    if all(map(usable, base)):
        for count, rows in detail.items():
            if (not all(map(usable, rows)) or (paired and
                        [r['panel_sha256'] for r in rows] != [r['panel_sha256'] for r in base])
                    or (not paired and {r['panel_sha256'] for r in base+rows} != {base[0]['panel_sha256']})):
                continue
            averages = {k:sum(r[k] for r in rows)/len(rows) for k in ('aicore', 'hbm')}
            speedup = throughput(rows, 'seconds') / throughput(base, 'seconds')
            wall = throughput(rows, 'container_wall_seconds') / throughput(base, 'container_wall_seconds')
            comparison[count] = dict(**averages, speedup=speedup, container_speedup=wall, passed=(
                averages['aicore'] >= settings['min_aicore'] and averages['hbm'] >= settings['min_hbm']
                and min(speedup, wall) > settings['min_speedup']))
    result = dict(scope='exact registered scoring subset, single physical card per lane',
                  lanes=detail, comparison=comparison, passed=any(row['passed'] for row in comparison.values()))
    if paired:
        result.update(scope='matched fixed samples, physical cards per arm', recordings=settings['recordings'],
                      pairs=sum(r['pairs'] for r in base) if all(map(usable, base)) else None,
                      deployment_sha256=file_hash(Path(deployment['archive']).parent/'operations-local.json'))
    commit_json(root/'screen-audit.json', result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('config')
    cli.add_argument('local')
    cli.add_argument('phase', choices=('prepare', 'run', 'audit'))
    args = cli.parse_args()
    settings, _ = load_config(args.config, args.local)
    path = Path(settings['deployment']['package'])/'operations-local.json'
    if args.phase == 'prepare':
        prepare(settings)
    else:
        deployment = read(path)['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
        if args.phase == 'run':
            screen(settings, deployment)
        print(json.dumps(summarize(settings, deployment)))


if __name__ == '__main__':
    main()
