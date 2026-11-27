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
    selected = panel(original['plan'], settings['blocks_per_panel'])
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
    commit_json(entry/'screen-panel.json', dict(plan=selected, sha256=content_key(selected)))
    archive_root = baseline['remote_root'] + '/sample-blocks'
    jobs = []
    for count, devices in settings['devices_by_replica_count'].items():
        if not count.isdecimal() or not devices:
            raise ValueError('replica groups must be explicit')
        from tools.reports.card_evaluation_replicas import replica_count
        replicas = replica_count(int(count))
        for device in devices:
            if type(device) is not int or device < 0:
                raise ValueError('invalid physical device')
            name = f'replay-{count}-{device}'
            spec = dict(starter, devices=[device], replicas_per_device=replicas,
                        source=dict(root='/jev-entry/source', sha256=snapshot['manifest_sha256']),
                        queue=dict(starter['queue'], root=f'/jev-work/{name}-blocks'),
                        output=f'/jev-work/{name}', result_path=f'/jev-work/{name}.json',
                        replica_replay=dict(plan='/jev-entry/screen-panel.json', archive=archive_root,
                                            sha256=file_hash(entry/'screen-panel.json')))
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


def screen(settings, deployment):
    root = Path(deployment['operations'])
    first, stopped = threading.Event(), threading.Event()

    def measure():
        while not first.wait(1):
            if stopped.is_set():
                return
        with (root/'telemetry.jsonl').open('x', buffering=1) as out:
            while not stopped.is_set():
                try:
                    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o',
                          'ConnectTimeout=10', deployment['host'],
                          'date +%s.%N && npu-smi info'], timeout=15, text=True)
                    timestamp, _, body = raw.partition('\n')
                    row = dict(unix=float(timestamp), devices=telemetry(body))
                except (ValueError, subprocess.SubprocessError) as error:
                    row = dict(error=str(error))
                out.write(json.dumps(row) + '\n')
                stopped.wait(settings['sampling_seconds'])

    with ThreadPoolExecutor(max_workers=1) as pool:
        sampling = pool.submit(measure)
        try:
            operate(deployment, on_started=lambda _: first.set())
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
        detail[count].append(dict(device=device, collected=True, seconds=end-start,
             container_wall_seconds=(dates[1]-dates[0]).total_seconds(), **averages,
             pairs=receipt['pairs'], readings=len(observed), differences=receipt['probability_differences'],
             panel_sha256=receipt['plan_sha256']))
    base = detail[1]
    comparison = {}
    usable = lambda row: row['collected'] and not row['differences'] and row['readings'] >= settings['min_samples']
    def throughput(rows, key):
        return sum(r['pairs'] for r in rows) / sum(r[key] for r in rows)
    if all(map(usable, base)):
        for count, rows in detail.items():
            if not all(map(usable, rows)) or {r['panel_sha256'] for r in base+rows} != {base[0]['panel_sha256']}:
                continue
            averages = {k:sum(r[k] for r in rows)/len(rows) for k in ('aicore', 'hbm')}
            speedup = throughput(rows, 'seconds') / throughput(base, 'seconds')
            wall = throughput(rows, 'container_wall_seconds') / throughput(base, 'container_wall_seconds')
            comparison[count] = dict(**averages, speedup=speedup, container_speedup=wall, passed=(
                averages['aicore'] >= settings['min_aicore'] and averages['hbm'] >= settings['min_hbm']
                and min(speedup, wall) > settings['min_speedup']))
    result = dict(scope='exact registered scoring subset, single physical card per lane',
                  lanes=detail, comparison=comparison, passed=any(row['passed'] for row in comparison.values()))
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
