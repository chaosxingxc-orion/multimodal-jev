"""Dispatch a bounded lane of an existing, frozen full TRAIN score."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from multimodal_jev.card_scoring_settings import scoring_settings
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze, verify
from tools.reports.card_scoring_benchmark import pinned, scoring_lineage, scoring_offsets
from tools.reports.native_seal_operations import run


def collect(host, remote_root, destination, names):
    destination.mkdir()
    command = shlex.join(['tar', '-cf', '-', '-C', remote_root, *names])
    with subprocess.Popen(['ssh', '-o', 'BatchMode=yes', host, command], stdout=subprocess.PIPE) as ssh:
        with tarfile.open(mode='r|', fileobj=ssh.stdout) as tar:
            tar.extractall(destination, filter='data')
        if ssh.wait():
            raise ValueError('scoring evidence transfer failed')


def prepare(config):
    parent = config['parent']
    upstream = Path(parent['entry'])
    spec = pinned({'path': upstream/'score.json', 'sha256': parent['sha256']})[
        'macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    checked = verify(upstream/'source', manifest_sha256=spec['source']['sha256'],
                     io_workers=spec['io_workers'])
    deployment = config['deployment'].copy()
    if deployment['image'] != spec['formal']['image'] or deployment['idle_devices'] != spec['devices']:
        raise ValueError('lane must preserve the frozen image and rank layout')
    deployment['read_only_mounts'] = [f"{config['policy']}:{spec['policy']}",
                                      f"{config['synthetic_cache']}:{spec['synthetic_cache']}"]
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', deployment['host']]
    remote = shlex.join(['sha256sum', str(Path(config['policy'])/'MANIFEST.json')])
    if subprocess.check_output([*ssh, remote], text=True, timeout=25).split()[0] != spec['policy_sha256']:
        raise ValueError('target policy does not match the frozen checkpoint')

    root = Path(config['output'])
    entry = root/'entry'
    entry.mkdir(parents=True)
    pins = ('full', 'panel', 'training', 'cards', *(f'full-{i}' for i in range(len(spec['devices']))))
    source_pins = {Path(item['path']).stem: item['sha256'] for item in
                   (spec['full'], spec['panel'], spec['training'], spec['cards'], *spec['full_shards'])}
    for name in pins:
        path = upstream/(name+'.json')
        if file_hash(path) != source_pins[name]:
            raise ValueError(f'frozen input differs: {name}')
        shutil.copyfile(path, entry/path.name)
    shutil.copyfile(upstream/'native-seal-container.json', entry/'native-seal-container.json')
    formal = spec['formal']
    for key in ('resume_from', 'resume_config', 'block_range'):
        formal.pop(key, None)
    formal['block_range'] = config['block_range']
    full = pinned({'path': entry/'full.json', 'sha256': spec['full']['sha256']})
    for ordinal in range(len(spec['devices'])):
        scoring_offsets(spec, len(full['rows'][ordinal::len(spec['devices'])]))

    repo = Path(__file__).resolve().parents[2]
    snapshot = freeze(repo, entry/'source', patterns=config['source_patterns'], io_workers=spec['io_workers'])
    files = verify(entry/'source', manifest_sha256=snapshot['manifest_sha256'], io_workers=spec['io_workers'])
    allowed = {'tools/reports/card_scoring_benchmark.py', 'tools/reports/card_scoring_shards.py',
               'tools/reports/audit_card_scoring_benchmark.py', 'tools/reports/experiment_package.py'}
    if {name for name in files.keys() | checked.keys() if files.get(name) != checked.get(name)} - allowed:
        raise ValueError('lane source changes extend beyond scheduling and collection')
    spec['source']['sha256'] = snapshot['manifest_sha256']
    commit_json(entry/'score.json', dict(schema='jev-local-macros-v1',
                                         macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=spec)))
    archive = root/'entry.tar'
    with tarfile.open(archive, 'x') as tar:
        tar.add(entry, arcname='entry')
    names = ['score.json', 'source/FILES.json', 'native-seal-container.json', *(f'{n}.json' for n in pins)]
    deployment.update(archive=str(archive), entry_archive_sha256=file_hash(archive),
                      entry_pins={name: file_hash(entry/name) for name in names},
                      device_group_lease=True, wait_for_lease=False)
    if [job['devices'] for chain in deployment['chains'] for group in chain for job in group] != [spec['devices']]:
        raise ValueError('exactly one scoring job with the frozen devices required')
    commit_json(root/'operations.json', deployment)
    return deployment


def execute(template, local):
    config, _ = load_config(template, local)
    deployment = prepare(config)
    if run(deployment):
        raise ValueError('scoring lane did not exit successfully')
    root = Path(config['output'])
    collect(deployment['host'], deployment['remote_root'], root/'collected', ['score'])
    from tools.reports.audit_card_scoring_benchmark import progress
    report = progress(root/'collected/score', root/'entry')
    operations = Path(deployment['operations'])
    state = json.loads((operations/'score-inspect.json').read_bytes())
    cid = json.loads((operations/'score-launch.json').read_bytes())['container_id']
    spec = json.loads((root/'entry/score.json').read_bytes())['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    size = spec['formal']['recordings_per_block']
    workers = report['workers']
    if (state['Id'] != cid or state['State']['Status'] != 'exited'
            or state['State']['ExitCode'] != 0 or state['State']['OOMKilled']
            or any(worker['blocks'] != len(scoring_offsets(spec, worker['total_recordings']))
                   or worker['recordings'] != sum(min(size, worker['total_recordings']-offset)
                                                for offset in scoring_offsets(spec, worker['total_recordings']))
                   for worker in workers)):
        raise ValueError('scoring lane did not commit exactly its assignment')
    report.update(passed=True, container_exited=True, container_id=cid, block_range=config['block_range'])
    commit_json(root/'audit.json', report)
    return report


def check_partition(source, lane_configs, population, head):
    if type(head) is not int or head < 0 or not lane_configs:
        raise ValueError('nonnegative head size and at least one scoring lane required')
    replicas, size = len(source['devices']), source['formal']['recordings_per_block']
    for ordinal in range(replicas):
        rows = len(range(ordinal, population, replicas))
        offsets = list(range(0, rows, size))
        assigned = [offset for spec in lane_configs for offset in scoring_offsets(spec, rows)]
        if head >= len(offsets) or assigned != offsets[head:]:
            raise ValueError('scoring lanes must cover the assigned population exactly once in block order')


def merge(template, local):
    """Audit the union of completed scoring lanes and any parent prefix."""
    from concurrent.futures import ThreadPoolExecutor

    from multimodal_jev.context_execution import map_cpu
    from tools.reports.audit_card_scoring_benchmark import compare, predictions, progress, read, validate_block
    from tools.reports.context_macro_probe_operations import inspect_runtime
    from tools.reports.native_seal_operations import verify_memory

    c, _ = load_config(template, local)
    root = Path(c['output'])
    root.mkdir(parents=True)
    parent = c['parent']
    entry = Path(parent['entry'])
    source = pinned(dict(path=entry/'score.json', sha256=parent['sha256']))['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    full = pinned(dict(path=entry/'full.json', sha256=source['full']['sha256']))
    panel = pinned(dict(path=entry/'panel.json', sha256=source['panel']['sha256']))
    training = pinned(dict(path=entry/'training.json', sha256=source['training']['sha256']))
    replicas = len(source['devices'])
    size = source['formal']['recordings_per_block']
    width, = source['widths']
    lanes = c['lanes']
    lane_configs = [read(Path(lane['package'])/'entry/score.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
                    for lane in lanes]
    if any(any(spec[k] != source[k] for k in ('full', 'full_shards', 'panel', 'training', 'cards',
             'policy_sha256', 'widths', 'devices', 'io_workers', 'prefetch_audio', 'streaming'))
           or any(spec['formal'][k] != source['formal'][k] for k in ('checkpoint_sha256', 'policy_id',
                  'image', 'recordings_per_block', 'updates'))
           for spec in lane_configs):
        raise ValueError('cannot merge differing frozen policies or scoring populations')
    head = c['head_blocks']
    check_partition(source, lane_configs, len(full['rows']), head)

    def wait_lane(lane, spec):
        path = Path(lane['package'])
        ops = Path(lane['operations'])
        while not (path/'audit.json').is_file():
            if (ops/'terminal.json').is_file() and read(ops/'terminal.json')['exit_code'] != 0:
                raise ValueError(f'scoring lane failed: {ops}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'no audited lane result: {path}')
            time.sleep(c['status_interval_seconds'])
        proof = read(path/'audit.json')
        if (proof['passed'] is not True or proof['container_exited'] is not True
                or proof['block_range'] != spec['formal']['block_range']
                or proof['container_id'] != read(ops/'score-launch.json')['container_id']
                or read(ops/'score-admission.json')['entry_pins']['score.json'] != file_hash(path/'entry/score.json')):
            raise ValueError('lane receipt differs')
        raw = path/'collected/score'
        observed = progress(raw, path/'entry')
        if any(proof[k] != value for k, value in observed.items()):
            raise ValueError('lane receipt differs from collected evidence')
        return raw, observed

    deadline = time.monotonic()+c['timeout_seconds']
    if c['status_interval_seconds'] < 900 or c['timeout_seconds'] <= 0:
        raise ValueError('positive timeout and repository observation cadence required')
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        results = [p.result() for p in [pool.submit(wait_lane, lane, spec)
                   for lane, spec in zip(lanes, lane_configs, strict=True)]]
    scoring = [(spec, raw, None) for spec, (raw, _) in
               zip(lane_configs, results, strict=True)]
    observed_parent = {}
    if head:
        handle = pinned(dict(path=parent['launch'], sha256=parent['launch_sha256']))
        host, remote_root, cid = (handle[k] for k in ('host', 'remote_root', 'container_id'))
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host]

        def remote(*args):
            return subprocess.check_output([*ssh, shlex.join(args)], text=True, timeout=45)

        def parent_state():
            state, = json.loads(remote('docker', 'inspect', cid))
            inspect_runtime(state, handle)
            verify_memory(state, handle['argv'])
            return state['State']

        names = [(f'score/worker-{i}/block-{j:06d}.json')
                 for i in range(replicas) for j in range(head)]
        while True:
            state = parent_state()
            existing = set(remote('find', remote_root+'/score', '-maxdepth', '2', '-name',
                                  'block-*.json', '-printf', '%P\\n').splitlines())
            if all(name.removeprefix('score/') in existing for name in names):
                break
            if not state['Running'] or time.monotonic() >= deadline:
                raise ValueError('parent did not reach the frozen head boundary')
            time.sleep(c['status_interval_seconds'])

        for i in range(replicas):
            names += [f'score/fsdp-{i}/reference.json', *(f'score/worker-{i}/{name}' for name in (
                'lineage.json', 'panel-ready.json', 'baseline.json', f'batch-{width}.json'))]
        collect(host, remote_root, root/'parent', ['score/admission.json', *names])
        first = root/'parent/score'
        admitted = progress(first, entry)
        if not admitted['admitted'] or admitted['blocks'] != replicas*head:
            raise ValueError('parent prefix is incomplete')

        scoring.insert(0, (source, first, range(head)))
        reference_raw, comparisons = first, results
    else:
        reference_raw, comparisons = results[0][0], results[1:]

    def panel_probs(raw):
        output = {}
        for i in range(replicas):
            expected = [row['id'] for row in panel['rows'][i::replicas]]
            part, _ = predictions(read(raw/f'worker-{i}/batch-{width}.json'), expected, full['labels'], width)
            output.update(part)
        return output

    reference = panel_probs(reference_raw)
    parity = [compare(panel, reference, panel_probs(raw), source['probability_atol'])
              for raw, _ in comparisons]
    if any(r[k] for r in parity for k in ('probabilities_over_atol', 'top1_disagreements', 'class_rank_disagreements')):
        raise ValueError(f'cross-host score parity failed: {parity}')
    lookup = {}
    for spec, raw, prefix in scoring:
        for i in range(replicas):
            ids = [row['id'] for row in full['rows'][i::replicas]]
            offsets = [j*size for j in prefix] if prefix is not None else scoring_offsets(spec, len(ids))
            for offset in offsets:
                block = raw/f'worker-{i}'/f'block-{offset//size:06d}.json'
                key = (i, offset)
                if key in lookup:
                    raise ValueError('duplicate scoring block assignment')
                lookup[key] = (block, scoring_lineage(spec, i), ids[offset:offset+size], full['labels'], width,
                               (spec['io_workers'], spec['prefetch_audio']))
    tasks = [lookup[i, offset] for i in range(replicas)
             for offset in range(0, len(full['rows'][i::replicas]), size)]
    values, _ = map_cpu(validate_block, tasks, workers=source['cpu_workers'],
                        min_parallel=source['audit_min_parallel_blocks'])
    labels = full['labels']
    index = {label: i for i, label in enumerate(labels)}
    gold = {row['id']: index[row['intent']] for row in full['rows']}
    counts = [0]*len(labels)
    totals = [[0.]*len(labels) for _ in labels]
    seen = set()
    for task, probs in zip(tasks, values, strict=True):
        for uid, row in zip(task[2], probs, strict=True):
            if uid in seen:
                raise ValueError('duplicate TRAIN recording')
            seen.add(uid)
            i = gold[uid]
            counts[i] += 1
            for j, p in enumerate(row):
                totals[i][j] += p
    if seen != gold.keys():
        raise ValueError('missing TRAIN recordings')
    import torch
    from multimodal_jev.intent_triplet_training import confusion_distribution
    distribution = confusion_distribution(labels, torch.tensor(totals, dtype=torch.float64),
                                           torch.tensor(counts), scoring_settings(training, source.get('scoring_reference'))['zipf_exponent'])
    formal = source['formal']
    pairs = len(seen)*len(labels)
    path = root/(f"confusion-step-{formal['updates']:04d}.json" if not head else 'confusion-epoch-1.json')
    commit_json(path, dict(labels=labels, counts=counts, probability_sums=totals,
                           distribution=distribution, pairs=pairs, **({'epoch': 0} if head else {}),
                           checkpoint_sha256=formal['checkpoint_sha256'],
                           policy_id=formal['policy_id'], full_sha256=source['full']['sha256']))
    if head:
        state = parent_state()
        if state['Running']:
            commit_json(root/'parent-stop-intent.json', dict(container_id=cid, reason='all TRAIN scores audited'))
            remote('docker', 'stop', '--time', str(c['stop_grace_seconds']), cid)
            state = parent_state()
        if state['Status'] != 'exited' or state['Pid'] != 0 or state['OOMKilled']:
            raise ValueError('parent container release not observed')
        observed_parent = dict(parent_container=cid, parent_exit=state['ExitCode'])
    report = dict(passed=True, container_exited=True, updates=formal['updates'],
                  recordings=len(seen), pairs=pairs, checkpoint_sha256=formal['checkpoint_sha256'],
                  confusion_sha256=file_hash(path), **observed_parent,
                  lanes=[proof for _, proof in results], cross_host_parity=parity)
    commit_json(root/'audit.json', report)
    return report
