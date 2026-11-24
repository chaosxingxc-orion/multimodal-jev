"""Seal, run and collect equal-exposure Stage1 batch comparisons."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.stage1_batch_probe import select_batches
from multimodal_jev.stage1_mixed_schedule import schedule


def read(path):
    return json.loads(path.read_bytes())


def pinned(raw, digest):
    from hashlib import sha256
    if sha256(raw).hexdigest() != digest:
        raise ValueError('comparison input differs from registered identity')
    return raw


def prepare(c):
    from multimodal_jev.source_snapshot import freeze, verify

    parent = Path(c['package'])
    old = read(parent/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    archive = parent/'entry.tar'
    checkpointing = c.get('activation_checkpointing')
    prior_comparison = None
    if checkpointing is not None:
        prior_comparison = Path(c['baseline_comparison'])
    if old['entry_archive_sha256'] != c['archive_sha256'] or file_hash(archive) != c['archive_sha256']:
        raise ValueError('comparison parent archive differs')
    deployment = c['deployment'].copy()
    if deployment['host'] != old['host']:
        raise ValueError('comparison must reuse the same host and prepared caches')
    if prior_comparison is not None:
        prior_entry = read(prior_comparison/'operations.json')
        if (not deployment.get('predecessor')
                or Path(deployment['predecessor']['terminal']) != Path(prior_entry['operations'])/'terminal.json'
                or deployment['host'] != prior_entry['host']
                or deployment['lease'] != prior_entry['lease']
                or deployment['idle_devices'] != prior_entry['idle_devices']):
            raise ValueError('recompute probe requires the exact baseline comparison and lease')
    upstream = Path(old['remote_root'])
    devices = deployment['idle_devices']
    widths = c['widths']
    if len(widths) != 2 or widths[1] != 2 * widths[0] or any(
            type(w) is not int or w < 1 or w & (w - 1) for w in widths):
        raise ValueError('two adjacent powers of two are required')
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('comparison requires a unique, nonempty device allocation')
    if not set(devices).isdisjoint(old['idle_devices']):
        # A retired run can lend its devices, but its successful preparation is still reusable.
        role = c.get('retired_parent_role')
        if not role or deployment['lease'] != old['lease']:
            raise ValueError('overlapping comparison requires the retired parent and its lease')
        ops = Path(old['operations'])
        launch = read(ops/f'{role}-launch.json')
        old_result = read(ops/f'{role}-inspect.json')
        cid = launch['container_id']
        cmd = shlex.join(['docker', 'inspect', cid])
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', deployment['host']]
        live, = json.loads(subprocess.check_output([*ssh, cmd], timeout=25))
        state = live['State']
        if (live['Id'] != cid or old_result['Id'] != cid or live['State'] != old_result['State']
                or live['Name'] != '/' + old['tag'] + '-' + role or live['Image'] != old['image']
                or state['Status'] != 'exited' or state['Pid'] != 0
                or live['HostConfig']['RestartPolicy']['Name'] not in ('', 'no')):
            raise ValueError('parent container retirement does not match the launch evidence')
    predecessors = deployment.get('device_predecessors', [])
    if not {d for job in predecessors for d in job['devices']} <= set(devices):
        raise ValueError('comparison may only wait for owners of its assigned devices')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', deployment['host']]

    def fetch(relative, digest):
        cmd = shlex.join(['cat', str(upstream/relative)])
        return pinned(subprocess.check_output([*ssh, cmd], timeout=25), digest)

    formal = json.loads(fetch('formal/config.json', c['formal_config_sha256']))
    bootstrap = fetch('formal/run/confusion-init.json', c['confusion_sha256'])
    prep = json.loads(fetch('prepare/receipt.json', c['prepare_receipt_sha256']))
    if (formal['source_policy_id'] != prep['policy_id']
            or formal['source_policy_id'] != json.loads(bootstrap)['policy_id']
            or formal['single_card']['sha256'] != prep['single_sha256']
            or formal['physical_devices'] != old['idle_devices']
            or formal['microbatch'] != widths[0]):
        raise ValueError('initial policy, single-card prompts or parent topology differs')

    source = parent/'entry'
    output = Path(c['output'])
    entry = output/'entry'
    entry.mkdir(parents=True)
    import shutil
    for name in ('admission.json', 'synthetic.json', 'mixed-task.json', 'design.json',
                 'card-plan.json', 'native-seal-container.json'):
        path = source/name
        if file_hash(path) != old['entry_pins'][name]:
            raise ValueError('parent entry differs: ' + name)
        shutil.copyfile(path, entry/name)
    prior_sources = verify(source/'source', manifest_sha256=old['entry_pins']['source/FILES.json'],
                           io_workers=c['io_workers'])
    repo = Path(__file__).resolve().parents[2]
    script = freeze(repo, entry/'source', patterns=c['source_patterns'], io_workers=c['io_workers'])
    current_sources = read(entry/'source/FILES.json')
    changed = {f for f in current_sources.keys() | prior_sources.keys()
               if current_sources.get(f) != prior_sources.get(f)}
    allowed = {'tools/jev/train_card_stage1.py', 'tools/reports/card_stage1_window.py',
               'tools/reports/card_capacity_sweep.py', 'tools/reports/card_batch_comparison.py',
               'src/multimodal_jev/stage1_batch_probe.py'}
    if prior_comparison is not None:
        allowed.add('tools/reports/experiment_package.py')
    if changed - allowed:
        raise ValueError(f'comparison snapshot changes unrelated code: {changed - allowed}')

    base_release = Path(formal['release'])
    release_manifest = json.loads(pinned((base_release/'MANIFEST.json').read_bytes(),
                                         formal['mixed']['real_release_sha256']))
    train = pinned((base_release/'learning.jsonl').read_bytes(), release_manifest['files']['learning.jsonl'])
    synthetic = read(entry/'synthetic.json')
    population = {r['id']: r for r in [*(json.loads(line) for line in train.splitlines()), *synthetic]}
    if len(population) != formal['batch_config']['source_queries']:
        raise ValueError('comparison TRAIN population differs')
    _, proof = schedule(population, formal['batch_config'], formal['mixed']['seed'])
    if (file_hash(source/'mixed-schedule.json') != formal['schedule_sha256']
            or read(source/'mixed-schedule.json') != proof):
        raise ValueError('comparison cannot reproduce parent TRAIN order')

    (entry/'bootstrap.json').write_bytes(bootstrap)
    warmup, ordinary, pressure = c['spans']
    if [s['name'] for s in c['spans']] != ['warmup', 'ordinary', 'long_audio']:
        raise ValueError('declare one warmup, ordinary and long audio span each')
    if min(s['recordings'] for s in (warmup, ordinary, pressure)) < 1:
        raise ValueError('empty comparison span')
    runtime = {'root': '/jev-entry/source', 'sha256': script['manifest_sha256']}
    freeze_identity = None
    names = []
    jobs = []
    for width in widths:
        if prior_comparison is not None and width == widths[0]:
            continue
        role = f'micro-{width}' + ('-recompute' if prior_comparison is not None else '')
        config = deepcopy(formal)
        config['batch_config'].update(training_devices=len(devices), microbatch=width,
                                      per_dp_replica_batch=width)
        plan = derive(config['batch_config'])
        global_batch = plan['global_batch']
        steps = sum(s['recordings'] for s in c['spans']) // global_batch
        if (plan['accumulation'] != 1 or not 0 < steps <= config['batch_config']['efficiency_steps']
                or warmup['recordings'] // global_batch < config['lr_schedule']['warmup_updates']):
            raise ValueError('invalid bounded comparison batch plan')
        batches, manifest = schedule(population, config['batch_config'], config['mixed']['seed'])
        prefix = select_batches(batches, global_batch, {'spans': c['spans']}, steps)
        spans = []
        cursor = 0
        for span in c['spans']:
            count = span['recordings'] // global_batch
            ids = [slot['id'] for b in prefix[cursor:cursor + count] for slot in b['slots']]
            spans.append(dict(span, anchor_sha256=content_hash(ids), steps=count))
            cursor += count
        identity = [s['anchor_sha256'] for s in spans]
        if freeze_identity is not None and identity != freeze_identity:
            raise ValueError('comparison widths do not expose identical anchors in the same order')
        freeze_identity = identity
        witness = c['long_audio_anchor']
        offset = pressure['offset'] // global_batch
        count = pressure['recordings'] // global_batch
        if witness not in {s['id'] for b in batches[offset:offset + count] for s in b['slots']}:
            raise ValueError('long audio witness missing from the fixed pressure span')
        manifest_name = f'micro-{width}-schedule.json'
        commit_json(entry/manifest_name, manifest)
        config['mixed']['schedule_manifest'] = '/jev-entry/' + manifest_name
        config.update(physical_devices=devices, global_batch=global_batch, microbatch=width,
                      schedule_sha256=file_hash(entry/manifest_name), engineering_prefix=True,
                      training_dispatch_allowed=False, mine_after_window=False, start_update=0,
                      stop_update=steps, checkpoint_interval=steps, parity_passes=0,
                      reference_endpoint=plan['total_updates'], evaluation_steps=[],
                      bootstrap_input=dict(path='/jev-entry/bootstrap.json', sha256=c['confusion_sha256']),
                      batch_probe=dict(spans=spans))
        config['single_card']['path'] = '/jev-parent-prepare/single.json'
        if prior_comparison is not None:
            base_name = f'micro-{width}-training.json'
            base_raw = pinned((prior_comparison/'entry'/base_name).read_bytes(), prior_entry['entry_pins'][base_name])
            baseline_config = json.loads(base_raw)
            differences = sorted(k for k in config.keys() | baseline_config.keys()
                                 if config.get(k) != baseline_config.get(k) or
                                 (k in config) != (k in baseline_config))
            if differences:
                raise ValueError('recompute probe differs from the frozen baseline: ' + ', '.join(differences))
            config['activation_checkpointing'] = checkpointing
        run_name = f'{role}-training.json'
        commit_json(entry/run_name, config)
        leaf = dict(phase='card_stage1', role=role, work='/jev-work', source=runtime,
                    io_workers=c['io_workers'], batch_probe=dict(path='/jev-entry/' + run_name,
                                                                sha256=file_hash(entry/run_name)),
                    master_port=c['master_port'])
        commit_json(entry/f'{role}.json', dict(schema='jev-local-macros-v1',
                                               macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=leaf)))
        names.extend([manifest_name, run_name, f'{role}.json'])
        jobs.append([dict(name=role, devices=devices, collect=f'{role}/receipt.json', **c['resources'])])
    new_archive = output/'entry.tar'
    with tarfile.open(new_archive, 'x') as tar:
        tar.add(entry, arcname='entry')
    names.extend(['source/FILES.json', 'admission.json', 'synthetic.json', 'mixed-task.json',
                  'design.json', 'card-plan.json', 'bootstrap.json', 'native-seal-container.json'])
    deployment.update(image=old['image'], driver_root=old['driver_root'], shared_root=old['shared_root'],
                      entrypoint=old['entrypoint'], transfer_timeout=old['transfer_timeout'],
                      read_only_mounts=[*old['read_only_mounts'], f'{upstream}/prepare:/jev-parent-prepare'],
                      archive=str(new_archive),
                      entry_archive_sha256=file_hash(new_archive),
                      entry_pins={name: file_hash(entry/name) for name in names},
                      chains=[jobs], device_group_lease=True, wait_for_lease=True)
    commit_json(output/'operations.json', deployment)
    return deployment


def summarize(root, config, outcome):
    width = config['global_batch']
    ranks = len(config['physical_devices'])
    cursor = 0
    spans = []
    for span in config['batch_probe']['spans']:
        rows = []
        for step in range(cursor + 1, cursor + 1 + span['steps']):
            paths = [root/f'rank-{rank}'/f'update-{step:04d}.json' for rank in range(ranks)]
            if not all(p.is_file() for p in paths):
                break
            records = [read(path) for path in paths]
            if any(r['update'] != step or r['real_queries'] != width or r['checkpoint']
                   or not r['optimizer_audit'] or not all(math.isfinite(v) for v in r['metrics'].values())
                   for r in records):
                raise ValueError(f'invalid optimizer evidence at update {step}')
            rows.append(dict(update=step, seconds=max(r['total_seconds'] for r in records),
                             actor_seconds=max(r['actor_seconds'] for r in records),
                             allocated_bytes=max(r['peak_allocated'] for r in records),
                             reserved_bytes=max(r['peak_reserved'] for r in records)))
        cursor += span['steps']
        seconds = sum(row['seconds'] for row in rows)
        spans.append(dict(span, completed_steps=len(rows), recorded_queries=len(rows)*width,
                          time_seconds=seconds, queries_per_second=len(rows)*width/seconds if seconds else None,
                          peak_allocated_bytes=max((r['allocated_bytes'] for r in rows), default=None),
                          peak_reserved_bytes=max((r['reserved_bytes'] for r in rows), default=None),
                          updates=rows))
    if outcome == 'completed' and any(s['completed_steps'] != s['steps'] for s in spans):
        raise ValueError('missing optimizer updates in completed comparison')
    return dict(outcome=outcome, microbatch=config['microbatch'], global_batch=width,
                devices=config['physical_devices'], accumulation=1, spans=spans,
                activation_checkpointing=config.get('activation_checkpointing'))


def run_probe(c):
    role = c['role']
    root = Path(c['work'])/role
    root.mkdir()
    pin = c['batch_probe']
    path = Path(pin['path'])
    config = json.loads(pinned(path.read_bytes(), pin['sha256']))
    output = root/'run'
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--node-rank=0',
               '--master-addr=127.0.0.1', '--master-port=' + str(c['master_port']), '--max-restarts=0',
               '--nproc-per-node=' + str(len(config['physical_devices'])),
               str(Path(c['source']['root'])/'tools/jev/train_card_stage1.py'),
               '--config', str(path), '--output', str(output)]
    log = root/'trainer.log'
    with log.open('x') as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code:
        raw = log.read_text(errors='replace')
        matches = re.findall(r'^.*(?:NPU out of memory|NPUOutOfMemoryError).*$',
                             raw, flags=re.IGNORECASE | re.MULTILINE)
        if not matches:
            print(raw[-16000:], flush=True)
            raise RuntimeError(f'comparison trainer exited with code {code}')
        outcome = 'npu_out_of_memory'
    else:
        result = read(output/'result.json')
        if result['passed'] is not True or result['stop_update'] != config['stop_update']:
            raise ValueError('comparison terminal update differs')
        matches, outcome = [], 'completed'
    result = summarize(output, config, outcome)
    result.update(trainer_exit_code=code, allocator_errors=list(dict.fromkeys(matches)),
                  trainer_log_sha256=file_hash(log), config_sha256=pin['sha256'])
    commit_json(root/'receipt.json', result)
    return result


def execute(c):
    from tools.reports.native_seal_operations import run

    deployment = prepare(c)
    print(f"Sealed batch comparison in {c['output']}", flush=True)
    if run(deployment):
        raise ValueError('batch comparison operations did not exit successfully')
    ops = Path(deployment['operations'])
    baseline = c.get('baseline_comparison')
    arms = read(Path(baseline)/'results.json')['arms'] if baseline else []
    for group in deployment['chains'][0]:
        role = group[0]['name']
        sample = read(ops/f'{role}-collected.json')
        launch = read(ops/f'{role}-launch.json')
        terminal = read(ops/f'{role}-inspect.json')['State']
        if terminal['Status'] != 'exited' or terminal['ExitCode'] != 0 or terminal['OOMKilled']:
            raise ValueError(f'comparison container did not commit its result: {role}')
        sample['container_id'] = launch['container_id']
        arms.append(sample)
    report = dict(arms=arms, scope='bounded Stage1 training only')
    commit_json(Path(c['output'])/'results.json', report)
    return report
