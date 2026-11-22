"""Freeze the scoring benchmark and dispatch it through the native operations runner."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import map_io
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze
from tools.reports.audit_card_scoring_benchmark import audit
from tools.reports.card_scoring_benchmark import make_panel, pinned
from tools.reports.native_seal_operations import run


def formal_policy(c, train, deploy):
    local = Path(c['formal']['epoch_handoff'])
    proof = json.loads((local/'receipt.json').read_bytes())
    capture = pinned(dict(path=local/'inspect.json', sha256=proof['inspect_sha256']))
    state = capture['State']
    updates = train['checkpoint_interval']
    stop = json.loads((local/'stop-intent.json').read_bytes())
    if (proof['passed'] is not True or proof['container_exited'] is not True
            or proof['updates'] != updates or updates >= train['stop_update']
            or proof['config_sha256'] != c['inputs']['training']['sha256']
            or proof['container_id'] != capture['Id'] or stop['container_id'] != capture['Id']
            or any(stop[k] != proof[k] for k in ('policy', 'policy_id', 'manifest_sha256', 'checkpoint_sha256'))
            or proof['host'] != deploy['host'] or proof['devices'] != deploy['devices']
            or state['Running'] or state['Status'] != 'exited' or state['Pid'] != 0
            or state['OOMKilled'] or proof['exit_code'] != state['ExitCode']):
        raise ValueError('audited, deliberately stopped epoch endpoint on target host required')
    target = Path(proof['policy'])
    if not target.is_absolute() or ':' in str(target) or target.name != f'policy-{updates:04d}':
        raise ValueError('invalid stopped epoch policy path')
    expected = {str(target/'MANIFEST.json'): proof['manifest_sha256'],
                str(target.parent/f'checkpoint-{updates:04d}-committed.json'): proof['checkpoint_sha256']}
    observed = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        deploy['host'], shlex.join(['sha256sum', '--', *expected])], text=True, timeout=30)
    if {path: digest for digest, path in (line.split(maxsplit=1) for line in observed.splitlines())} != expected:
        raise ValueError('committed Stage1 checkpoint or manifest differs')
    return target, proof['manifest_sha256'], proof['policy_id'], proof['checkpoint_sha256']


def package(c):
    root = Path(c['output'])
    entry = root/'entry'
    repo = Path(__file__).resolve().parents[2]
    deploy = c['deployment'].copy()
    inputs = c['inputs']
    policy = Path(inputs['policy'])
    names = ('training', 'cards', 'acoustics', 'synthetic', 'real_lengths')
    formal = c.get('formal')
    devices = deploy['devices']
    if formal:
        check_paths({'scoring package': root})
        if (len(devices) != c['replicas'] or len(set(devices)) != len(devices)
                or any(type(d) is not int or d < 0 for d in devices)):
            raise ValueError('one independent scoring replica per declared card required')
        if (len(c['widths']) != 1 or set(c['streaming']) != {'length_window'}
                or any(type(n) is not int or n < 1 for n in (
                    c['widths'][0], c['recordings_per_block'], c['streaming']['length_window'],
                    c['io_workers'], c['cpu_workers'], c['audit_min_parallel_blocks'],
                    c['torch_threads'], c['worker_timeout_seconds'], c['coordination_seconds']))
                or type(c['prefetch_audio']) is not int or c['prefetch_audio'] < c['io_workers']):
            raise ValueError('one positive batch width and a bounded streaming deployment required')
        train = pinned(inputs['training'])
        target, policy_sha, policy_id, checkpoint_sha = formal_policy(c, train, deploy)
        synthetic_cache = Path(formal['synthetic_cache'])
        if not synthetic_cache.is_absolute() or ':' in str(synthetic_cache):
            raise ValueError('absolute synthetic cache on target host required')
    deploy.pop('devices')
    entry.mkdir(parents=True)
    panel = make_panel(dict(c, **{n: inputs[n] for n in names},
                            policy=str(policy), policy_sha256=inputs['policy_sha256']), entry/'panel.json')
    if formal:
        policy_root = '/jev-endpoint'
        cache_root = '/jev-acoustic-prepare/synthetic-cache'
        mounts = [f'{target}:{policy_root}', f'{synthetic_cache}:{cache_root}']
        if formal.get('resume_from'):
            from tools.reports.context_macro_probe_operations import inspect_runtime

            origin = pinned(formal['resume_from'])
            state, = json.loads(subprocess.check_output([
                'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', deploy['host'],
                shlex.join(['docker', 'inspect', origin['container_id']])], timeout=30))
            if (origin['host'] != deploy['host'] or state['Image'] != deploy['image']
                    or state['State']['Running'] or state['State']['Status'] != 'exited'
                    or state['State']['Pid'] != 0 or state['State']['OOMKilled']):
                raise ValueError('resume requires inspected stopped scoring container on the target host')
            inspect_runtime(state, origin)
            resume = Path(origin['remote_root'])/'score'
            if not resume.is_absolute() or not resume.is_relative_to(Path(deploy['remote_root']).parent):
                raise ValueError('isolated stopped scoring output required')
            mounts.append(f'{resume}:/jev-score-resume')
        deploy['read_only_mounts'] = [*deploy.get('read_only_mounts', []), *mounts]
    elif inputs.get('synthetic_cache'):
        synthetic_cache = Path(inputs['synthetic_cache'])
        if not synthetic_cache.is_absolute() or ':' in str(synthetic_cache):
            raise ValueError('absolute synthetic cache on target host required')
        cache_root = '/jev-benchmark-synthetic-cache'
        deploy['read_only_mounts'] = [*deploy.get('read_only_mounts', []),
                                     f'{synthetic_cache}:{cache_root}']
        shutil.copytree(policy, entry/'policy')
        policy_root, policy_sha = '/jev-entry/policy', inputs['policy_sha256']
    else:
        from multimodal_jev.encoded_audio_cache import key_for
        keys = sorted({key_for(row['binding'])+'.pt' for row in panel['rows'] if row['synthetic']})
        synthetic = entry/'synthetic-cache'
        synthetic.mkdir()
        with subprocess.Popen(['ssh', '-o', 'BatchMode=yes', inputs['cache_host'],
                                shlex.join(['docker', 'exec', inputs['cache_container'], 'tar', '-C',
                                            inputs['cache_path'], '-cf', '-', '--', *keys])],
                               stdout=subprocess.PIPE) as remote:
            with tarfile.open(mode='r|', fileobj=remote.stdout) as stream:
                stream.extractall(synthetic, filter='data')
            if remote.wait() or sorted(p.name for p in synthetic.iterdir()) != keys:
                raise ValueError('failed to transfer matched synthetic feature entries')
        shutil.copytree(policy, entry/'policy')
        policy_root, policy_sha = '/jev-entry/policy', inputs['policy_sha256']
        cache_root = '/jev-entry/synthetic-cache'
    for name in ('training', 'cards'):
        shutil.copyfile(inputs[name]['path'], entry/(name+'.json'))
    snapshot = freeze(repo, entry/'source', patterns=c['source_patterns'], io_workers=c['io_workers'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json', entry/'native-seal-container.json')
    spec = {k:c[k] for k in ('widths','io_workers','prefetch_audio','torch_threads',
                             'worker_timeout_seconds','max_model_len','probability_atol')}
    if 'streaming' in c:
        spec['streaming'] = c['streaming']
    if c.get('capture_host_memory'):
        spec['capture_host_memory'] = True
    spec.update(phase='card_scoring_benchmark', devices=devices, output='/jev-work/score',
                worker_config='/jev-entry/score.json', policy=policy_root,
                policy_sha256=policy_sha, synthetic_cache=cache_root,
                source=dict(root='/jev-entry/source', sha256=snapshot['manifest_sha256']))
    entries = ('panel', 'training', 'cards')
    if formal:
        entries += ('full',)
        spec.update(cpu_workers=c['cpu_workers'], audit_min_parallel_blocks=c['audit_min_parallel_blocks'])
        spec['formal'] = dict(policy_id=policy_id, updates=train['checkpoint_interval'], image=deploy['image'],
                              checkpoint_sha256=checkpoint_sha, coordination_seconds=c['coordination_seconds'],
                              recordings_per_block=c['recordings_per_block'])
        if formal.get('resume_from'):
            spec['formal']['resume_from'] = '/jev-score-resume'
    for name in entries:
        spec[name] = dict(path=f'/jev-entry/{name}.json', sha256=file_hash(entry/(name+'.json')))
    commit_json(entry/'score.json', dict(schema='jev-local-macros-v1',
                                        macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=spec)))
    archive = root/'entry.tar'
    with tarfile.open(archive, 'x') as tar:
        tar.add(entry, arcname='entry')
    pins = ['score.json', 'source/FILES.json', 'native-seal-container.json',
            *(name+'.json' for name in entries)]
    if not formal:
        pins.append('policy/MANIFEST.json')
    deploy.update(archive=str(archive), entry_archive_sha256=file_hash(archive),
                  entry_pins=dict(map_io(lambda p: (p, file_hash(entry/p)), pins, workers=c['io_workers'])[0]),
                  chains=[[[dict(name='score', devices=devices, collect='score/receipt.json', **c['resources'])]]],
                  device_group_lease=True, wait_for_lease=bool(formal), idle_devices=devices)
    commit_json(root/'operations.json', deploy)
    return deploy


def execute(config):
    root = Path(config['output'])
    start = time.monotonic()
    deployment = package(config)
    if run(deployment):
        raise ValueError('scoring failed; see native operations receipts')
    collected = root/'collected'
    collected.mkdir()
    command = shlex.join(['tar', '-C', deployment['remote_root'], '-cf', '-', 'score'])
    with subprocess.Popen(['ssh', '-o', 'BatchMode=yes', deployment['host'], command],
                          stdout=subprocess.PIPE) as remote:
        with tarfile.open(mode='r|', fileobj=remote.stdout) as stream:
            stream.extractall(collected, filter='data')
        if remote.wait():
            raise ValueError('scoring result transport failed')
    report = audit(collected/'score', root/'entry', Path(deployment['operations']))
    if config.get('formal'):
        confusion = dict(report.pop('confusion'), epoch=0, checkpoint_sha256=report['checkpoint_sha256'],
                         policy_id=report['policy_id'], full_sha256=report['full_sha256'])
        confusion_path = root/'confusion-epoch-1.json'
        commit_json(confusion_path, confusion)
        report['confusion_sha256'] = file_hash(confusion_path)
        report.update(wall_seconds=time.monotonic()-start, predictions_root=str(collected/'score'))
    commit_json(root/'audit.json', report)
    return report


def capacity(experiment, local):
    from tools.reports.experiment_ledger import plan
    from tools.reports.experiment_package import REGISTRY, committed

    selected = plan(experiment, states=('in_progress', 'planned'))
    pipeline = json.loads(committed(selected['commit'], REGISTRY/(selected['experiment_id']+'.json')))
    template = Path(pipeline['scoring_capacity_window'])
    if template.read_bytes() != committed(selected['commit'], template):
        raise ValueError('scoring capacity protocol differs from the committed ledger')
    return execute(load_config(template, local)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('local')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    c, _ = load_config(args.config, args.local)
    result = package(c) if args.prepare_only else execute(c)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
