"""Freeze the scoring benchmark and dispatch it through the native operations runner."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import map_io
from multimodal_jev.encoded_audio_cache import key_for
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.audit_card_scoring_benchmark import audit
from tools.reports.card_scoring_benchmark import make_panel
from tools.reports.native_seal_operations import run


def package(c):
    root = Path(c['output'])
    entry = root/'entry'
    entry.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    inputs = c['inputs']
    policy = Path(inputs['policy'])
    names = ('training', 'cards', 'acoustics', 'synthetic', 'real_lengths')
    panel = make_panel(dict(c, **{n: inputs[n] for n in names},
                            policy=str(policy), policy_sha256=inputs['policy_sha256']), entry/'panel.json')
    # Reuse only the authenticated synthetic feature entries needed for this finite window.
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
    for name in ('training', 'cards'):
        shutil.copyfile(inputs[name]['path'], entry/(name+'.json'))
    snapshot = freeze(repo, entry/'source', patterns=c['source_patterns'], io_workers=c['io_workers'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json', entry/'native-seal-container.json')
    deploy = c['deployment'].copy()
    devices = deploy.pop('devices')
    spec = {k:c[k] for k in ('widths','io_workers','prefetch_audio','torch_threads',
                             'worker_timeout_seconds','max_model_len','probability_atol')}
    if 'streaming' in c:
        spec['streaming'] = c['streaming']
    spec.update(phase='card_scoring_benchmark', devices=devices, output='/jev-work/score',
                worker_config='/jev-entry/score.json', policy='/jev-entry/policy',
                policy_sha256=inputs['policy_sha256'], synthetic_cache='/jev-entry/synthetic-cache',
                source=dict(root='/jev-entry/source', sha256=snapshot['manifest_sha256']))
    for name in ('panel', 'training', 'cards'):
        spec[name] = dict(path=f'/jev-entry/{name}.json', sha256=file_hash(entry/(name+'.json')))
    commit_json(entry/'score.json', dict(schema='jev-local-macros-v1',
                                        macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=spec)))
    archive = root/'entry.tar'
    with tarfile.open(archive, 'x') as tar:
        tar.add(entry, arcname='entry')
    pins = ['score.json','panel.json','training.json','cards.json','policy/MANIFEST.json',
            'source/FILES.json','native-seal-container.json']
    deploy.update(archive=str(archive), entry_archive_sha256=file_hash(archive),
                  entry_pins=dict(map_io(lambda p: (p, file_hash(entry/p)), pins, workers=c['io_workers'])[0]),
                  chains=[[[dict(name='score', devices=devices, collect='score/receipt.json', **c['resources'])]]],
                  device_group_lease=True, wait_for_lease=False, idle_devices=devices)
    commit_json(root/'operations.json', deploy)
    return deploy


def execute(config):
    root = Path(config['output'])
    deployment = package(config)
    if run(deployment):
        raise ValueError('scoring benchmark failed; see native operations receipts')
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
