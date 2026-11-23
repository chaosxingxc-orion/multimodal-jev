"""Dispatch a bounded lane of an existing, frozen full TRAIN score."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze, verify
from tools.reports.card_scoring_benchmark import pinned, scoring_offsets
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
