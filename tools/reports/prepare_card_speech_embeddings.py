"""Launch sharded pre-card speech export against an audited Stage1 checkpoint."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_device_config import physical_devices
from multimodal_jev.card_experiment_config import reference_settings
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze
from tools.reports.card_scoring_benchmark import pinned
from tools.reports.native_seal_operations import run
from tools.reports.prepare_card_scoring_benchmark import formal_policy, ssh_limits
from tools.reports.prepare_tpo_windows import macros


def execute(c):
    deployment = c['deployment'].copy()
    devices = physical_devices(deployment.pop('devices'))
    root = Path(c['output'])
    check_paths({'speech embedding export': root})
    endpoint = Path(c['endpoint'])
    proof = pinned(dict(path=endpoint/'receipt.json',sha256=c['endpoint_sha256']))
    inputs = pinned(dict(path=endpoint/'inputs.json',sha256=proof['inputs_sha256']))
    train = pinned(inputs['training'])
    acoustics = pinned(inputs['acoustics'])
    scoring = pinned(c['score_audit'])
    roster = pinned(c['roster'])
    if (scoring['policy_id'] != proof['policy_id'] or scoring['checkpoint_sha256'] != proof['checkpoint_sha256']
            or scoring['full_sha256'] != c['roster']['sha256'] or scoring['recordings'] != len(roster['rows'])
            or scoring['passed'] is not True or roster['source_policy_id'] != proof['policy_id']):
        raise ValueError('membership score and speech reference must be the same audited endpoint and TRAIN roster')
    policy, policy_sha, policy_id, _ = formal_policy(
        dict(c, formal=dict(epoch_handoff=str(endpoint)), inputs=inputs), train, deployment)
    index = train['packed_acoustics']
    mount = c['acoustic_mount']
    source, sep, target = mount.partition(':')
    index_path = Path(index['path'])
    if not sep or not Path(source).is_absolute() or not index_path.is_relative_to(target):
        raise ValueError('published acoustic pack mount must contain the pinned index')
    ssh, timeout = ssh_limits(c)
    observed = subprocess.check_output([*ssh, deployment['host'], shlex.join(
        ['sha256sum', '--', str(Path(source)/index_path.relative_to(target))])], timeout=timeout)
    if observed.decode().split()[0] != index['sha256']:
        raise ValueError('remote acoustic index digest differs')
    entry = root/'entry'
    entry.mkdir(parents=True, exist_ok=False)
    roster_path = entry/'roster.json'
    roster_path.write_bytes(Path(c['roster']['path']).read_bytes())
    if file_hash(roster_path) != c['roster']['sha256']:
        raise ValueError('roster changed during packaging')
    repo = Path(__file__).resolve().parents[2]
    snapshot = freeze(repo, entry/'source', patterns=c['source_patterns'], io_workers=c['io_workers'])
    source_pin = dict(root='/jev-entry/source',sha256=snapshot['manifest_sha256'])
    config_name = 'native-seal-container.json'
    (entry/config_name).write_bytes((repo/'configs/portable/operations'/config_name).read_bytes())
    tasks = []
    for shard, device in enumerate(devices):
        name = f'vectors-{shard}'
        spec = dict(phase='card_speech_embeddings', source=source_pin, io_workers=c['io_workers'],
                    batch=c['batch'], prefetch=c['prefetch'], shard=shard, shards=len(devices),
                    output='/jev-work/'+name, index=index, audio_token_id=acoustics['acoustic_encoding']['audio_token_id'],
                    roster=dict(path='/jev-entry/roster.json',sha256=c['roster']['sha256']),
                    policy_sha256=policy_sha, policy=dict(
                        snapshot='/jev-endpoint', policy_id=policy_id, base_id=train['base_id'],
                        model=train['model'], contract=train['source_contract'],physical_devices=[device],
                        reference_inference=reference_settings(c), process_device_scope=dict(
                            kind='same_container_runtime_visibility_v1',mounted_physical=[device],visible_ordinal='0')))
        commit_json(entry/(name+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',spec))
        tasks.append(dict(name=name, devices=[device], visible_ordinal='0',
                          collect=name+'/receipt.json', **c['resources']))
    archive = root/'entry.tar'
    with tarfile.open(archive,'x') as stream:
        stream.add(entry, arcname='entry')
    pins = {*entry.glob('*.json'),entry/'source/FILES.json'}
    deployment.update(archive=str(archive), entry_archive_sha256=file_hash(archive),
                      entry_pins={str(p.relative_to(entry)):file_hash(p) for p in sorted(pins)},
                      chains=[[tasks]], device_group_lease=True, wait_for_lease=True,
                      read_only_mounts=[*deployment.get('read_only_mounts',[]), mount, f'{policy}:/jev-endpoint'])
    commit_json(root/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',deployment))
    if run(deployment):
        raise ValueError('speech embedding containers did not exit successfully')
    output = root/'collected'
    output.mkdir()
    command = shlex.join(['tar','-C',deployment['remote_root'],'-cf','-',*[task['name'] for task in tasks]])
    with subprocess.Popen([*ssh,deployment['host'],command],stdout=subprocess.PIPE) as remote:
        with tarfile.open(mode='r|', fileobj=remote.stdout) as stream:
            stream.extractall(output, filter='data')
        if remote.wait():
            raise ValueError('speech vector transport failed')
    ids = sorted(r['id'] for r in roster['rows'])
    blocks, widths = [], set()
    import numpy as np
    for shard, task in enumerate(tasks):
        base = output/task['name']
        path = base/'receipt.json'
        rec = json.loads(path.read_bytes())
        authority = Path(deployment['operations'])/(task['name']+'-collected.json')
        vectors_path = base/'vectors.npy'
        vectors = np.load(vectors_path, mmap_mode='r',allow_pickle=False)
        if (json.loads(authority.read_bytes()) != rec or rec['schema'] != 'jev-card-speech-vectors-v1'
                or rec['policy_id'] != policy_id or rec['policy_sha256'] != policy_sha
                or rec['roster_sha256'] != c['roster']['sha256'] or rec['acoustic_sha256'] != index['sha256']
                or rec['shard'] != shard or rec['shards'] != len(devices)
                or sorted(rec['ids']) != ids[shard::len(devices)]
                or vectors.dtype != np.dtype('float32') or rec['recordings'] != len(rec['ids'])
                or vectors.shape != (len(rec['ids']),rec['width']) or not np.isfinite(vectors).all()
                or file_hash(vectors_path) != rec['vectors_sha256']):
            raise ValueError('transported vector scope, values or digest differ')
        widths.add(rec['width'])
        blocks.append(dict(name=task['name'],receipt_sha256=file_hash(path),vectors_sha256=rec['vectors_sha256']))
    if len(widths) != 1:
        raise ValueError('speech readout width differs across devices')
    manifest = dict(schema='jev-audited-card-speech-embeddings-v1', policy_id=policy_id, policy_sha256=policy_sha,
                    roster_sha256=c['roster']['sha256'], acoustic_sha256=index['sha256'],
                    recordings=len(ids), width=widths.pop(), devices=devices, blocks=blocks)
    commit_json(root/'audit.json',manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(execute(load_config(*sys.argv[1:])[0])))
