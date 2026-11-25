"""Collect the exact completed Stage1 checkpoint and frozen inputs for scoring."""
from pathlib import Path
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.native_seal_operations import verify_memory
from tools.reports.stop_card_stage1_at_epoch import endpoint, inspect, read, remote


def audit(c):
    operations, output = Path(c['operations']), Path(c['output'])
    handle_path = operations/'formal-launch.json'
    launch = read(handle_path)
    host, cid = launch['host'], launch['container_id']
    original = inspect(launch)
    verify_memory(original, launch['argv'])
    if (original['State']['Status'] != 'exited' or original['State']['Running']
            or original['State']['Pid'] != 0 or original['State']['ExitCode'] != 0
            or read(operations/'formal-wait.json') != {'exit_code': 0}
            or read(operations/'formal-inspect.json')['Id'] != cid):
        raise ValueError('the exact training container has not exited successfully')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(launch['remote_root'])
    training = output/'training.json'
    training.write_bytes(remote(host, 'cat', str(root/'formal/config.json')).stdout)
    train = read(training)
    if 'updates' in c:
        updates = c['updates']
    else:
        design = Path(c['package'])/'entry/design.json'
        if file_hash(design) != read(operations/'formal-admission.json')['entry_pins']['design.json']:
            raise ValueError('Stage1 design differs from the admitted package')
        updates = read(design)['stage1']['total_updates']
    if (train['stop_update'] != updates or train['source_contract']['base_id'] != train['base_id']
            or read(operations/'formal-collected.json')['stop_update' if train['start_update'] else 'updates'] != updates):
        raise ValueError('training receipt differs from the registered endpoint')
    proof = endpoint(dict(updates=updates, training=dict(sha256=file_hash(training)),
                          hash_timeout_seconds=c['hash_timeout_seconds'], completed_window=True), train, launch)
    views = {'/jev-entry': root/'entry', '/jev-work': root}
    for mount in launch['argv']:
        if mount.endswith(':ro') and mount.count(':') == 2:
            source, destination, _ = mount.split(':')
            views[destination] = Path(source)

    def collect(container_path, filename, expected=None):
        path = Path(container_path)
        for mount in sorted(views, key=len, reverse=True):
            if path.is_relative_to(mount):
                remote_path = views[mount]/path.relative_to(mount)
                local = output/filename
                local.write_bytes(remote(host, 'cat', str(remote_path)).stdout)
                digest = file_hash(local)
                if expected is not None and digest != expected:
                    raise ValueError(f'endpoint input changed: {filename}')
                return dict(path=str(local), sha256=digest)
        raise ValueError('endpoint input was not mounted in the stopped container')

    inputs = {'training': dict(path=str(training), sha256=file_hash(training))}
    for key, name in [('single_card', 'cards'), ('layout_cache', 'acoustics')]:
        source = train[key]
        inputs[name] = collect(source['path'], name+'.json', source['sha256'])
    entry = Path(c['package'])/'entry'
    for name, filename in [('synthetic', 'synthetic.json'), ('real_lengths', 'native-lengths.json')]:
        path = entry/filename
        expected = read(operations/'formal-admission.json')['entry_pins'][filename]
        if file_hash(path) != expected:
            raise ValueError(f'frozen training input changed: {filename}')
        inputs[name] = dict(path=str(path), sha256=expected)
    (output/'source').mkdir()
    manifest = collect(str(Path(train['source_policy'])/'MANIFEST.json'), 'source/MANIFEST.json')
    source = read(Path(manifest['path']))
    if (source['policy_id'] != train['source_policy_id'] or source['contract'] != train['source_contract']):
        raise ValueError('fresh source manifest differs from the training contract')
    checkpoint_manifest = output/'endpoint-manifest.json'
    checkpoint_manifest.write_bytes(remote(host, 'cat', str(Path(proof['policy'])/'MANIFEST.json')).stdout)
    if file_hash(checkpoint_manifest) != proof['manifest_sha256']:
        raise ValueError('checkpoint manifest changed after audit')
    inputs.update(policy=str(output/'source'), policy_sha256=manifest['sha256'])
    commit_json(output/'inputs.json', inputs)
    commit_json(output/'inspect.json', original)
    commit_json(output/'receipt.json', dict(proof, passed=True, container_exited=True,
                 completed_window=True, ranks=len(train['physical_devices']), container_id=cid, host=host,
                 launch_sha256=file_hash(handle_path), config_sha256=file_hash(training),
                 exit_code=0, inspect_sha256=file_hash(output/'inspect.json'),
                 inputs_sha256=file_hash(output/'inputs.json')))


if __name__ == '__main__':
    audit(load_config(*sys.argv[1:])[0])
