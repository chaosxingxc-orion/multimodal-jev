"""Authenticate completed resident replay before the unchanged FSC workload."""
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed


def authenticate(config, root):
    if config.get('priority_kind') == 'utility_curves12':
        from run_intent_utility_curves import authenticate_complete
        return authenticate_complete(config,root)
    prior = read(root/'config.json')
    if prior['image'] != config['image']:
        raise ValueError('priority predecessor image differs')
    check_pins(prior['pins'])
    for field in ('training', 'suite'):
        target = Path(prior[field]); receipt = read(target/'launch.json')
        if receipt != read(root/f'run/{field}-dispatched.json'):
            raise ValueError('priority dispatch differs')
        if receipt['image'] != config['image'] or not completed(receipt['container_id'], config['image']):
            raise ValueError('actual priority completion required')
        check_pins({str(target/'config.json'): receipt['config_sha256'],
                    str(target/'source-manifest.json'): receipt['source_manifest_sha256']})
    suite = Path(prior['suite']); accepted = read(suite/'ACCEPTED.json'); verified = read(suite/'verified.json')
    if (read(suite/'controller-terminal.json')['exit_code'] != 0
            or accepted.get('passed') is not True or verified.get('passed') is not True
            or accepted['verified_sha256'] != file_hash(suite/'verified.json')
            or verified['updates'] != 2 or len(verified['ranks']) != 4
            or verified['full_gate']['requests'] != 178
            or accepted['adapter_id'] != verified['adapter_id']
            or accepted != read(root/'run/accepted.json')):
        raise ValueError('complete independently accepted resident replay required')
    check_pins(verified['files'])
    return dict(root=str(root), accepted=accepted,
                verified_sha256=file_hash(suite/'verified.json'))


def wait_tail(config, output):
    root = Path(config['priority_predecessor'])
    for name in ('config.json', 'launch.json'):
        if config['pins'].get(str(root/name)) != file_hash(root/name):
            raise ValueError('pinned priority predecessor required')
    receipt = read(root/'launch.json')
    while True:
        check_pins(config['pins']); terminal = root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code'] != 0:
                raise ValueError('priority predecessor failed')
            atomic_json(output/'priority-accepted.json', authenticate(config, root))
            return
        try:
            actual = (Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():
                continue
            raise ValueError('priority predecessor missing without terminal')
        if actual != receipt['argv']:
            raise ValueError('priority predecessor PID reused')
        atomic_json(output/'progress.json', dict(stage='waiting_priority_training_acceptance',
                    pid=receipt['pid'], updated_unix=time.time()))
        time.sleep(600)
