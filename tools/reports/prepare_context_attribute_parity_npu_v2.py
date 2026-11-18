"""Seal an NPU parity package from the authenticated CPU preparation.

Produces a reviewable Docker command, but never contacts or starts a server.
Existing CPU packages and active experiment dependencies remain immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tools.reports.run_context_attribute_parity_pair_v2 import verify

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
PREFIX = '/data/xingchao/multimodal-jev/simple-final-v1/'
TAG = 'context-attribute-parity-npu-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relocate(value, old, new):
    if isinstance(value, dict):
        return {relocate(k, old, new): relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old, new) for v in value]
    if isinstance(value, str) and (value == old or value.startswith(old + '/')):
        return new + value[len(old):]
    return value


def command(root, model):
    if root != PREFIX + TAG or model != '/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf':
        raise ValueError('Fixed root and local model required')
    argv = ['docker', 'create', '--name', TAG, '--pull=never', '--network=none',
            '--cpus=8', '--memory=64g', '--shm-size=2g', '--log-driver=json-file',
            '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2']
    for device in ('davinci7', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
        argv += ['--device', '/dev/' + device + ':/dev/' + device]
    for mount in (root + ':' + root, root + '/package:' + root + '/package:ro',
                  root + '/home:/root', root + '/cache:/jev-tmp', model + ':' + model + ':ro',
                  '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):
        argv += ['-v', mount]
    env = dict(PYTHONPATH=root+'/package:'+root+'/package/src:'+root+'/package/tools/jev',
               PYTHONDONTWRITEBYTECODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
               HCCL_DETERMINISTIC='true', TMPDIR='/jev-tmp', VLLM_RPC_BASE_PATH='/jev-tmp')
    for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
        env[key] = '/jev-tmp'
    for key, value in env.items():
        argv += ['-e', key + '=' + value]
    return argv + ['--entrypoint', '/opt/multimodal-jev/runtime-entrypoint.sh', IMAGE,
        'python', root+'/package/tools/reports/run_context_attribute_parity_pair_v2.py',
        '--package', root+'/package', '--output', root+'/evaluation']


def prepare(source, evidence, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    evidence = Path(evidence)
    proof = json.loads(evidence.read_text())
    state = proof['state']
    original_digest = verify(source)
    if (proof['image'] != IMAGE or state['Status'] != 'exited' or state['ExitCode'] != 0
            or state['OOMKilled'] or proof['source_manifest_sha256'] != original_digest
            or proof['counts'] != dict(tests=9, failures=0, errors=0, skipped=0)):
        raise ValueError('Authenticated successful fixed-image CPU preparation required')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Separate output package required')
    output.mkdir(exist_ok=False)
    package = output/'package'
    shutil.copytree(source, package)
    (package/'FILES.json').unlink()
    runner = Path('tools/reports/run_context_attribute_parity_pair_v2.py')
    shutil.copy2(runner, package/runner)
    probe = Path('tools/reports/probe_context_attribute_parity_v2.py')
    shutil.copy2(probe, package/probe)
    config = json.loads((package/'probe-config.json').read_text())
    root = PREFIX+TAG
    config = relocate(config, str(Path(config['source_root']).parent), root)
    config['physical_devices'] = [7]
    # Bind the copied payload and runner; exclude self-referential manifests.
    config['pins'] = {root+'/package/'+str(p.relative_to(package)): sha(p)
                      for p in package.rglob('*') if p.is_file() and p.name != 'probe-config.json'}
    (package/'probe-config.json').write_text(json.dumps(config, indent=2)+'\n')
    files = {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()}
    (package/'FILES.json').write_text(json.dumps(files, indent=2)+'\n')
    digest = verify(package)
    receipt = dict(schema='context-attribute-parity-npu-preparation-v1',
        status='sealed_not_launched', training_dispatch_allowed=False,
        remote_root=root, image=IMAGE, source_manifest_sha256=digest, source_files=len(files),
        source_cpu_manifest_sha256=original_digest,
        cpu_evidence=dict(path=str(evidence.resolve()), sha256=sha(evidence)),
        argv=command(root, config['model']),
        lease='.local/simple-final-v1/npu175-evaluation-training.lock',
        prerequisites=['existing_simple_training_evaluation_queue_priority',
                       'exclusive_lease_and_actual_device7_idle', 'fixed_image_present',
                       'local_hot_storage_capacity', 'remote_package_hash_verification'],
        lifecycle='Persist attempt and container ID before start; retain lease through exact docker wait; observation failure never authorizes recreation.',
        scope='Six TRAIN requests, HF then vLLM in fresh processes; not full input, update, restore or efficacy admission')
    (output/'PREPARED.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'evidence', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    receipt = prepare(args.source, args.evidence, args.output)
    print(json.dumps({k: receipt[k] for k in ('status', 'source_files', 'source_manifest_sha256')}))
