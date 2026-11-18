"""Render the fixed Ascend runtime topology; never execute Docker or SSH."""
import argparse
import json
import os
from pathlib import PurePosixPath
import re

from multimodal_jev.local_config import load_config


IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def absolute(value):
    if (not isinstance(value, str) or not value.startswith('/') or value == '/'
            or any(c in value for c in (':', '\n', '\r', '\x00'))
            or '..' in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        raise ValueError('Canonical absolute mount path without delimiters required')
    return value


def plan(config):
    keys = {'schema', 'host', 'image', 'mount_source_container', 'remote_root',
            'encoded_audio_cache', 'audio_mirror_root', 'shared_root', 'driver_root',
            'physical_devices', 'resources'}
    if set(config) != keys or config['schema'] != 'ascend-context-runtime-plan-v1':
        raise ValueError('Unsupported runtime plan schema')
    if config['image'] != IMAGE:
        raise ValueError('Unvalidated runtime image')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config['host']):
        raise ValueError('Explicit SSH host alias required')
    if not re.fullmatch(r'[0-9a-f]{64}', config['mount_source_container']):
        raise ValueError('Exact mount source container identity required')
    devices = config['physical_devices']
    if (not isinstance(devices, list) or len(devices) != 4 or len(set(devices)) != 4
            or any(type(i) is not int or not 0 <= i <= 7 for i in devices)):
        raise ValueError('Four distinct physical NPU IDs required')
    if config['resources'] != dict(cpus=24, memory='128g', shm_size='8g'):
        raise ValueError('v1 preserves the admitted resource profile')
    root, cache, mirror, shared, driver = [absolute(config[k]) for k in
        ('remote_root', 'encoded_audio_cache', 'audio_mirror_root', 'shared_root', 'driver_root')]
    if not re.fullmatch(r'context-rl-(grpo|rloo)-[a-z0-9-]+', PurePosixPath(root).name):
        raise ValueError('Named experiment directory required')
    hot = [PurePosixPath(root), PurePosixPath(cache)]
    if any(p.is_relative_to(PurePosixPath(shared)) for p in hot):
        raise ValueError('Hot paths cannot be under shared assets')
    argv = ['docker','create','--pull=never','--network=none','--name',PurePosixPath(root).name,
            '--cpus=24','--memory=128g','--shm-size=8g','--volumes-from',config['mount_source_container']+':ro',
            '--log-driver=json-file','--log-opt','max-size=20m','--log-opt','max-file=2']
    for device in [*(f'davinci{i}' for i in devices), 'davinci_manager','devmm_svm','hisi_hdc']:
        argv += ['--device',f'/dev/{device}:/dev/{device}']
    mounts = [root+':'+root,root+'/exchange:/jev-exchange',root+'/home:/root',root+'/cache:/jev-tmp',
              str(PurePosixPath(cache).parent)+':'+str(PurePosixPath(cache).parent)+':ro',
              cache+':'+cache,mirror+':'+mirror+':ro',shared+':'+shared+':ro',
              driver+':/usr/local/Ascend/driver:ro']
    for name in ('source','initial','evidence','run.py','config.json','service-config.json','RELEASE.json','PREPARED.json'):
        mounts.append(root+'/'+name+':'+root+'/'+name+':ro')
    for mount in mounts:
        argv += ['-v',mount]
    env = dict(PYTHONPATH=root+'/source/src:'+root+'/source/tools/jev',PYTHONDONTWRITEBYTECODE='1',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',TMPDIR='/jev-tmp')
    for key in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[key] = root+'/cache'
    for key,value in env.items():
        argv += ['-e',key+'='+value]
    argv += ['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',root+'/run.py']
    return dict(schema='ascend-context-runtime-plan-v1',host=config['host'],argv=argv,
                training_dispatch_allowed=False,
                required_before_launch=['target_mount_storage_check','device_lease_and_availability',
                    'exact_image_and_successful_mount_donor','new_package_and_release_identity'],
                scope='Command plan only; no host access or deployment admission')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--local',default=os.environ.get('JEV_LOCAL_CONFIG','.local/config/macros.json'))
    args = parser.parse_args()
    config, provenance = load_config(args.config,args.local)
    print(json.dumps(dict(plan(config),provenance=provenance),indent=2))


if __name__ == '__main__':
    main()
