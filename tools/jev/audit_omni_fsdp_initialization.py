#!/usr/bin/env python3
"""Independently audit a finished two-rank initialization, never dispatch it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_fsdp_initialization_audit import validate_ranks
from watch_omni_npu_layer import inspect, read, remote, successful

PREPARATION = '24ee2d400d01058931aa49e2e70da408d7a4ea1756d71b0261a0ff5383c6fceb'
LAYOUT = '8829043e68573275c3fbed92b9896d7f0f86c5ee5973948be4502ef56101965b'


def validate_runtime(actual, prep, cid, entrypoint):
    successful(actual, cid)
    host = actual['HostConfig']; argv = prep['argv']
    devices = ['/dev/davinci4', '/dev/davinci5', '/dev/davinci_manager',
               '/dev/devmm_svm', '/dev/hisi_hdc']
    expected_devices = [dict(PathOnHost=d, PathInContainer=d, CgroupPermissions='rwm') for d in devices]
    if (actual['Image'] != prep['image'] or actual['RestartCount'] != 0
            or actual['State']['Dead'] or actual['State']['Error']
            or host['NanoCpus'] != 16*10**9 or host['Memory'] != 256*1024**3
            or host['MemorySwap'] != 256*1024**3 or host['ShmSize'] != 8*1024**3
            or host['NetworkMode'] != 'host' or host['IpcMode'] != 'private'
            or host['Privileged'] or host.get('DeviceRequests') or host.get('DeviceCgroupRules')
            or sorted(host['Devices'], key=lambda x:x['PathOnHost']) != sorted(expected_devices,key=lambda x:x['PathOnHost'])
            or actual['Config']['Cmd'] != argv[argv.index(prep['image'])+1:]
            or actual['Config']['Entrypoint'] != entrypoint
            or set(host['Binds']) != {argv[i+1] for i,v in enumerate(argv[:-1]) if v == '-v'}
            or not {argv[i+1] for i,v in enumerate(argv[:-1]) if v == '-e'} <= set(actual['Config']['Env'])):
        raise ValueError('exact prepared two-card runtime required')


def audit(root, layout, output):
    if output.exists():
        raise FileExistsError('preserve previous audit')
    if file_hash(root/'preparation.json') != PREPARATION or file_hash(layout) != LAYOUT:
        raise ValueError('original prepared run and accepted layout required')
    prep = read(root/'preparation.json'); launch = read(root/'launch.json')
    if launch['source_commit'] != prep['source_commit'] or launch['image'] != prep['image']:
        raise ValueError('launch source/image differs from preparation')
    cid = launch['container_id']
    actual = inspect(cid)
    image = json.loads(remote(['docker','image','inspect',prep['image']]).stdout)[0]
    validate_runtime(actual, prep, cid, image['Config']['Entrypoint'])
    for rank in (0,1):
        if read(root/'run'/f'rank-{rank}'/'terminal.json')['exit_code'] != 0:
            raise ValueError('both rank applications must terminate successfully')
    for name, pin in prep['source_files'].items():
        if (file_hash(Path(prep['source_root'])/name) != pin
                or hashlib.sha256(subprocess.check_output(['git','show',prep['source_commit']+':'+name])).hexdigest() != pin):
            raise ValueError('immutable runner differs from registered Git source')
    pre = Path(prep['preflight_root'])
    acceptance = read(pre/'acceptance.json')
    if (file_hash(pre/'acceptance.json') != prep['preflight_acceptance_sha256']
            or file_hash(pre/'run/result.json') != acceptance['result_sha256']):
        raise ValueError('accepted full CPU loading preflight changed')
    pre_result = read(pre/'run/result.json')
    if not pre_result['all_checkpoint_values_exact'] or not pre_result['all_buffers_exact']:
        raise ValueError('complete original CPU state proof required')
    for name, pin in pre_result['artifacts'].items():
        if Path(name).name != name or file_hash(pre/'run'/name) != pin:
            raise ValueError('accepted preflight artifact changed')
    original_path = root/'run/original-parameters.json'; original = read(original_path)
    ranks = []; pins = []
    for rank in (0,1):
        directory = root/'run'/f'rank-{rank}'
        result = read(directory/'result.json'); parameters = read(directory/'parameters.json')
        if result['rank'] != rank or result['parameters_sha256'] != file_hash(directory/'parameters.json'):
            raise ValueError('rank output file binding changed')
        ranks.append((result, parameters))
        pins.append(dict(rank=rank, result_sha256=file_hash(directory/'result.json'),
                         parameters_sha256=file_hash(directory/'parameters.json')))
    summary = validate_ranks(original, read(pre/'run/meta-parameters.json'),
        read(pre/'run/buffers.json'), read(layout)['explicit_units'], ranks,
        original_sha256=file_hash(original_path), preflight_sha256=prep['preflight_acceptance_sha256'])
    if (summary['parameters'] != 31719205488 or summary['parameter_tensors'] != 1407
            or summary['units'] != 108 or summary['buffers'] != 4):
        raise ValueError('complete original Omni state required')
    report = dict(schema='omni-fsdp-initialization-audit-v1', container_id=cid,
        image=prep['image'], source_commit=prep['source_commit'], preparation_sha256=PREPARATION,
        layout_sha256=LAYOUT, source_files_verified=len(prep['source_files']),
        preflight_acceptance_sha256=prep['preflight_acceptance_sha256'],
        original_parameters_sha256=file_hash(original_path), rank_outputs=pins, **summary,
        scope='Completed original two-rank NPU FSDP initialization manifests and runtime. '
              'Hashes compare saved rank evidence to the CPU reference loaded in this run; '
              'no independent second weight load, forward, probability parity or training claim. '
              'Allocator peaks are not total device HBM; dispatch prerequisite review remains separate.')
    write_once(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','layout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root,args.layout,args.output),indent=2))
