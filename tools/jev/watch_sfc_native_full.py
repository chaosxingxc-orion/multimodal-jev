#!/usr/bin/env python3
"""Run full native SFC ID then OOD after the exact queued Jev TRAIN probe."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_native_full import RUNTIME, select_inputs, check_probe
from infer_sfc_omni_full import SOURCE_FILES
from verify_sfc_omni_full import score
from verify_sfc_jev_probe import accept as accept_jev_probe
from watch_omni_transfer_readiness import observe_dependency
from watch_qwen3_omni_probe import inspect, wait


def read(path):
    return json.loads(Path(path).read_text())


def check_pins(root,plan):
    if (plan['schema']!='sfc-native-full-queue-v1' or plan['splits']!=['test_id','test_ood']
            or plan['physical_devices']!=[6,7] or plan['automatic_retry'] is not False
            or plan['wall_seconds']!=86400 or plan['dependency_seconds']!=110000):
        raise ValueError('complete registered ID/OOD queue and bounded runtime required')
    for name,value in plan['source_files'].items():
        if file_hash(root/'source'/name)!=value:raise ValueError('queue source changed')
    for path,value in plan['external_files'].items():
        if file_hash(path)!=value:raise ValueError('registered external input changed')
    if {k:importlib.metadata.version(k) for k in plan['scorer_versions']}!=plan['scorer_versions']:
        raise ValueError('registered scorer environment required')


def prepare_split(root,plan,split):
    """Resolve successful TRAIN acceptance only after registered dependencies finish."""
    if split not in plan['splits']:raise ValueError('registered split required')
    native=Path(plan['native_probe']);prior=read(native/'config.json');launch=read(native/'launch.json')
    release=Path(plan['release']);manifest=read(release/'MANIFEST.json')
    cfg=dict(RUNTIME,split=split,release=str(release),manifest_sha256=file_hash(release/'MANIFEST.json'),
        requests_sha256=manifest['splits'][split]['requests_sha256'],native_probe=str(native),
        native_probe_container_id=launch['container_id'],native_probe_acceptance_sha256=file_hash(native/'acceptance.json'),
        audio_model=prior['audio_model'],versions=prior['versions'],asset_audit=prior['asset_audit'],image=plan['image'])
    select_inputs(cfg);check_probe(cfg)
    item=inspect(launch['container_id'])
    if (item['Id']!=launch['container_id'] or item['Image']!=plan['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0):
        raise ValueError('successful exact native TRAIN predecessor required')
    child=root/split;child.mkdir(exist_ok=False)
    (child/'source').symlink_to(root/'source',target_is_directory=True)
    atomic_json(child/'config.json',cfg)
    argv=['docker','run','-d','--pull=never','--name',root.name+'-'+split,
          '--network=host','--cpus=16','--memory=128g','--shm-size=8g']
    for device in ('davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv+=['--device',f'/dev/{device}:/dev/{device}']
    env=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        HCCL_SOCKET_IFNAME='bond0',GLOO_SOCKET_IFNAME='bond0',TP_SOCKET_IFNAME='bond0',
        HCCL_IF_BASE_PORT='63500',PYTHONPATH=str(root/'source/src'))
    for key,value in env.items():argv+=['-e',key+'='+value]
    for path in ('/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/usr/local/dcmi',
                 '/etc/hccn.conf','/etc/ascend_install.info','/mnt/cephfs/xingchao/models',
                 '/mnt/cephfs/xingchao/datasets',str(native)):
        argv+=['-v',path+':'+path+':ro']
    argv+=['-v',f'{root}:{root}:rw','-v',f'{root}/source:{root}/source:ro',plan['image'],
           'timeout','--signal=TERM','--kill-after=30s','86400s','python3','-B',
           str(root/'source/tools/jev/infer_sfc_omni_full.py'),'--config',str(child/'config.json'),
           '--output',str(child/'run')]
    prep=dict(argv=argv,image=plan['image'],source_commit=plan['source_commit'],physical_devices=[6,7],
        config_sha256=file_hash(child/'config.json'),source_files={n:file_hash(root/'source'/n) for n in SOURCE_FILES},
        requests=manifest['splits'][split]['requests'],wall_seconds=86400,automatic_retry=False)
    atomic_json(child/'preparation.json',prep)
    return child,prep


def dispatch_once(child,prep):
    if any((child/n).exists() for n in ('dispatch-intent.json','launch.json','run','dispatch.log')):
        raise FileExistsError('prior TEST attempt; no replay')
    if file_hash(child/'config.json')!=prep['config_sha256']:
        raise ValueError('prepared TEST config changed')
    for name,value in prep['source_files'].items():
        if file_hash(child/'source'/name)!=value:raise ValueError('prepared TEST source changed')
    devices={prep['argv'][i+1].split(':')[0] for i,a in enumerate(prep['argv'][:-1]) if a=='--device'}
    if {d for d in devices if d.removeprefix('/dev/davinci').isdigit()}!={'/dev/davinci6','/dev/davinci7'}:
        raise ValueError('only NPU6/7 may be dispatched')
    for cid in subprocess.check_output(['ssh','npu','docker ps -q'],text=True,timeout=20).split():
        item=inspect(cid)
        if any(d['PathOnHost'] in ('/dev/davinci6','/dev/davinci7') for d in item['HostConfig'].get('Devices') or []):
            raise RuntimeError('reserved devices occupied; no dispatch')
    with (child/'dispatch-intent.json').open('x') as handle:
        json.dump(dict(preparation_sha256=file_hash(child/'preparation.json'),started_unix=time.time(),automatic_retry=False),handle)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    result=subprocess.run(['ssh','npu',shlex.join(prep['argv'])],capture_output=True,text=True,timeout=90)
    (child/'dispatch.log').write_text(result.stdout+result.stderr);cid=result.stdout.strip()
    if result.returncode or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
        raise RuntimeError('ambiguous dispatch; no retry')
    atomic_json(child/'launch.json',dict(container_id=cid,**prep))
    state=wait(child,cid,prep['image'],time.monotonic()+86700)
    atomic_json(child/'docker-terminal.json',dict(container_id=cid,state=state))
    if state['Status']!='exited' or state['ExitCode']!=0:
        raise RuntimeError('full native TEST container failed')


def execute(root):
    if (root/'evaluation-terminal.json').exists() or any((root/s).exists() for s in ('test_id','test_ood')):
        raise FileExistsError('preserve existing full TEST queue attempt')
    plan=read(root/'queue-plan.json');check_pins(root,plan)
    deadline=time.monotonic()+plan['dependency_seconds']
    while True:
        try:ready=observe_dependency(plan['dependency'])
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'queue-progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'queue-progress.json',dict(phase='waiting_jev_train78',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        if time.monotonic()>deadline:raise TimeoutError('dependency observation expired; no restart')
        time.sleep(20)
    check_pins(root,plan)
    predecessor=Path(plan['dependency']['root']);accepted=read(predecessor/'acceptance.json')
    accept_jev_probe(predecessor)
    if (accepted['decisions']!=78 or accepted['bundles']!=6
            or accepted['source_groups']!=4 or accepted['weight_updates']!=0
            or accepted['metrics_sha256']!=file_hash(predecessor/'score/metrics.json')):
        raise ValueError('complete Jev engineering predecessor acceptance required')
    for split in plan['splits']:
        check_pins(root,plan)
        child,prep=prepare_split(root,plan,split)
        atomic_json(root/'queue-progress.json',dict(phase='running_full_split',split=split,heartbeat_unix=time.time()))
        dispatch_once(child,prep)
        score(child,Path(plan['official_scorer']),plan['official_scorer_sha256'])
        atomic_json(child/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    atomic_json(root/'queue-progress.json',dict(phase='complete',splits=plan['splits'],heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'evaluation-terminal.json').exists():raise FileExistsError('existing queue terminal')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),automatic_retry=False))
