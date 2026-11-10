#!/usr/bin/env python3
"""One full-decode profile after both original capacity queues succeed."""
import argparse
import fcntl
from pathlib import Path
import tempfile
import time

import watch_audio2tool_jev_capacity as jev
from multimodal_jev.audio2tool_acceptance import check_container
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import atomic_json, file_hash

native = jev.native
JEV_PLAN = '60062a646a9aa20fb5d11ce05a8762f9d46612edae92cc51cabc7b016951e83a'
PREFLIGHT_INTENT = 'e6c8839e63fd766641f41f06ec5c3337aa6f6516067a126128a8bcd79126ea6b'
PREFLIGHT_RESULT = '01d64292294d6912a47cd46bf3abc3ad4e679320625c3bf6a13cddd926cba50f'
PREFLIGHT_ID = 'fdf7a9a736b0be8b40951908b886f1121c98898d199c7d9527fb8e421511aee1'


def dependency_ready(base):
    root = base/'audio2tool-jev-engine-capacity-v1'
    if file_hash(root/'queue-plan.json') != JEV_PLAN:
        raise ValueError('original Jev capacity queue required')
    launch = native.read(root/'evaluation-launch.json')
    if (launch['pid'] != 2932524 or launch['process_start'] != '291613588'
            or launch['queue_plan_sha256'] != JEV_PLAN):
        raise ValueError('original Jev supervisor identity required')
    terminal = root/'evaluation-terminal.json'
    if not terminal.exists():
        try: stat = Path(f"/proc/{launch['pid']}/stat").read_text().rsplit(')',1)[1].split()
        except FileNotFoundError: stat = []
        if len(stat)<20 or stat[19] != launch['process_start'] or stat[0] == 'Z':
            raise RuntimeError('Jev capacity supervisor missing/replaced/dead; inspect original attempt')
        return False
    if native.read(terminal)['exit_code'] != 0:
        raise RuntimeError('Jev capacity failed; no decode dispatch')
    jev.check_pins(root, JEV_PLAN)
    if not jev.native_ready(base/'audio2tool-engine-capacity-v1'):
        raise ValueError('complete accepted native capacity required')
    cid = native.read(root/'launch.json')['container_id']; item = native.inspect(cid)
    with tempfile.TemporaryDirectory(prefix='decode-capacity-') as tmp:
        path = Path(tmp)/'inspection.json'; write_once(path,item)
        accepted = jev.require_capacity(root,file_hash(root/'intent.json'),path,file_hash(path),cid,
            base/'audio2tool-jev-context-v1/run',jev.RELEASE,Path(jev.REGISTRY).read_bytes())
    prior = native.read(root/'acceptance.json')
    if any(accepted[k] != prior[k] for k in ('container_id','intent_sha256','result_sha256','source_commit')):
        raise ValueError('Jev capacity acceptance changed')
    return True


def check_pins(root, pin):
    if file_hash(root/'queue-plan.json') != pin: raise ValueError('registered decode queue changed')
    queue = native.read(root/'queue-plan.json')
    if (queue['schema'] != 'audio2tool-decode-queue-v1' or queue['automatic_retry'] is not False
            or queue['dependency_seconds'] != 350000 or queue['observation_seconds'] != 7500
            or queue['resource_lock'] != str(root.parent/'npu67-sampling-resource.lock')
            or file_hash(root/'preparation.json') != queue['preparation_sha256']):
        raise ValueError('fixed sequential decode queue required')
    for name, sha in queue['source_files'].items():
        if file_hash(root/'queue-source'/name) != sha: raise ValueError('coordinator source changed')
    prep = native.read(root/'preparation.json'); source = Path(prep['source_root']); prior = source.parent
    if (prior != root.parent/'audio2tool-decode-preflight-v1'
            or file_hash(prior/'intent.json') != PREFLIGHT_INTENT
            or file_hash(prior/'run/result.json') != PREFLIGHT_RESULT
            or prep['preflight_container_id'] != PREFLIGHT_ID
            or prep['physical_devices'] != [6,7] or prep['wall_seconds'] != 7200
            or prep['automatic_retry'] is not False or prep['weight_updates'] != 0):
        raise ValueError('original real decode preflight and frozen budget required')
    original = native.read(prior/'intent.json'); argv = original['argv']; image = prep['image']
    item = native.successful(PREFLIGHT_ID, image)
    if item['Config']['Cmd'] != argv[argv.index(image)+1:]: raise ValueError('actual preflight command differs')
    if prep['source_files'] != original['source_files'] or prep['source_commit'] != original['source_commit']:
        raise ValueError('profile must use the accepted preflight source')
    for name, sha in prep['source_files'].items():
        if file_hash(source/name) != sha: raise ValueError('accepted profile source changed')
    command = argv[argv.index(image)+1:]
    if command[-1] != '--preflight-only': raise ValueError('original preflight flag required')
    command = command[:-1]; command[2] = '--kill-after=30s'; command[3] = '7200s'
    if prep['argv'][prep['argv'].index(image)+1:] != command:
        raise ValueError('profile command must preserve all original boundary inputs')
    return queue, prep


def capture_capacities(root):
    proof = {}
    for name, directory in [('native','audio2tool-engine-capacity-v1'),('jev','audio2tool-jev-engine-capacity-v1')]:
        prior = root.parent/directory; cid = native.read(prior/'launch.json')['container_id']
        item = native.inspect(cid); path = root/(name+'-capacity-inspection.json'); write_once(path,item)
        proof[name] = dict(root=str(prior),container_id=cid,inspection=str(path),inspection_sha256=file_hash(path))
        if name == 'jev': proof[name]['intent_sha256'] = file_hash(prior/'intent.json')
    write_once(root/'capacity-evidence.json',proof)


def dispatch(root, prep):
    # Existing files make an interrupted preparation a visible non-replayable attempt.
    capture_capacities(root)
    argv = prep['argv'] + ['--capacity-evidence',str(root/'capacity-evidence.json'),
                          '--capacity-evidence-hash',file_hash(root/'capacity-evidence.json')]
    intent = dict(prep,argv=argv); write_once(root/'intent.json',intent)
    try: result = native.remote(argv,timeout=90)
    except jev.OBSERVATION_ERRORS as exc:
        write_once(root/'create-error.json',dict(error=repr(exc))); raise
    write_once(root/'create-result.json',dict(stdout=result.stdout,stderr=result.stderr))
    cid = result.stdout.strip()
    if not jev.re.fullmatch('[0-9a-f]{64}',cid): raise RuntimeError('ambiguous create; no replay')
    write_once(root/'launch.json',dict(container_id=cid,image=prep['image'],source_commit=prep['source_commit'],
                                     intent_sha256=file_hash(root/'intent.json'),physical_devices=[6,7]))
    item = native.inspect(cid)
    if item['Id'] != cid or item['Image'] != prep['image'] or item['State']['Status'] != 'created':
        raise ValueError('unexpected created profile container')
    write_once(root/'start-intent.json',dict(container_id=cid,started_unix=time.time(),automatic_retry=False))
    try: started = native.remote(['docker','start',cid],timeout=90)
    except jev.OBSERVATION_ERRORS as exc:
        write_once(root/'start-error.json',dict(error=repr(exc),container_id=cid)); raise
    write_once(root/'start-result.json',dict(stdout=started.stdout,stderr=started.stderr))
    return cid


def execute(root,pin):
    if any((root/n).exists() for n in ('intent.json','capacity-evidence.json','native-capacity-inspection.json',
                                      'launch.json','run','evaluation-terminal.json')):
        raise FileExistsError('previous profile attempt; no replay')
    queue,prep = check_pins(root,pin); deadline = time.monotonic()+queue['dependency_seconds']
    while True:
        try:
            ready = dependency_ready(root.parent)
            atomic_json(root/'queue-progress.json',dict(phase='waiting_both_capacities',ready=ready,heartbeat_unix=time.time()))
            if ready: break
        except jev.OBSERVATION_ERRORS as exc:
            atomic_json(root/'queue-progress.json',dict(phase='dependency_observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        if time.monotonic()>deadline: raise TimeoutError('dependency observation expired; no dispatch')
        time.sleep(20)
    with Path(queue['resource_lock']).open('a') as lock:
        while True:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: locked=False
            else: locked=True
            try: free=locked and jev.free_devices(root)
            except jev.OBSERVATION_ERRORS: free=False
            if free: break
            if locked: fcntl.flock(lock,fcntl.LOCK_UN)
            atomic_json(root/'queue-progress.json',dict(phase='waiting_npu67',heartbeat_unix=time.time()))
            if time.monotonic()>deadline: raise TimeoutError('resource observation expired; no dispatch')
            time.sleep(20)
        queue,prep=check_pins(root,pin)
        if not dependency_ready(root.parent): raise ValueError('capacity dependencies changed')
        cid=dispatch(root,prep); deadline=time.monotonic()+queue['observation_seconds']
        while True:
            try:
                item=native.inspect(cid)
                if item['Id']!=cid or item['Image']!=prep['image']: raise ValueError('profile identity changed')
                atomic_json(root/'queue-progress.json',dict(phase='profile_running',container_id=cid,state=item['State'],heartbeat_unix=time.time()))
                if item['State']['Status'] in ('exited','dead'): break
            except jev.OBSERVATION_ERRORS as exc:
                atomic_json(root/'queue-progress.json',dict(phase='observation_retry',container_id=cid,error_type=type(exc).__name__,heartbeat_unix=time.time()))
            if time.monotonic()>deadline: raise TimeoutError('observation expired; inspect same ID, never replay')
            time.sleep(20)
        write_once(root/'profile-inspection.json',item)
        check_container(item,native.read(root/'intent.json'),cid)
        result=native.read(root/'run/result.json')
        if (native.read(root/'run/terminal.json')['exit_code']!=0 or result['status']!='profile_complete'
                or result['model_loaded'] is not True or len(result['records'])!=16
                or result['weight_updates']!=0 or result['targets_read'] is not False):
            raise ValueError('complete frozen profile execution required')
        # Artifact/tokenizer and capacity provenance audit remains independent.
        write_once(root/'execution-complete.json',dict(container_id=cid,result_sha256=file_hash(root/'run/result.json'),
            accepted=False,scope='Successful execution only; independent full output/tokenizer audit pending.'))
        atomic_json(root/'queue-progress.json',dict(phase='execution_complete_pending_audit',container_id=cid,heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-hash',required=True)
    a=p.parse_args()
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'evaluation-terminal.json').exists(): raise FileExistsError('preserve terminal')
        try: execute(a.root,a.plan_hash)
        except BaseException as exc:
            write_once(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        write_once(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),automatic_retry=False))
