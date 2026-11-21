"""Dispatch a single sealed dataset to its assigned host, with exact Docker handles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.launch_context_batch_probe_candidate import wait_exact
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.finalize_context_infonce import wait_exact as wait_driver
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.baseline import file_hash


def run_group(group, launch, failed):
    """A failed lane blocks only its own dependent phases."""
    ready = [job for job in group if job.get('lane') not in failed]
    if not ready:
        return
    with ThreadPoolExecutor(max_workers=len(ready)) as pool:
        pending = [(job, pool.submit(launch, job)) for job in ready]
        for job, future in pending:
            try:
                future.result()
            except Exception as exc:
                if 'lane' not in job:
                    raise
                failed[job['lane']] = dict(stage=job['name'], error=repr(exc))


def wait_for_devices(predecessors, checked, remote, record):
    """Wait for declared Docker owners before taking the evaluation lease."""
    ids = [job['container_id'] for job in predecessors]
    if (len(set(ids)) != len(ids) or any(
            set(job) != {'container_id', 'name', 'image', 'devices'}
            or not re.fullmatch('[a-f0-9]{64}', job['container_id'])
            or not job['devices'] or len(set(job['devices'])) != len(job['devices'])
            or any(type(i) is not int or i < 0 for i in job['devices'])
            for job in predecessors)):
        raise ValueError('exact predecessor containers and device bindings required')

    def wait(job):
        cid = job['container_id']
        def inspect():
            obj, = json.loads(checked(['docker', 'inspect', cid]))
            devices = {d['PathOnHost'] for d in obj['HostConfig']['Devices'] or []
                       if re.fullmatch(r'/dev/davinci[0-9]+', d['PathOnHost'])}
            if (obj['Id'] != cid or obj['Name'] != job['name'] or obj['Image'] != job['image']
                    or devices != {f'/dev/davinci{i}' for i in job['devices']}):
                raise ValueError('predecessor container identity or device mapping differs')
            return obj['State']
        initial = inspect()
        if initial['Status'] not in ('running', 'exited'):
            raise ValueError('predecessor must be running or exited')
        record(cid+'-waiting.json', job)
        code = wait_exact(remote, cid)
        state = inspect()
        if state['Status'] != 'exited' or state['Running'] or state['ExitCode'] != code:
            raise ValueError('predecessor has not released its devices')
        record(cid+'-released.json', dict(container_id=cid, state=state))

    with ThreadPoolExecutor(max_workers=len(predecessors)) as pool:
        for result in pool.map(wait, predecessors):
            pass


def run(c):
    root = Path(c['operations'])
    check_paths({'evaluation operations': root})
    root.mkdir(parents=True, exist_ok=False)
    host, remote_root = c['host'], c['remote_root']
    if (not re.fullmatch(r'[\w.@-]+', host) or host.startswith('-')
            or not re.fullmatch(r'/[\w./-]+', remote_root) or '..' in Path(remote_root).parts):
        raise ValueError('explicit host and isolated absolute destination required')
    group_lease = c.get('device_group_lease', False)
    wait = c.get('wait_for_lease', False)
    predecessors = c.get('device_predecessors', [])
    if type(group_lease) is not bool or type(wait) is not bool:
        raise ValueError('explicit Boolean lease options required')
    if not isinstance(predecessors, list) or (predecessors and not group_lease):
        raise ValueError('container waits require device group leasing')
    if group_lease:
        if len(c['chains']) != 1:
            raise ValueError('device group leasing requires one ordered chain')
        for group in c['chains'][0]:
            scope = [i for job in group for i in job['devices']]
            lanes = [job['lane'] for job in group if 'lane' in job]
            if (not group or len(scope) != len(set(scope)) or
                    len(lanes) != len(set(lanes)) or len(lanes) not in (0, len(group)) or
                    any(bool(job['devices']) != bool(scope) or 'wait_for' in job or 'after_driver' in job
                        for job in group)):
                raise ValueError('device groups must be disjoint; CPU phases and waits must be separate')
    elif any('lane' in job for chain in c['chains'] for group in chain for job in group):
        raise ValueError('independent failure lanes require device group leasing')

    @contextmanager
    def lease():
        with Path(c['lease']).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            yield

    def remote(args, bounded=True):
        return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],
                              text=True, capture_output=True, timeout=55 if bounded else None)

    def checked(args):
        r = remote(args)
        if r.returncode:
            raise RuntimeError(r.stderr or r.stdout)
        return r.stdout

    def launch(job):
        role, devices = job['name'], job['devices']
        if 'after_driver' in job:
            prior = job['after_driver']
            wait_driver(prior['driver'], Path(prior['terminal']))
        if 'wait_for' in job:
            prerequisite = job['wait_for']
            deadline = time.monotonic()+c['dependency_timeout']
            while not Path(prerequisite).is_file():
                terminal = Path(c['predecessor_terminal'])
                if terminal.is_file() or time.monotonic() >= deadline:
                    raise ValueError('prerequisite did not produce '+prerequisite)
                time.sleep(c['dependency_check_seconds'])
            if json.loads(Path(prerequisite).read_text()) != {'exit_code':0}:
                raise ValueError('prerequisite failed: '+prerequisite)
        if devices:
            required_idle = c.get('idle_devices',devices)
            smi = checked(['npu-smi','info'])
            if any(f'No running processes found in NPU {i}' not in smi for i in required_idle):
                raise ValueError('assigned devices are busy: '+str(required_idle))
            active = checked(['docker','ps','-q']).split()
            if active:
                for obj in json.loads(checked(['docker','inspect',*active])):
                    occupied = {d['PathOnHost'] for d in obj['HostConfig']['Devices'] or []}
                    if occupied & {f'/dev/davinci{i}' for i in required_idle}:
                        raise ValueError('assigned devices are mapped by '+obj['Name'])
            if 'entry_pins' in c:
                if checked(['docker','image','inspect','--format','{{.Id}}',c['image']]).strip()!=c['image']:
                    raise ValueError('frozen runtime image differs')
                paths = {remote_root+'/entry/'+p:h for p,h in c['entry_pins'].items()}
                remote_pins = {line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0]
                               for line in checked(['sha256sum',*paths]).splitlines()}
                if remote_pins != paths:
                    raise ValueError('remote frozen entry changed')
                commit_json(root/(role+'-admission.json'),dict(entry_pins=c['entry_pins'],idle_devices=required_idle))
        argv = ['docker','create','--pull=never','--network=none','--name',c['tag']+'-'+role,
                '--cpus',str(job['cpus']),'--memory',job['memory'],'--shm-size',job['shm'],
                '--log-driver=json-file','--log-opt','max-size=100m','--log-opt','max-file=3']
        if devices:
            for name in [*(f'davinci{i}' for i in devices), 'davinci_manager','devmm_svm','hisi_hdc']:
                argv += ['--device','/dev/'+name+':/dev/'+name]
        for mount in (remote_root+':/jev-work',remote_root+'/entry:/jev-entry:ro',
                      remote_root+'/'+role+'-tmp:/jev-tmp',remote_root+'/'+role+'-home:/root',
                      c['shared_root']+':'+c['shared_root']+':ro',c['driver_root']+':/usr/local/Ascend/driver:ro'):
            argv += ['-v',mount]
        for mount in c.get('read_only_mounts', []):
            argv += ['-v',mount+':ro']
        env = dict(PYTHONPATH='/jev-entry/source/src:/jev-entry/source:/jev-entry/source/tools/jev',
                   PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
        for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                    'ASCEND_WORK_PATH','ASCEND_CACHE_PATH','VLLM_RPC_BASE_PATH'):
            env[key] = '/jev-tmp'
        for k,v in env.items():
            argv += ['-e',k+'='+v]
        argv += ['--entrypoint',c['entrypoint'],c['image'],'python','-B','-m','tools.reports.native_seal_container',
                 '--config','/jev-entry/native-seal-container.json','--local','/jev-entry/'+role+'.json']
        commit_json(root/(role+'-intent.json'), dict(argv=argv))
        cid = checked(argv).strip()
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise ValueError('exact container ID missing')
        h = dict(container_id=cid, argv=argv, host=host, remote_root=remote_root)
        commit_json(root/(role+'-launch.json'), h)
        inspect_runtime(json.loads(checked(['docker','inspect',cid]))[0], h)
        checked(['docker','start',cid])
        code = wait_exact(remote, cid)
        observation = json.loads(checked(['docker','inspect',cid]))[0]
        commit_json(root/(role+'-inspect.json'), observation)
        logs = remote(['docker','logs',cid])
        (root/(role+'.log')).write_text(logs.stdout+logs.stderr)
        commit_json(root/(role+'-wait.json'), dict(exit_code=code))
        inspect_runtime(observation, h, terminal=True)
        if 'collect' in job:
            source = remote_root+'/'+job['collect']
            raw = checked(['cat',source])
            payload = json.loads(raw)
            commit_json(root/(role+'-collected.json'),payload)

    try:
        if 'predecessor' in c:
            p = c['predecessor']
            status = p.get('required_status', 'declared_stages_finished')
            if not isinstance(status, str) or not status:
                raise ValueError('explicit predecessor success status required')
            commit_json(root/'waiting.json', p)
            wait_driver(p['driver'], Path(p['terminal']))
            terminal = json.loads(Path(p['terminal']).read_text())
            if (type(terminal.get('exit_code')) is not int or terminal['exit_code'] != 0
                    or terminal.get('status') != status):
                raise ValueError('exact prerequisite preparation did not finish successfully')
        with nullcontext() if group_lease else lease():
            checked(['mkdir','-p',str(Path(remote_root).parent)])
            checked(['mkdir',remote_root])
            roles = [j['name'] for chain in c['chains'] for group in chain for j in group]
            checked(['mkdir',*(remote_root+'/'+r+s for r in roles for s in ('-tmp','-home'))])
            if 'entry_archive_sha256' in c and file_hash(c['archive'])!=c['entry_archive_sha256']:
                raise ValueError('frozen entry archive changed')
            subprocess.run(['scp',c['archive'],host+':'+remote_root+'/entry.tar'],check=True,timeout=c['transfer_timeout'])
            if ('entry_archive_sha256' in c and
                    checked(['sha256sum',remote_root+'/entry.tar']).split()[0]!=c['entry_archive_sha256']):
                raise ValueError('transferred entry archive changed')
            checked(['tar','-xf',remote_root+'/entry.tar','-C',remote_root])
            if predecessors:
                wait_for_devices(predecessors, checked, remote,
                                 lambda name, value: commit_json(root/name, value))
            def sequence(chain):
                failed = {}
                for group in chain:
                    needs_cards = group_lease and any(j['devices'] and j.get('lane') not in failed for j in group)
                    with lease() if needs_cards else nullcontext():
                        run_group(group, launch, failed)
                return failed
            with ThreadPoolExecutor(max_workers=len(c['chains'])) as pool:
                failures = [future.result() for future in [pool.submit(sequence, chain) for chain in c['chains']]]
        if any(failures):
            commit_json(root/'terminal.json', dict(exit_code=1, status='evaluation_lanes_failed',
                                                 failures=failures[0], restart=False))
            return 1
        commit_json(root/'terminal.json', dict(exit_code=0, status='declared_stages_finished'))
        return 0
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    c, _ = load_config(a.config, a.local)
    if a.detach:
        root = Path(c['operations'])
        argv = [sys.executable,'-B','-m',__spec__.name,'--config',a.config,'--local',a.local]
        with root.with_suffix('.log').open('x') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        record = dict(pid=child.pid, argv=argv, started_unix=time.time(),
                      process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
        commit_json(root.with_suffix('.started.json'), record)
        print(json.dumps(record))
    else:
        return run(c)


if __name__ == '__main__':
    sys.exit(main())
