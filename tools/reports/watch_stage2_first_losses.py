"""Wait for formal launch and three committed updates using file events.

No model calls, polling loop, restarts or changes to training. The remote
observer opens inotify on the declared run directory and checks exact Docker
identity before reporting. Ordinary training observations stay at 900 seconds.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import time

def activation(receipt,policy,replicas):
    if len(receipt.get('replica_receipts',[]))!=replicas:
        raise ValueError('complete replica activation required')
    for r in [receipt,*receipt['replica_receipts']]:
        if r.get('policy_id')!=policy or r.get('head_marker_exact')is not True or r.get('native_loaded_tensors')!=112:
            raise ValueError('complete adapter and head activation required')


def formal_topology(controller, launch):
    window=json.loads((Path(controller)/'formal-window.json').read_text())
    deployment=window['deployment']
    package=Path(deployment['package'])
    manifest_bytes=(package/'PREPARED.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    config_bytes=(package/'config.json').read_bytes()
    config=json.loads(config_bytes)
    if (window['window']!='formal' or manifest['view_window']!='formal'
            or hashlib.sha256(manifest_bytes).hexdigest()!=launch['manifest_sha256']
            or deployment['manifest_sha256']!=launch['manifest_sha256']
            or deployment['host']!=launch['host'] or manifest['remote_root']!=launch['remote_root']
            or hashlib.sha256(config_bytes).hexdigest()!=manifest['files']['config.json']
            or manifest['files']['config.json']!=launch['config_sha256']
            or config['training_dispatch_allowed'] is not True):
        raise ValueError('require the frozen formal package bound to the launched container')
    train=manifest['training_devices']; services=manifest['inference_services']
    devices=[*train,*(s['device'] for s in services)]
    if (not train or not services or any(type(i) is not int or i<0 for i in devices)
            or len(set(devices))!=len(devices)
            or config['view_lineage']['topology']!=dict(training_devices=train,inference_services=services)):
        raise ValueError('invalid formal training and inference topology')
    return dict(world_size=len(train),replicas=len(services))


def first_losses(root,*,count=3,world_size=2,replicas=2):
    root=Path(root);rows=[];previous=None
    for update in range(1,count+1):
        ranks=[json.loads((root/f'rank-{r}/update-{update:04d}.json').read_text()) for r in range(world_size)]
        ack=json.loads((root/f'activation-{update:04d}.json').read_text())
        policy=ranks[0]['next_policy_id'];behavior=ranks[0]['rollout']['behavior_policy_id']
        if ack['error'] is not None or ack['policy_id']!=policy:raise ValueError('successful activation required')
        activation(ack['receipt'],policy,replicas)
        if previous is not None and previous!=behavior:raise ValueError('nonconsecutive behavior policies')
        if any(r['update']!=update or r['next_policy_id']!=policy or r['rollout']!=ranks[0]['rollout']
               or r['real_queries']!=ranks[0]['real_queries'] for r in ranks):raise ValueError('matching committed rank outputs required')
        totals={}
        for name in ('jev/context_loss','jev/dual_rl','jev/dual_auxiliary','jev/dual_kl_unweighted'):
            values=[]
            for r in ranks:
                parts=r['metrics'][name]
                if not isinstance(parts,list) or not parts or any(type(x)not in (int,float) or not math.isfinite(x) for x in parts):
                    raise ValueError('complete finite microbatch losses required')
                values.append(sum(parts))
            totals[name]=sum(values)/world_size
        for r in ranks:
            if abs(sum(r['metrics']['jev/context_loss'])-r['metrics']['actor/pg_loss'])>1e-6:
                raise ValueError('aggregated actor loss differs from microbatch trace')
        rows.append(dict(update=update,real_queries=ranks[0]['real_queries'],losses=totals,
                         behavior_policy_id=behavior,next_policy_id=policy))
        previous=policy
    return rows


class Events:
    def __init__(self):
        self.lib=ctypes.CDLL(None,use_errno=True);self.fd=self.lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify unavailable')
        self.paths=set();self.poll=select.poll();self.poll.register(self.fd,select.POLLIN)
    def add(self,path):
        path=Path(path)
        if str(path) in self.paths:
            return
        # A pending launch may not have created its operations tree yet.
        if not path.is_dir() and path.parent != path:
            self.add(path.parent)
        if path.is_dir():
            if self.lib.inotify_add_watch(self.fd,os.fsencode(path),0x100|0x80|0x8)<0:raise OSError('watch failed')
            self.paths.add(str(path))
    def wait(self):
        self.poll.poll(900000)
        try:os.read(self.fd,65536)
        except BlockingIOError:pass
    def close(self):os.close(self.fd)


def tpo_watch(c):
    root=Path(c['remote_root'])
    watch=Events()
    waiter=None; pidfd=None
    def inspect():
        state=json.loads(subprocess.check_output(['docker','inspect',c['container_id']],text=True))[0]
        if (state['Id']!=c['container_id'] or state['Name']!='/'+c['name'] or state['Image']!=c['image']
                or state['Args'][-1]!='/jev-entry/tpo.json'):
            raise ValueError('incorrect formal TPO container')
        if 'config_files' in c:
            for name,sha in c['config_files'].items():
                if hashlib.sha256((root.parent/'entry'/name).read_bytes()).hexdigest()!=sha:
                    raise ValueError('actual formal configuration differs: '+name)
            devices={d['PathOnHost'] for d in state['HostConfig']['Devices']}
            if {d for d in devices if d.removeprefix('/dev/davinci').isdigit()}!={f'/dev/davinci{i}' for i in c['devices']}:
                raise ValueError('actual formal device map differs')
        return state
    try:
        state=inspect()
        # docker wait returns immediately on a merely created handle. Wait for
        # the first output directory event before arming the termination fd.
        while True:
            watch.add(root)
            if (root/'update-0003.json').is_file():
                state=inspect()
                updates=[json.loads((root/f'update-{i:04d}.json').read_bytes()) for i in range(1,4)]
                return dict(status=state['State']['Status'],updates=updates,observed_unix=time.time(),
                            state=state['State'])
            if waiter is None and (root.is_dir() or state['State']['Running']):
                waiter=subprocess.Popen(['docker','wait',c['container_id']],stdout=subprocess.DEVNULL)
                pidfd=os.pidfd_open(waiter.pid)
                watch.poll.register(pidfd,select.POLLIN)
            if state['State']['Status'] in ('exited','dead') or (waiter is not None and waiter.poll() is not None):
                raise ValueError('formal container ended before third update')
            watch.wait()
    finally:
        if pidfd is not None:
            os.close(pidfd)
        if waiter is not None:
            if waiter.poll() is None:
                waiter.terminate()
            waiter.wait()
        watch.close()


def remote_watch(c):
    if c.get('format')=='tpo':
        return tpo_watch(c)
    root=Path(c['remote_root']);watch=Events()
    try:
        while True:
            watch.add(root);watch.add(root/'run');watch.add(root/'supervision')
            for r in range(c['world_size']):watch.add(root/f'run/rank-{r}')
            try:rows=first_losses(root/'run',count=c['count'],world_size=c['world_size'],replicas=c['replicas'])
            except FileNotFoundError:rows=None
            state=json.loads(subprocess.check_output(['docker','inspect',c['container_id']],text=True))[0]
            if state['Id']!=c['container_id'] or state['Image']!=c['image']:raise ValueError('exact formal container required')
            if rows is not None:return dict(status='formal_first_losses_committed',updates=rows,state=state['State'],container_id=c['container_id'])
            if state['State']['Status'] in ('exited','dead'):raise ValueError('formal container ended before three committed updates')
            watch.wait()
    finally:watch.close()


def run(c):
    from multimodal_jev.local_checkpoint import commit_json
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False);commit_json(root/'config.json',c)
    controller=Path(c['controller']);operations=Path(c['formal_operations']);watch=Events()
    try:
        handle=operations/(c['formal_tag']+'-launch.json')
        while True:
            watch.add(controller);watch.add(operations);watch.add(operations/'finalization');watch.add(operations.parent)
            terminal=controller/'terminal.json'
            if terminal.exists() and json.loads(terminal.read_text())['exit_code']:
                raise ValueError('controller failed before formal three-update milestone: '+terminal.read_text())
            try:
                launch = json.loads(handle.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # The launcher writes its read-only handle in place. A create
                # event can precede the close event that completes the JSON.
                watch.wait()
            else:
                break
        commit_json(root/'launch.json',launch)
        if c.get('require_view_topology') or (controller/'formal-window.json').is_file():
            topology=formal_topology(controller,launch)
        else:
            topology=dict(world_size=2,replicas=2)
        remote_c=dict(container_id=launch['container_id'],image=launch['image'],remote_root=launch['remote_root'],
                      count=3,**topology)
        # Run the frozen observer source from stdin. The host needs only JSON,
        # ctypes and filesystem access; no ML framework or model is loaded.
        payload=Path(__file__).read_text()
        command=shlex.join(['python3','-','--remote-json',json.dumps(remote_c)])
        result=subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],command],input=payload,text=True,capture_output=True)
        (root/'remote.stderr').write_text(result.stderr)
        if result.returncode:raise ValueError('formal milestone observer failed: '+result.stderr[-3000:])
        report=json.loads(result.stdout);commit_json(root/'report.json',report)
        commit_json(root/'terminal.json',dict(exit_code=0,status=report['status']))
    except BaseException as error:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(error),restart=False));raise
    finally:watch.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--local');p.add_argument('--remote-json')
    a=p.parse_args()
    if a.remote_json:print(json.dumps(remote_watch(json.loads(a.remote_json))))
    else:
        from multimodal_jev.local_config import load_config
        c,_=load_config(a.config,a.local);run(c)
