"""Wait for formal launch and three committed updates using file events.

No model calls, polling loop, restarts or changes to training. The remote
observer opens inotify on the declared run directory and checks exact Docker
identity before reporting. Ordinary training observations stay at 900 seconds.
"""
import argparse
import ctypes
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
        if path.is_dir() and str(path) not in self.paths:
            if self.lib.inotify_add_watch(self.fd,os.fsencode(path),0x100|0x80|0x8)<0:raise OSError('watch failed')
            self.paths.add(str(path))
    def wait(self):
        self.poll.poll(900000)
        try:os.read(self.fd,65536)
        except BlockingIOError:pass
    def close(self):os.close(self.fd)


def remote_watch(c):
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
            if handle.exists():break
            watch.wait()
        launch=json.loads(handle.read_text());commit_json(root/'launch.json',launch)
        remote_c=dict(container_id=launch['container_id'],image=launch['image'],remote_root=launch['remote_root'],
                      world_size=2,replicas=2,count=3)
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
