"""Read-only deterministic startup observer; emits once per completed phase.

Local phase checks do not query remote hardware. Once formal launch is recorded,
Linux inotify waits for rank update files; no per-minute remote status polling.
"""
import argparse,json,subprocess,shlex,time
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json

REMOTE = r'''
import ctypes,json,os,select,time,math
from pathlib import Path
c=CONFIG
root=Path(c['root']);fd=ctypes.CDLL(None).inotify_init1(os.O_NONBLOCK)
libc=ctypes.CDLL(None);watched=set();deadline=time.monotonic()+c['timeout']
try:
 while time.monotonic()<deadline:
  for p in [root,root/'run',*[root/'run'/f'rank-{r}' for r in range(c['world'])]]:
   if p.is_dir() and str(p) not in watched:
    if libc.inotify_add_watch(fd,str(p).encode(),0x100|0x80|0x8)>=0:watched.add(str(p))
  start=c.get('start_update',0)
  names=[root/'run'/f'rank-{r}'/f'update-{s:04d}.json' for s in range(start+1,start+c['updates']+1) for r in range(c['world'])]
  if all(p.is_file() for p in names):
   records=[json.loads(p.read_text()) for p in names]
   if any(not all(math.isfinite(float(x)) for v in rec['metrics'].values() for x in (v if isinstance(v,list) else [v]) if isinstance(x,(int,float))) for rec in records):raise ValueError('nonfinite metrics')
   print(json.dumps(dict(status='formal_first_updates_observed',records=records)));break
  terminal=root/'supervision/terminal.json'
  if terminal.is_file():raise ValueError('trainer exited before required updates: '+terminal.read_text())
  ready,_,_=select.select([fd],[],[],min(60,max(0,deadline-time.monotonic())))
  if ready:
   try:os.read(fd,1024*1024)
   except BlockingIOError:pass
 else:raise TimeoutError('startup observation deadline')
finally:os.close(fd)
'''

def run(c):
    from multimodal_jev.context_topology_batch_plan import derive
    from multimodal_jev.baseline import file_hash
    settings=c; c=settings['pipeline']
    if any(type(settings[k]) is not int or settings[k]<1 for k in ('updates','timeout_seconds')):
        raise ValueError('positive observation window required')
    out=Path(c['observation_output']);out.mkdir(parents=True,exist_ok=False)
    prior=None
    while True:
        root=Path(c['output']);p=root/'progress.json'
        if p.exists():
            progress=json.loads(p.read_text())
            if progress['stage']!=prior:print(json.dumps(progress),flush=True);prior=progress['stage']
        terminal=root/'terminal.json'
        if terminal.exists() and json.loads(terminal.read_text())['exit_code']:
            raise ValueError('pipeline failed: '+terminal.read_text())
        ops=c['formal'];handle=Path(ops['operations_root'])/(ops['tag']+'-launch.json')
        if handle.exists():break
        time.sleep(5)
    launch=json.loads(handle.read_text());package=Path(ops['package'])
    manifest=json.loads((package/'PREPARED.json').read_bytes())
    if (file_hash(package/'PREPARED.json')!=launch['manifest_sha256']
            or file_hash(package/'config.json')!=manifest['files']['config.json']):
        raise ValueError('formal source changed')
    training=json.loads((package/'config.json').read_bytes())
    spec=dict(root=launch['remote_root'],world=derive(training['batch_config'])['dp_replicas'],
              updates=settings['updates'],timeout=settings['timeout_seconds'])
    raw=subprocess.check_output(['ssh',c['deployment']['host'],shlex.join(['python3','-c',REMOTE.replace('CONFIG',repr(spec),1)])],text=True)
    observed=json.loads(subprocess.check_output(['ssh',c['deployment']['host'],shlex.join(['docker','inspect',launch['container_id']])],text=True))[0]
    result=json.loads(raw);result.update(container_id=launch['container_id'],image=observed['Image'],state=observed['State'],observed_unix=time.time())
    commit_json(out/'first-updates.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True);a=p.parse_args();c,_=load_config(a.config,a.local);run(c)
