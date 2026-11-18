"""Read exact macro-training handles at most once per 900 seconds.

No model calls, restart, cleanup, partial collection or policy changes.
Failed remote observations also consume the interval.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.local_config import load_config
from tools.reports.context_macro_training_operations import validate, command
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.launch_context_batch_probe_candidate import write_new

INTERVAL = 900


def observe(c, *, now=None, remote=None):
    accepted=validate(c);base=Path(c['operations_root'])
    attempt=base/'attempt.json'
    if not attempt.is_file():raise ValueError('existing launched attempt required')
    def ssh(args):
        r=subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
            capture_output=True,text=True,timeout=55)
        if r.returncode:raise RuntimeError(r.stderr[-8000:])
        return r.stdout
    remote=remote or ssh
    observations=base/'observations';observations.mkdir(exist_ok=True)
    with (observations/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        history=[json.loads(p.read_text()) for p in observations.glob('*-attempt.json')]
        if any(r['files_sha256']!=accepted['files_sha256'] for r in history):
            raise ValueError('observation belongs to another package')
        last=max([attempt.stat().st_mtime]+[r['observed_unix'] for r in history])
        stamp=time.time() if now is None else now
        if stamp<last+INTERVAL:
            return dict(status='not_due_no_remote_call',next_ordinary_observation_unix=last+INTERVAL)
        record=dict(observed_unix=stamp,next_ordinary_observation_unix=stamp+INTERVAL,
            files_sha256=accepted['files_sha256'],containers={},scope='read-only exact attempt; not full training audit')
        stem=str(time.time_ns());write_new(observations/(stem+'-attempt.json'),record)
        try:
            for role in ('preflight','service-0','service-1','client'):
                path=base/(role+'-launch.json')
                if not path.exists():
                    record['containers'][role]=dict(status='not_created');continue
                h=json.loads(path.read_text())
                if h['argv']!=command(c,role):raise ValueError('actual launch differs')
                obs=json.loads(remote(['docker','inspect',h['container_id']]))[0]
                inspect_runtime(obs,h)
                record['containers'][role]=dict(id=obs['Id'],state=obs['State'])
            code="""import json
from pathlib import Path
root=Path(ROOT)/'run';ranks=[]
for rank in range(2):
 paths=sorted((root/f'rank-{rank}').glob('update-*.json'))
 if not paths:ranks.append(dict(rank=rank,completed_updates=0));continue
 p=paths[-1];v=json.loads(p.read_text())
 ranks.append(dict(rank=rank,completed_updates=v['update'],**{k:v[k] for k in ('real_queries','total_seconds','actor_seconds','preparation_seconds','peak_allocated','next_policy_id')}))
print(json.dumps(dict(ranks=ranks,committed_checkpoints=len(list(root.glob('checkpoint-*-committed.json'))))))
""".replace('ROOT',repr(c['remote_root']))
            record['progress']=json.loads(remote(['python3','-B','-c',code]))
            record['status']='observed_exact_attempt'
        except Exception as exc:
            record.update(status='observation_failed_preserve_handles',error=repr(exc))
        write_new(observations/(stem+'-result.json'),record)
        return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local)
    print(json.dumps(observe(c),indent=2))
