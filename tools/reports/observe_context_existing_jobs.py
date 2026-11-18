"""One-shot, read-only observation of original handles with a 900-second guard.

No daemon, dispatch, restart, lease transfer, or collection ownership changes.
Attempts (including timeout) consume the ordinary-observation interval.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path('.local/decision-objective-research')
INTERVAL = 900
TARGETS = {
    'grpo': ('context-rl-grpo-arithmetic-v2-launch.json',
             'docs/engineering/context-arithmetic-formal-live-v*.json'),
    'inputs': ('context-attribute-population-cpu-v1-launch.json',
               str(BASE/'context-attribute-population-cpu-v1-live-v*.json')),
}


def latest_time(paths, container_id, image):
    times = []
    for path in paths:
        value = json.loads(Path(path).read_text())
        if value.get('container_id') != container_id: continue
        # Early receipts omitted image. Their exact container timestamp still
        # throttles observations; it does not establish image provenance.
        if value.get('image') is not None and value['image'] != image:
            raise ValueError('observation image identity differs')
        stamp = value['observed_unix']
        if type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError('invalid observation time')
        times.append(stamp)
    if not times: raise ValueError('original observation history required')
    return max(times)


def ssh(args):
    return subprocess.check_output(['ssh','-o','ConnectTimeout=10','npu',shlex.join(args)],
                                   text=True,timeout=55)


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def observe(target, *, remote=ssh, now=None):
    launch_name, history_glob = TARGETS[target]
    launch = json.loads((BASE/launch_name).read_text())
    root = BASE/'ordinary-observations-v1';root.mkdir(exist_ok=True)
    with (root/(target+'.lock')).open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        history = list(Path('.').glob(history_glob)) + list(root.glob(target+'-*-attempt.json'))
        last = latest_time(history,launch['container_id'],launch['image'])
        stamp = time.time() if now is None else now
        if stamp < last+INTERVAL:
            return dict(target=target,status='not_due_no_remote_call',next_ordinary_observation_unix=last+INTERVAL)
        record = dict(target=target,container_id=launch['container_id'],image=launch['image'],
            observed_unix=stamp,next_ordinary_observation_unix=stamp+INTERVAL,
            scope='read_only_original_handle; no restart or dispatch')
        stem = target+'-'+str(time.time_ns())
        # Persist before remote I/O; loss of SSH output is not a terminal state.
        write_new(root/(stem+'-attempt.json'),record)
        try:
            inspected = json.loads(remote(['docker','inspect',launch['container_id']]))[0]
            if inspected['Id']!=launch['container_id'] or inspected['Image']!=launch['image']:
                raise ValueError('original container/image identity differs')
            record['state']=inspected['State']
            if inspected['State']['Running']:
                if target=='grpo':
                    code="""import json
from pathlib import Path
r=Path(ROOT)/'run';ranks=[]
for rank in range(4):
 paths=sorted((r/f'rank-{rank}').glob('update-*.json'))
 if not paths:
  ranks.append(dict(rank=rank,completed_updates=0));continue
 p=paths[-1];d=json.loads(p.read_text())
 ranks.append(dict(rank=rank,file=p.name,total_seconds=d['total_seconds'],actor_seconds=d['actor_seconds'],preparation_seconds=d['preparation_seconds']))
print(json.dumps(dict(ranks=ranks,commits=sorted(p.name for p in r.glob('checkpoint-*-committed.json')))))
""".replace('ROOT',repr(launch['remote_root']))
                else:
                    code='from pathlib import Path; print(Path('+repr(launch['remote_root']+'/run/progress.json')+').read_text())'
                record['progress']=json.loads(remote(['docker','exec',launch['container_id'],'python','-c',code]))
            record['status']='observed_original_handle'
        except Exception as exc:
            record.update(status='observation_failed_preserve_original_handle',error=repr(exc))
        write_new(root/(stem+'-result.json'),record)
        return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=TARGETS,required=True)
    args=parser.parse_args();print(json.dumps(observe(args.target),indent=2))
