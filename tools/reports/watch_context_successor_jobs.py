"""Read-only 900-second observer for the existing, finite successor queue.

Does not cover GRPO training/inputs (their original observer owns them), launch
work, restart containers, collect results, acquire device leases, or tune TEST.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

from tools.reports.observe_context_existing_jobs import BASE, INTERVAL, ssh, write_new

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
REMOTE = '/data/xingchao/multimodal-jev/simple-final-v1/'
TARGETS = {'context-rl-rloo-arithmetic-v2': 'training'}
for algorithm in ('grpo', 'rloo'):
    for step in (0, *range(64, 634, 64), 634):
        TARGETS[f'context-rl-eval-{algorithm}-arithmetic-v2-step{step:04d}'] = 'selection'
TARGETS.update({'fsc-context-pair-arithmetic-v2': 'fsc',
                'slurp-context-test-pair-arithmetic-v2': 'test',
                'minds-context-pair-arithmetic-v1': 'minds'})


def progress_code(kind, remote):
    if kind == 'training':
        body = """r=root/'run';rows=[]
for rank in range(4):
 paths=sorted((r/f'rank-{rank}').glob('update-*.json'))
 if paths:
  p=paths[-1];d=json.loads(p.read_text())
  rows.append(dict(rank=rank,file=p.name,**{k:d[k] for k in ('total_seconds','preparation_seconds','actor_seconds')}))
 else:rows.append(dict(rank=rank,completed_updates=0))
print(json.dumps(dict(ranks=rows,commits=sorted(p.name for p in r.glob('checkpoint-*-committed.json')))))
"""
    else:
        paths = (['evaluation/selection/progress.json'] if kind == 'selection' else
                 [f'evaluation/{role}/' + ('inference/' if kind == 'fsc' else '') + 'progress.json'
                  for role in ('initial', 'selected')])
        # Read counters only. Do not consume scores, predictions or target gold.
        body = 'result={}\nfor name in '+repr(paths)+':\n p=root/name\n if p.is_file():result[name]=json.loads(p.read_text())\nprint(json.dumps(result))\n'
    return 'import json\nfrom pathlib import Path\nroot=Path('+repr(remote)+')\n'+body


def observe(tag, *, base=BASE, remote=ssh, now=None):
    if tag not in TARGETS:
        raise ValueError('not a frozen successor target')
    stamp = time.time() if now is None else now
    folder = base/'successor-observations-v1';folder.mkdir(exist_ok=True)
    with (folder/(tag+'.lock')).open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_path = base/(tag+'-launch.json')
        if not launch_path.exists():
            return dict(target=tag, status='awaiting_original_launch_no_remote_call')
        raw = launch_path.read_bytes();launch = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if (launch['image'] != IMAGE or launch['remote_root'] != REMOTE+tag
                or not re.fullmatch('[0-9a-f]{64}', launch['container_id'])):
            raise ValueError('original launch identity differs')
        state_path = folder/(tag+'-state.json')
        if not state_path.exists():
            state = dict(target=tag, launch_sha256=digest, container_id=launch['container_id'],
                         discovered_unix=stamp, next_due=stamp+INTERVAL, terminal=False)
            write_new(state_path, state)
            return dict(state, status='discovered_original_handle_no_remote_call')
        state = json.loads(state_path.read_text())
        if state['launch_sha256'] != digest or state['container_id'] != launch['container_id']:
            raise ValueError('previously bound launch changed; preserve original handle')
        if state['terminal']:
            return dict(state, status='terminal_already_observed')
        if stamp < state['next_due']:
            return dict(state, status='not_due_no_remote_call')
        record = dict(target=tag, container_id=launch['container_id'], image=IMAGE,
                      launch_sha256=digest, observed_unix=stamp, next_due=stamp+INTERVAL)
        stem = tag+'-'+str(time.time_ns())
        write_new(folder/(stem+'-attempt.json'), record)
        # Persist throttling before network I/O, including failed observations.
        state['next_due'] = stamp+INTERVAL
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n');temporary.replace(state_path)
        try:
            inspected = json.loads(remote(['docker', 'inspect', launch['container_id']]))[0]
            if inspected['Id'] != launch['container_id'] or inspected['Image'] != IMAGE:
                raise ValueError('observed container/image differs')
            record['state'] = inspected['State']
            if inspected['State']['Running']:
                record['progress'] = json.loads(remote(['docker', 'exec', launch['container_id'],
                    'python', '-c', progress_code(TARGETS[tag], launch['remote_root'])]))
            # A persisted launch can precede docker start. Created/paused/
            # restarting are not terminal and never authorize a new launch.
            state['terminal'] = (not inspected['State']['Running'] and
                                 inspected['State']['Status'] in ('exited', 'dead'))
            record['status'] = 'observed_original_handle'
        except Exception as exc:
            record.update(status='observation_failed_preserve_original_handle', error=repr(exc))
        temporary.write_text(json.dumps(state, indent=2)+'\n');temporary.replace(state_path)
        write_new(folder/(stem+'-result.json'), record)
        return record


def main():
    folder=BASE/'successor-observations-v1';folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending=set(TARGETS);errors_until={}
        while pending:
            statuses={}
            for tag in sorted(pending):
                if time.time()<errors_until.get(tag,0):
                    statuses[tag]=dict(status='error_deferred',next_due=errors_until[tag]);continue
                try:
                    result=observe(tag)
                except Exception as exc:
                    errors_until[tag]=time.time()+INTERVAL
                    result=dict(status='local_observer_error',error=repr(exc),next_due=errors_until[tag])
                statuses[tag]=result
                if result.get('terminal') or (result.get('status')=='observed_original_handle' and
                    not result['state']['Running'] and result['state']['Status'] in ('exited','dead')):
                    pending.remove(tag)
                if result['status'] not in ('awaiting_original_launch_no_remote_call','not_due_no_remote_call'):
                    print(json.dumps(dict(target=tag,**result)) if 'target' not in result else json.dumps(result),flush=True)
            path=folder/'watcher-status.json';temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),targets=statuses,
                scope='read_only_original_successors; no dispatch/restart/score_consumption'),indent=2)+'\n')
            temporary.replace(path)
            if pending:
                deadlines=[v['next_due'] for k,v in statuses.items() if k in pending and 'next_due' in v]
                delay=min(60,max(1,min(deadlines)-time.time())) if deadlines else 60
                time.sleep(delay)


if __name__=='__main__':main()
