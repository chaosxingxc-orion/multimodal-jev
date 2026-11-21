"""Read the exact replica-training handle at most once per 900 seconds.

Read-only: no dispatch, restart, collection, cleanup or policy change. A failed
remote observation also consumes the ordinary interval. Container exit is not
training success; the independent collector and audit still own that verdict.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.local_config import load_config
from tools.reports.launch_context_batch_probe_candidate import write_new

INTERVAL = 900

REMOTE = """import json
from pathlib import Path
run=Path(ROOT)/'run';out={}
for p in sorted(run.glob('rank-*/progress.json')):
 out.setdefault('progress',{})[p.parent.name]=json.loads(p.read_text())
for p in sorted(run.glob('rank-*/update-*.json')):
 name=p.parent.name
 out.setdefault('updates',{})[name]=out.get('updates',{}).get(name,0)+1
commits=sorted(p.name for p in run.glob('policy-*-committed.json'))
out['policy_commits']=len(commits)
out['last_policy_commit']=commits[-1] if commits else None
out['checkpoint_commits']=sorted(p.name for p in run.glob('checkpoint-*-committed.json'))
for name in ('result.json',):
 path=run/name
 if path.is_file():out[name]=json.loads(path.read_text())
print(json.dumps(out))
"""


def handle(c):
    base = Path(c['operations_root'])
    path = base / (c['tag'] + '-launch.json')
    if not path.is_file():
        raise ValueError('existing launched handle required')
    launch = json.loads(path.read_text())
    if (launch.get('manifest_sha256') != c['manifest_sha256']
            or launch.get('image') is None
            or not isinstance(launch.get('container_id'), str)):
        raise ValueError('launch handle belongs to another declared package')
    return base, launch


def observe(c, *, now=None, remote=None, read_progress=None):
    base, launch = handle(c)
    root = Path(c.get('observations_root') or (base / (c['tag'] + '-observations')))
    root.mkdir(exist_ok=True)

    def ssh(args):
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', c['host'], shlex.join(args)],
            capture_output=True, text=True, timeout=55)
        if result.returncode:
            raise RuntimeError(result.stderr[-8000:])
        return result.stdout

    remote = remote or ssh
    with (root / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        history = [json.loads(p.read_text()) for p in root.glob('*-attempt.json')]
        if any(r['container_id'] != launch['container_id'] for r in history):
            raise ValueError('observation belongs to another container')
        last = max([(base / (c['tag'] + '-launch.json')).stat().st_mtime]
                   + [r['observed_unix'] for r in history])
        stamp = time.time() if now is None else now
        if stamp < last + INTERVAL:
            return dict(status='not_due_no_remote_call',
                        next_ordinary_observation_unix=last + INTERVAL)
        record = dict(container_id=launch['container_id'], image=launch['image'],
                      observed_unix=stamp,
                      next_ordinary_observation_unix=stamp + INTERVAL,
                      scope='read-only exact handle; not a training audit')
        stem = str(time.time_ns())
        # Persist before remote I/O; a lost SSH reply is not a terminal state.
        write_new(root / (stem + '-attempt.json'), record)
        inspected = json.loads(remote(['docker', 'inspect', launch['container_id']]))[0]
        if inspected['Id'] != launch['container_id'] or inspected['Image'] != launch['image']:
            raise ValueError('observed container/image identity differs')
        record['state'] = inspected['State']
        if inspected['State']['Running']:
            if read_progress is None:
                record['progress'] = json.loads(remote(
                    ['docker', 'exec', launch['container_id'], 'python', '-c',
                     REMOTE.replace('ROOT', repr(launch['remote_root']))]))
            else:
                record['progress'] = read_progress(launch, remote)
        record['status'] = 'observed_exact_handle'
        write_new(root / (stem + '-result.json'), record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.local)
    print(json.dumps(observe(cfg), indent=2))
