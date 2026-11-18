"""Repair shared-reader modes of exact, terminal policy artifacts only.

For legacy frozen runtimes: never changes artifact bytes or training sources.
Uses the fixed CPU image with only this attempt's run directory mounted.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.context_macro_training_operations import validate, command
from tools.reports.context_macro_probe_operations import inspect_runtime, IMAGE


def repair(root, steps):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('regular run root required')
    if (not steps or any(type(s) is not int or not 1 <= s <= 100 for s in steps)
            or len(set(steps)) != len(steps)):
        raise ValueError('explicit unique completed steps required')
    paths = []
    def fail(error):
        raise error
    for step in steps:
        for kind in ('policy', 'adapter'):
            base = root/f'{kind}-{step:04d}'
            if base.is_symlink() or not base.is_dir():
                raise ValueError('complete regular policy/adapter directories required')
            paths.append(base)
            for directory, dirs, files in os.walk(base, onerror=fail, followlinks=False):
                for name in dirs+files:
                    p = Path(directory)/name
                    if p.is_symlink() or not (p.is_file() or p.is_dir()):
                        raise ValueError('regular artifact paths only')
                    paths.append(p)
    # Finish validation and content snapshots before the first chmod.
    before = {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}
    modes = {str(p.relative_to(root)): oct(p.stat().st_mode & 0o7777) for p in paths}
    for p in paths:
        if p.is_file():
            with p.open('rb') as stream:
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
        else:
            p.chmod(0o755)
    after = {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}
    if before != after:
        raise ValueError('artifact bytes changed during permission repair')
    return dict(schema='context-artifact-access-repair-v1', steps=steps,
        original_modes=modes, unchanged_files=after, file_mode='0644', directory_mode='0755')


def run(c):
    validate(c)
    base = Path(c['operations_root'])
    receipt_path = base/'artifact-access-repair.json'
    if receipt_path.exists():
        raise FileExistsError('repair receipt already exists; preserve it')
    terminal = json.loads((base/'terminal.json').read_text())
    if terminal['exit_code'] != 0:
        raise ValueError('successful terminal attempt required')
    def remote(args):
        r = subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
            capture_output=True,text=True,timeout=55)
        if r.returncode:
            raise RuntimeError(r.stderr[-8000:])
        return r.stdout
    containers = {}
    for role in ('preflight','client','service-0','service-1'):
        h = json.loads((base/(role+'-launch.json')).read_text())
        if h['argv'] != command(c,role):
            raise ValueError('launch configuration differs')
        inspect_runtime(json.loads(remote(['docker','inspect',h['container_id']]))[0],h,terminal=True)
        containers[role] = h['container_id']
    t = json.loads((Path(c['prepared'])/'package/training.json').read_text())
    steps = list(range(t['start_update']+1,t['stop_update']+1))
    # Ship this small pure filesystem function, not imports of mutable training code.
    import inspect
    code = ('import os,json,hashlib\nfrom pathlib import Path\n'
        'def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()\n'
        + inspect.getsource(repair)
        + '\nprint(json.dumps(repair(Path("/repair"),'+repr(steps)+')))\n')
    argv = ['docker','run','--rm','--pull=never','--network=none','--user','0',
        '--entrypoint','python','--mount','type=bind,source='+c['remote_root']+'/run,target=/repair',
        IMAGE,'-B','-c',code]
    result = json.loads(remote(argv))
    result.update(containers=containers, image=IMAGE, remote_root=c['remote_root'],
        implementation_sha256=file_hash(__file__))
    write_new(receipt_path,result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local)
    print(json.dumps(run(c),indent=2))
