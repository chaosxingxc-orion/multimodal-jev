"""Grant read access to exactly pinned immutable policy assets; verify bytes."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.context_replica_training_operations import IMAGE, path


def repair(root, pins):
    import hashlib
    from pathlib import Path
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or not pins:
        raise ValueError('regular pinned policy required')
    files = []; directories = {root}
    for name, digest in pins.items():
        p = Path(name)
        if not p.is_relative_to(root) or p == root:
            raise ValueError('asset outside exact policy')
        for parent in [p, *p.parents]:
            if parent.is_symlink(): raise ValueError('symlink forbidden')
            if parent == root: break
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('policy bytes differ before repair')
        files.append(p)
        directories.update(parent for parent in p.parents if parent.is_relative_to(root))
    modes = {str(p): oct(p.stat().st_mode & 0o7777) for p in [*directories, *files]}
    for p in directories: p.chmod(0o755)
    for p in files: p.chmod(0o644)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != pins[str(p)] for p in files):
        raise ValueError('policy bytes changed')
    return dict(original_modes=modes, unchanged_files=pins, directories_mode='0755', files_mode='0644')


def run(c):
    root = path(c['policy_root']); output = Path(c['receipt'])
    if output.exists(): raise ValueError('existing repair receipt; do not repeat')
    # Mount only the declared asset root; no model, dataset, or run-state access.
    code = inspect.getsource(repair) + '\nimport json\nprint(json.dumps(repair(' + repr(root) + ',' + repr(c['pins']) + ')))'
    argv = ['docker','run','--rm','--pull=never','--network=none','--user','0',
            '--mount','type=bind,src='+root+',dst='+root,'--entrypoint','python',IMAGE,'-B','-c',code]
    result = subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(argv)],
                            capture_output=True,text=True,timeout=55,check=True)
    report = dict(json.loads(result.stdout), image=IMAGE, root=root, host=c['host'])
    commit_json(output, report)
    return dict(status='exact_policy_access_repaired', files=len(c['pins']), receipt=str(output))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local)
    print(json.dumps(run(c),indent=2))
