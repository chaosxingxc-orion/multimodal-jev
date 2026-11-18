"""Retention for explicitly inventoried, closed experiments; no directory scans.

The producer declares closure and dependencies. This tool authenticates those
declarations and archived conclusions; it cannot discover unregistered jobs.
"""
import argparse
import fcntl
import os
from pathlib import Path
import re
import stat
import json

from multimodal_jev.local_config import load_config
from tools.cache_cleanup import delete_one, identity, protected
from tools.collect_training_artifacts import sha, verify_files
from tools.reports.committed_signal_report import check_paths, read, write


def prepare(config):
    required = {'schema', 'root', 'inventory', 'inventory_sha256', 'pin_manifests',
                'protected_paths', 'output_directory', 'lock_path', 'max_files', 'max_bytes'}
    if (set(config) != required or config['schema'] != 'closed-experiment-retention-v1'
            or any(type(config[k]) is not int or config[k] <= 0 for k in ('max_files', 'max_bytes'))):
        raise ValueError('Explicit bounded retention configuration required')
    root = Path(config['root']).absolute()
    if (root.resolve() != root or not root.is_dir() or len(root.parts) < 4
            or any(p in {'models', 'datasets', '.git'} for p in root.parts)):
        raise ValueError('Narrow nonsymlink experiment root required')
    inventory = Path(config['inventory']).resolve()
    sources = {str(inventory): config['inventory_sha256']}
    verify_files(sources)
    data = read(inventory)
    if (set(data) != {'schema', 'closed', 'evaluation_complete', 'terminal', 'terminal_sha256',
                     'files', 'conclusion_archives', 'selected_checkpoint_steps'}
            or data['schema'] != 'closed-experiment-inventory-v1'
            or data['closed'] is not True or data['evaluation_complete'] is not True):
        raise ValueError('Closed experiment and completed evaluation declarations required')
    def member(name):
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.resolve() != path
                or not path.is_file()):
            raise ValueError('Regular nonsymlink inventoried file required')
        return path
    terminal = member(data['terminal'])
    sources[str(terminal)] = data['terminal_sha256']; verify_files(sources)
    if read(terminal).get('exit_code') != 0:
        raise ValueError('Retain failed run evidence; successful terminal receipt required')
    protection = [Path(p).resolve() for p in config['protected_paths']]
    protection += [inventory, terminal, Path(config['output_directory']).resolve(), Path(config['lock_path']).resolve()]
    for name in config['pin_manifests']:
        path = Path(name).resolve(); pins = read(path); pins = pins.get('pins', pins)
        if not isinstance(pins, dict) or not pins or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in pins.values()):
            raise ValueError('Nonempty dependency pin maps required')
        sources[str(path)] = sha(path)
        protection += [path] + [Path(p).resolve() for p in pins]
    rows, steps, roles = {}, set(), set()
    if not isinstance(data['files'], list) or not data['files']:
        raise ValueError('Explicit nonempty file inventory required')
    for row in data['files']:
        if set(row) != {'path', 'sha256', 'role', 'step'} or row['path'] in rows:
            raise ValueError('Distinct fully specified inventory entries required')
        role = row['role']; path = member(row['path'])
        if role not in {'checkpoint', 'final', 'config', 'conclusion', 'intermediate'}:
            raise ValueError('Unknown artifact role')
        if any(part in {'models', 'datasets', '.git'} for part in Path(row['path']).parts):
            raise ValueError('Shared asset categories cannot be retention candidates')
        if role == 'checkpoint':
            if type(row['step']) is not int or row['step'] < 0:
                raise ValueError('Checkpoint step required')
            steps.add(row['step'])
        elif row['step'] is not None:
            raise ValueError('Noncheckpoint step must be null')
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid artifact digest')
        rows[row['path']] = row; roles.add(role)
        if role in {'final', 'config', 'conclusion'}:
            protection.append(path); sources[str(path)] = row['sha256']
    if not {'final', 'config', 'conclusion'} <= roles:
        raise ValueError('Final results, configurations and conclusions must be inventoried')
    selected = data['selected_checkpoint_steps']
    if (not isinstance(selected, list) or any(type(s) is not int or s not in steps for s in selected)):
        raise ValueError('Selected checkpoints must exist in inventory')
    keep_steps = set(selected) | ({max(steps)} if steps else set())
    keep_steps.update(row['step'] for name, row in rows.items()
                      if row['role'] == 'checkpoint' and protected(member(name), protection))
    conclusions = {n for n, row in rows.items() if row['role'] == 'conclusion'}
    if set(data['conclusion_archives']) != conclusions:
        raise ValueError('Every conclusion requires an archive')
    for name, archived in data['conclusion_archives'].items():
        p = Path(archived).absolute()
        if p.resolve() != p or p.is_relative_to(root) or not p.is_file():
            raise ValueError('Conclusion archive must be outside experiment root')
        sources[str(p)] = rows[name]['sha256']; protection.append(p)
    verify_files(sources)
    candidates, retained = [], []
    for name, row in rows.items():
        path = member(name)
        if row['role'] == 'checkpoint' and row['step'] in keep_steps:
            protection.append(path); sources[str(path)] = row['sha256']
    verify_files(sources)
    for name, row in rows.items():
        path = member(name)
        if protected(path, protection):
            # Active dependency files are authenticated even when not deletable.
            sources[str(path)] = row['sha256']; retained.append(name); continue
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or sha(path) != row['sha256']:
            raise ValueError('Candidate identity or hardlink changed')
        candidates.append(dict(path=str(path), root=str(root), relative=name,
            sha256=row['sha256'], identity=identity(st), reason=row['role']))
    verify_files(sources)
    total = sum(row['identity']['size'] for row in candidates)
    if len(candidates) > config['max_files'] or total > config['max_bytes']:
        raise ValueError('Retention deletion budget exceeded')
    return dict(files=candidates, retained=retained, logical_bytes=total,
                kept_checkpoint_steps=sorted(keep_steps), sources=sources), protection


def run(template, local, apply=False):
    config, provenance = load_config(template, local)
    root, output, lock = [Path(config[k]).resolve() for k in ('root', 'output_directory', 'lock_path')]
    if output == root or output.is_relative_to(root) or root.is_relative_to(output) or lock.is_relative_to(root):
        raise ValueError('Retention publication and lock must be outside experiment root')
    check_paths({'root':root, 'publication':output, 'lock':lock.parent})
    lock.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as owner, (output/'owner.lock').open('a') as pub_owner:
        for stream in (owner, pub_owner): fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = output/'COMPLETE.json'
        if complete.exists():
            previous = read(complete)
            if previous['provenance'] != provenance or previous['apply'] != apply:
                raise ValueError('Different retention run; use a new output version')
            verify_files(previous['sources']); verify_files(previous['outputs'])
            if apply and any(Path(p).exists() or Path(p).is_symlink() for p in previous['deleted_paths']):
                raise ValueError('Deleted artifact path reappeared; do not delete again')
            return dict(status='already_complete', **previous['summary'])
        if (output/'status.json').exists():
            raise ValueError('Partial retention preserved; automatic retry disabled')
        write(output/'status.json',dict(phase='planning', apply=apply))
        deleted=[]
        try:
            plan, protection = prepare(config)
            sources = plan['sources']
            sources.update({str(Path(template).resolve()):provenance['template_sha256'],
                            str(Path(local).resolve()):provenance['local_bindings_sha256']})
            for p in (__file__, 'tools/cache_cleanup.py', 'tools/collect_training_artifacts.py',
                      'tools/jev/collect_context_rl_training.py', 'src/multimodal_jev/local_config.py',
                      'src/multimodal_jev/local_runtime_storage.py', 'tools/reports/committed_signal_report.py'):
                sources[str(Path(p).resolve())]=sha(p)
            if any(Path(p)==Path(row['path']) for p in sources for row in plan['files']):
                raise ValueError('Cleanup input cannot be deleted')
            verify_files(sources); write(output/'plan.json',plan)
            events=output/'events';events.mkdir()
            outputs={str(output/'plan.json'):sha(output/'plan.json')}
            for i,row in enumerate(plan['files'] if apply else []):
                verify_files(sources)
                check_paths({'candidate':Path(row['path']).parent})
                intent=events/f'{i:06d}-intent.json';write(intent,row)
                outputs[str(intent)]=sha(intent)
                delete_one(row,protection);deleted.append(row['path'])
                event=events/f'{i:06d}-deleted.json';write(event,dict(path=row['path'],deleted=True))
                outputs[str(event)]=sha(event)
            verify_files(sources)
            summary=dict(candidates=len(plan['files']),deleted_files=len(deleted),
                         candidate_logical_bytes=plan['logical_bytes'],kept_checkpoint_steps=plan['kept_checkpoint_steps'])
            write(complete,dict(schema='closed-experiment-retention-publication-v1',provenance=provenance,
                apply=apply,sources=sources,outputs=outputs,deleted_paths=deleted,summary=summary))
            write(output/'status.json',dict(phase='complete'));return dict(status='complete',**summary)
        except Exception as exc:
            write(output/'status.json',dict(phase='failed_preserve_evidence',error=repr(exc),deleted_paths=deleted))
            raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--local',default=os.environ.get('JEV_LOCAL_CONFIG','.local/config/macros.json'))
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.config,a.local,a.apply)))
