"""Bounded recovery admission over the unchanged arithmetic training loop.

Validates the original formal release first, then permits only two recovery
updates. This is not a release of resumed full training or permission to stop
the original job. Device ownership remains a launcher responsibility.
"""
import json
from pathlib import Path

try:
    from context_arithmetic_formal_runtime import validate as validate_original
    from context_formal_runtime import sha
except ModuleNotFoundError:
    from tools.reports.context_arithmetic_formal_runtime import validate as validate_original
    from tools.jev.context_formal_runtime import sha


def expected_config(original, *, origin_config, remote_root, step):
    if (type(step)!=int or not 0<step<633 or step%64 or
            not Path(remote_root).is_absolute() or '..' in Path(remote_root).parts or
            str(Path(remote_root))!=remote_root):
        raise ValueError('explicit committed interval and canonical recovery root required')
    parent=Path(origin_config).parent
    if (Path(remote_root).is_relative_to(parent) or parent.is_relative_to(remote_root)):
        raise ValueError('separate recovery output required')
    result={k:v for k,v in original.items() if k!='pins'}
    result.update(start_update=step,stop_update=step+2,engineering_prefix=True,
        training_dispatch_allowed=False,resume_root=str(parent/'run'),
        initial_adapter=str(parent/'run'/f'adapter-{step:04d}'),
        reference_cache=remote_root+'/reference-cache',resume_origin_config=origin_config,
        resume_runtime_root=remote_root)
    return result


def validate(c):
    origin=Path(c['resume_origin_config'])
    if origin.is_symlink() or c['pins'].get(str(origin))!=sha(origin):
        raise ValueError('pinned original formal config required')
    original=json.loads(origin.read_text());accepted=validate_original(original)
    expected=expected_config(original,origin_config=str(origin),remote_root=c['resume_runtime_root'],step=c['start_update'])
    if {k:v for k,v in c.items() if k!='pins'}!=expected:
        raise ValueError('recovery changed objective, data, reference or execution configuration')
    root=Path(c['resume_runtime_root']);record=root/'RESUME.json'
    if c['pins'].get(str(record))!=sha(record):raise ValueError('pinned recovery manifest required')
    binding=json.loads(record.read_text())
    if (binding['schema']!='context-arithmetic-recovery-prefix-v1' or
            binding['original_config_sha256']!=sha(origin) or binding['step']!=c['start_update'] or
            binding['runtime_root']!=str(root) or binding['formal_release_id']!=accepted['release_id'] or
            binding['training_dispatch_allowed'] is not False):
        raise ValueError('explicit original experiment recovery binding required')
    old=origin.parent
    expected_pins={str(root/name) if name.startswith('source/') else str(old/name):digest
                   for name,digest in binding['original_files'].items()}
    for name,digest in original['pins'].items():
        if not name.startswith(str(old)+'/source/'):
            expected_pins[name]=digest
    expected_pins[str(origin)]=sha(origin)
    for name,digest in binding['changed_files'].items():
        if name not in ('source/tools/jev/train_context_formal.py',
                        'source/tools/jev/context_arithmetic_resume_runtime.py','run.py'):
            raise ValueError('unregistered recovery source change')
        expected_pins[str(root/name)]=digest
    required=binding['checkpoint_files']
    essentials={f'run/checkpoint-{c["start_update"]:04d}-committed.json',
                *[f'run/rank-{rank}/checkpoint-{c["start_update"]:04d}.pt' for rank in range(4)]}
    if not essentials<=required.keys():raise ValueError('all-rank committed recovery files required')
    for name,digest in required.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('run/'):
            raise ValueError('relative recovery artifact required')
        expected_pins[str(old/name)]=digest
    expected_pins[str(record)]=sha(record)
    if c['pins']!=expected_pins:raise ValueError('complete original/recovery source pins required')
    for name,digest in c['pins'].items():
        p=Path(name)
        if p.is_symlink() or sha(p)!=digest:raise ValueError('recovery dependency changed: '+name)
    commit=json.loads((Path(c['resume_root'])/f'checkpoint-{c["start_update"]:04d}-committed.json').read_text())
    if (commit['updates']!=c['start_update'] or
            commit['joint_policy']['policy_id']!=binding['behavior_policy_id']):
        raise ValueError('recovery cursor or behavior differs')
    return dict(**accepted,recovery_step=c['start_update'],finite_prefix_only=True)
