"""One inventory for Stage1 CPU evidence and packaging (including tail-only files)."""
from pathlib import Path
from hashlib import sha256

from .source_snapshot import regular

FINITE = ('tools/jev/train_context_instruction_npu.py', 'tools/jev/context_process_supervisor.py',
          'tools/reports/context_infonce_actor.py', 'tools/reports/context_infonce_trainer.py')
TAIL = ('tools/reports/context_infonce_tail.py', 'tools/reports/context_replica_formal_candidate.py',
        'tools/reports/context_replica_trainer_candidate.py', 'tools/reports/context_macro_trainer_candidate.py',
        'tools/reports/context_dual_trainer_candidate.py')


def runtime_inventory(files):
    required = (*FINITE, *TAIL, 'src/multimodal_jev/__init__.py')
    missing = set(required) - files.keys()
    if missing:
        raise ValueError('CPU snapshot lacks Stage1 runtime: ' + ', '.join(sorted(missing)))
    return {n: h for n, h in files.items() if n in required or
            (Path(n).parent == Path('src/multimodal_jev') and n.endswith('.py'))}


def copy_checked(project, destination, names, inventory):
    for name in names:
        raw = regular(Path(project), name).read_bytes()
        if sha256(raw).hexdigest() != inventory[name]:
            raise ValueError('Stage1 runtime changed before copy: ' + name)
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
