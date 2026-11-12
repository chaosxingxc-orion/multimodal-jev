"""Verify Docker bind mappings expose every pinned host file at its exact path."""
from pathlib import Path,PurePosixPath
from .baseline import file_hash


def validate_bind_pins(argv,pins):
    mounts=[]
    for index,arg in enumerate(argv):
        if arg not in ('-v','--volume'):continue
        fields=argv[index+1].split(':')
        if len(fields)!=3 or fields[-1] not in ('ro','rw'):
            raise ValueError('explicit absolute bind with ro/rw required')
        source,target,mode=fields
        if not source.startswith('/') or not target.startswith('/'):
            raise ValueError('absolute bind required')
        mounts.append((PurePosixPath(target),Path(source),mode))
    if len({str(t) for t,_,_ in mounts})!=len(mounts):raise ValueError('duplicate bind target')
    verified=[]
    for name,digest in pins.items():
        path=PurePosixPath(name)
        if not path.is_absolute() or '..' in path.parts:raise ValueError('canonical absolute pin required')
        matches=[(t,s,m) for t,s,m in mounts if path==t or t in path.parents]
        if not matches:raise ValueError('pinned path not mounted: '+name)
        target,source,mode=max(matches,key=lambda x:len(x[0].parts))
        actual=source/str(path.relative_to(target))
        if not actual.is_file() or file_hash(actual)!=digest:raise ValueError('mounted pin missing/changed: '+name)
        verified.append(dict(path=name,host=str(actual),mode=mode))
    return dict(passed=True,pinned_files=len(verified),bindings=verified)
