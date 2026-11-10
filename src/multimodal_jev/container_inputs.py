"""Check declared filesystem inputs against explicit same-path Docker mounts."""
from pathlib import Path


def require_visible_inputs(argv, cfg):
    mounts=[]
    for i,arg in enumerate(argv[:-1]):
        if arg=='-v':
            source,target,*_=argv[i+1].split(':')
            mounts.append((Path(source),Path(target)))
    keys = ['requests','instruction','asset_audit','partition','audio_model']
    if 'selection_protocol' in cfg:
        keys.append('selection_protocol')
    for key in keys:
        path=Path(cfg[key])
        candidates=[(src,dst) for src,dst in mounts if path==dst or dst in path.parents]
        if not candidates:
            raise ValueError('input outside container mounts: '+key)
        src,dst=max(candidates,key=lambda pair:len(pair[1].parts))
        if not (src/path.relative_to(dst)).exists():
            raise ValueError('mounted input does not exist: '+key)
