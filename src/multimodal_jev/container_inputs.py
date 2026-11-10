"""Check declared filesystem inputs against explicit same-path Docker mounts."""
import json
from pathlib import Path


def require_visible_inputs(argv, cfg):
    mounts=[]
    for i,arg in enumerate(argv[:-1]):
        if arg=='-v':
            source,target,*_=argv[i+1].split(':')
            mounts.append((Path(source),Path(target)))
    keys = ['requests','instruction','asset_audit','partition','audio_model']
    if cfg.get('evaluation_phase') == 'test':
        keys.remove('partition')
        keys.append('test_protocol')
    if cfg.get('evaluation_phase') == 'slurp-jev-transfer':
        keys.remove('partition')
        keys.append('transfer_protocol')
    if cfg.get('evaluation_phase') == 'slurp-jev-transfer-train':
        keys.remove('partition')
    if cfg.get('task_format') == 'slurp-native':
        keys.remove('partition')
        keys.extend(['taxonomy','population_reference'])
    elif cfg.get('evaluation_phase') == 'slurp-jev-probe':
        keys.remove('partition')
        keys.append('population_reference')
    if 'selection_protocol' in cfg:
        keys.append('selection_protocol')
    inputs = {key:cfg[key] for key in keys}
    if cfg.get('evaluation_phase') == 'test':
        protocol = json.loads(Path(cfg['test_protocol']).read_text())
        inputs.update({key:protocol[key] for key in ('selection_result','selection_protocol')})
    if cfg.get('evaluation_phase') == 'slurp-jev-transfer':
        protocol=json.loads(Path(cfg['transfer_protocol']).read_text())
        inputs.update({key:protocol[key] for key in ('selection_result','selection_protocol','shard_plan')})
        plan=json.loads(Path(protocol['shard_plan']).read_text())
        inputs['jev_release']=plan['release']
    for key, value in inputs.items():
        path=Path(value)
        candidates=[(src,dst) for src,dst in mounts if path==dst or dst in path.parents]
        if not candidates:
            raise ValueError('input outside container mounts: '+key)
        src,dst=max(candidates,key=lambda pair:len(pair[1].parts))
        if not (src/path.relative_to(dst)).exists():
            raise ValueError('mounted input does not exist: '+key)
