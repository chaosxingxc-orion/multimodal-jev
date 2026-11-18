#!/usr/bin/env python3
"""Verify immutable baseline bytes; this does not grant runtime admission."""
import argparse,hashlib,json
from pathlib import Path


def verify(path):
    lock=json.loads(Path(path).read_text())
    payload={k:v for k,v in lock.items() if k!='lock_id'}
    digest=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=lock['lock_id']:raise ValueError('baseline lock identity changed')
    bound=dict(lock['files'])
    bound.update(lock['evidence'])
    archive=lock['existing_execution_archive'];bound[archive['path']]=archive['sha256']
    errors=[]
    for name,expected in bound.items():
        p=Path(name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append(name)
    if errors:raise ValueError('baseline file missing or changed: '+str(errors))
    registry=json.loads((Path(path).parent/'algorithms.json').read_text())
    if [x['id'] for x in registry['algorithms']]!=['grpo_v1','rloo_v1','ppo_critic_design_v1']:
        raise ValueError('algorithm registry differs')
    if registry['global_training_dispatch_allowed'] or lock['stage2_launcher'] is not None:
        raise ValueError('this baseline has not admitted continuous Stage2')
    return dict(lock_id=digest,verified_files=len(bound),stage2_released=False,
                scope='File and contract identity only; no added runtime or efficacy evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',default='configs/releases/context-prob-v1/LOCK.json')
    print(json.dumps(verify(p.parse_args().lock),indent=2))
