#!/usr/bin/env python3
"""Authenticate complete D/Q/P updates before comparing learning effects."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_matched_effects import compare
from multimodal_jev.intent_optimizer import optimizer_kwargs
from multimodal_jev.intent_resume import validate_lineage
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed
from compare_intent_learning_update import run as score_update


# These are artifact locations, arm choice, and cursor, not optimization factors.
LINEAGE = {'pins','arm','step','initial_adapter','acceptance_root','previous_root','previous_suite'}


def run(config):
    check_pins(config['pins'])
    protocol=read(config['protocol'])
    if config['pins'].get(config['protocol'])!=file_hash(config['protocol']):raise ValueError('pinned protocol required')
    count=protocol['total_updates']
    if set(config['arms'])!={'D','Q','P'} or any(len(v)!=count for v in config['arms'].values()):
        raise ValueError('complete three-arm budget required')
    initial=Path(config['initial_gate']);launch=read(initial/'launch.json')
    if (config['pins'].get(str(initial/'launch.json'))!=file_hash(initial/'launch.json')
            or launch['image']!=config['image'] or not completed(launch['container_id'],config['image'])):
        raise ValueError('actual common initial gate required')
    arms={};reference=None;seen=set();evidence={}
    for arm,steps in config['arms'].items():
        points=[];prior=None
        for index,step in enumerate(steps):
            train,suite=Path(step['training_root']),Path(step['suite_root'])
            for root in (train,suite):
                if str(root) in seen:raise ValueError('distinct training and suite roots required')
                seen.add(str(root))
                if read(root/'launch.json')['image']!=config['image']:raise ValueError('same fixed image required')
            cfg=read(train/'config.json')
            if (cfg['arm']!=arm or cfg['step']!=index or cfg['seed']!=protocol['seed']
                    or cfg['lr']!=protocol['lr'] or cfg['global_requests']!=protocol['global_requests_per_update']):
                raise ValueError('actual training arm or matched budget differs')
            factors={k:v for k,v in cfg.items() if k not in LINEAGE}
            if reference is None:reference=factors
            if factors!=reference or optimizer_kwargs(cfg)!=optimizer_kwargs(reference):
                raise ValueError('non-arm training factors differ')
            if prior is None:
                if cfg['acceptance_root']!=str(initial) or cfg.get('previous_root') or cfg.get('previous_suite'):
                    raise ValueError('common fresh initialization required')
            else:
                validate_lineage(prior,cfg)
                previous=steps[index-1]
                if cfg['previous_root']!=previous['training_root'] or cfg['previous_suite']!=previous['suite_root']:
                    raise ValueError('contiguous arm lineage required')
            checked=score_update(train,suite)
            selection=read(train/'run/selection.json')
            if selection['recording_ids']!=protocol['schedule'][index]:raise ValueError('source budget differs')
            if not points:points.append(dict(update=0,score=checked['full_before']))
            elif checked['full_before']!=points[-1]['score']:raise ValueError('predecessor scoring differs')
            points.append(dict(update=index+1,score=checked['full_after']))
            evidence.update(checked['pins']);prior=cfg
        arms[arm]=points
    result=compare(arms,protocol['schedule'])
    evidence.update(config['pins'])
    for path in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_matched_effects.py'):
        evidence[str(path)]=file_hash(path)
    return dict(result,evidence=evidence,protocol_sha256=file_hash(config['protocol']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable matched comparison already exists')
    atomic_json(a.output,run(read(a.config)))
