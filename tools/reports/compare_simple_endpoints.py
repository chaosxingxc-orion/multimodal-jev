"""Replay audited full-population endpoints; compare fixed terminal policies after collection."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from multimodal_jev.context_execution import map_cpu
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.paired_event_decisions import compare
from multimodal_jev.stage2_view_runtime import pinned


def bound(pin):
    return pinned(pin['path'],pin['sha256'])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',',':'),
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_cell(name, spec, ref, policy):
    audit = bound(ref['audit'])
    # The native audit can contain one dense confusion matrix per source group.
    audit.pop('by_group',None)
    plan = bound(ref['plan'])
    heldout = spec['split'] != 'source_selection'
    expected = 'jev-stage2-view-heldout-endpoint-v1' if heldout else 'jev-stage2-view-selection-source-v1'
    point = plan['checkpoints'][-1]
    manifest = ref['seal']
    root = Path(manifest['root'])
    seal = pinned(root/'MANIFEST.json',manifest['sha256'])
    if (seal['schema'] != 'jev-native-evaluation-seal-v1' or seal['target_tuning_allowed'] is not False
            or plan['schema'] != expected or plan['evaluation_dataset'] != name
            or plan['target_tuning_allowed'] is not False or plan['model_selection_performed'] is not False
            or (heldout and (len(plan['checkpoints']) != 1 or plan['fixed_terminal_step'] != point['step']
                             or plan['seal_sha256'] != manifest['sha256']))
            or (not heldout and (seal['files']['source-plan.json'] != ref['plan']['sha256']
                                 or [p['step'] for p in plan['checkpoints']] != sorted(set(p['step'] for p in plan['checkpoints']))))
            or point['step'] != audit['step'] or point['policy_id'] != policy
            or audit['dataset'] != name or audit['seal_sha256'] != manifest['sha256']
            or audit['target_tuning_allowed'] is not False):
        raise ValueError('wrong policy, endpoint roster, population or evaluation seal')
    meta = seal['datasets'][name]
    if (meta['split'],meta['recordings'],meta['candidates']) != (spec['split'],spec['recordings'],spec['classes']):
        raise ValueError('full frozen evaluation split required')
    def sealed(rel):
        return pinned(root/rel,seal['files'][rel])
    panel = sealed(f'inference/{name}/panel.json')
    assets = sealed(f'inference/{name}/rows.json')
    targets = sealed(f'scoring/{name}/targets.json')
    ids = panel['candidate_ids']
    metrics = audit['metrics']
    if (ids != metrics['candidate_ids'] or len(ids) != spec['classes'] or len(set(ids)) != len(ids)
            or metrics['queries'] != spec['recordings'] or len(assets) != spec['recordings']):
        raise ValueError('wrong candidate order or population')
    pred = pinned(ref['predictions'],audit['predictions_sha256'])
    if audit['status'] == 'complete_hf_shards_audited':
        if audit['runtime'] != 'HF_standalone' or pred['policy_id'] != policy:
            raise ValueError('audited standalone HF policy required')
        runtime = 'HF_standalone'
    elif audit['status'] == 'complete_macro_evaluation_outputs_audited':
        if (audit['policy_id'],audit['recordings'],audit['decisions']) != (policy,len(assets),len(ids)):
            raise ValueError('audited macro service policy required')
        runtime = 'macro_service'
    else:
        raise ValueError('completed independent full audit required')
    probs = {}
    for a,t,r in zip(assets,targets,pred['records'],strict=True):
        if (t['id'] in probs or r['id'] != a['id'] or a['id'] != t['id'] or a['audio_sha256'] != t['audio_sha256']
                or r['request_identity']['policy_id'] != policy or r.get('policy_id',policy) != policy
                or not isinstance(t['group'],str) or not t['group'] or t['gold'] not in ids):
            raise ValueError('recording, scoring group or request policy differs')
        probs[t['id']] = dict(zip(ids,r['prob'],strict=True))
    values = np.array([list(v.values()) for v in probs.values()])
    if values.dtype.kind not in 'fi' or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('finite raw candidate probabilities required')
    gold = np.array([ids.index(t['gold']) for t in targets])
    guessed = values.argmax(axis=1)
    mask = np.arange(len(ids)) == gold[:,None]
    total = np.bincount(gold,minlength=len(ids))+np.bincount(guessed,minlength=len(ids))
    tp = np.bincount(gold[guessed==gold],minlength=len(ids))
    with np.errstate(divide='ignore'):
        losses = -np.log(np.where(mask,values,1-values))
    observed = dict(top1_macro_f1=float(np.divide(2*tp,total,out=np.zeros(len(ids)),where=total>0).mean()),
        top1_event_accuracy=float((guessed==gold).mean()), mean_top1_probability=float(values.max(axis=1).mean()),
        candidate_binary_nll=float(losses.mean()) if np.isfinite(losses).all() else None,
        candidate_brier=float(np.square(values-mask).mean()))
    if any((metrics[k] is not None if v is None else
            metrics[k] is None or not math.isclose(v, metrics[k], rel_tol=1e-10, abs_tol=1e-12)) for k,v in observed.items()):
        raise ValueError('full raw predictions do not reproduce independent scores')
    identity = dict(panel=fingerprint(panel), roster=fingerprint(targets), split=meta['split'],
                    recordings=len(targets), candidates=len(ids), cluster_unit=spec['cluster'])
    cell = dict(identity=identity, policy_id=policy, base_id=plan['base_id'], source_release=plan['release']['sha256'],
        training_audit_sha256=plan['training_audit']['sha256'], step=point['step'], audit_sha256=ref['audit']['sha256'],
        source_selection_sha256=plan['source_selection_sha256'] if heldout else manifest['sha256'],
        predictions_sha256=audit['predictions_sha256'], runtime=runtime, metrics=observed,
        predicted_classes=len(set(guessed.tolist())), clusters=len(set(t['group'] for t in targets)))
    return cell, ids, targets, probs


def evaluate(item):
    name,spec,left,right,interval = item
    a,ids,targets,p = load_cell(name,spec,*left)
    if right is None:
        return name,dict(baseline=a)
    b,_,_,q = load_cell(name,spec,*right)
    if a['identity'] != b['identity'] or a['base_id'] != b['base_id'] or a['source_release'] != b['source_release']:
        raise ValueError('paired predictions require identical native panel, recordings, groups, gold and base')
    delta = compare(ids,targets,p,q,interval=interval,cluster='group')
    quality = all(delta['left'][m] is not None and delta['right'][m] is not None
                  and delta['right'][m]<=delta['left'][m] for m in ('brier','nll'))
    return name,dict(baseline=a,candidate=b,paired=delta,
                     improves_with_probability_quality=delta['macro_f1_delta_ci95'][0]>0 and quality)


def run(c):
    if c['schema']!='jev-simple-endpoint-comparison-v1' or c['mode'] not in ('bind_baseline','compare'):
        raise ValueError('explicit baseline or paired comparison required')
    ds = c['datasets']
    if not ds or len([v for v in ds.values() if v['split']=='source_selection'])!=1:
        raise ValueError('one complete source selection required')
    sources = [c['baseline']]
    locked = None
    if c['mode']=='compare':
        locked = bound(c['baseline_receipt'])
        if (locked['mode'] != 'bind_baseline' or locked['design_sha256'] != c['design']['sha256']
                or set(locked['datasets']) != set(ds)):
            raise ValueError('frozen full baseline receipt required')
        sources.append(c['candidate'])
        if sources[0]['policy_id']==sources[1]['policy_id']:raise ValueError('distinct policies required')
    elif c['candidate'] is not None or c.get('baseline_receipt') is not None:
        raise ValueError('candidate not expected during baseline binding')
    if any(set(s['cells'])!=set(ds) for s in sources):
        raise ValueError('all declared datasets required')
    interval = bound(c['design'])['interval']
    ref = lambda s,k:(s['cells'][k],s['policy_id'])
    jobs = [(k,v,ref(sources[0],k),ref(sources[1],k) if len(sources)>1 else None,interval) for k,v in ds.items()]
    execution = dict(c['execution'])
    rows,stats = map_cpu(evaluate,jobs,workers=execution.pop('cpu_workers'),**execution)
    cells = dict(rows)
    if locked and any(cells[k]['baseline'] != locked['datasets'][k]['baseline'] for k in ds):
        raise ValueError('baseline endpoint differs from preregistered scores')
    for key in ('baseline','candidate')[:len(sources)]:
        if any(len({cell[key][field] for cell in cells.values()})!=1 for field in (
                'base_id','step','training_audit_sha256','source_release','source_selection_sha256')):
            raise ValueError('mixed source checkpoint lineage across datasets')
    return dict(mode=c['mode'],design_sha256=c['design']['sha256'],datasets=cells,
                wall_seconds=stats['wall_seconds'],target_tuning_allowed=False,npu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--local',required=True)
    args=p.parse_args(); cfg,provenance=load_config(args.config,args.local)
    result=run(cfg);commit_json(cfg['output'],dict(result,provenance=provenance))
