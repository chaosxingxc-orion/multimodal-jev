"""Independently compare a candidate's recorded replay against pinned HF evidence."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json


def audit(root, manifest_path, refpin, methods, dataset, limit):
    def read(path):
        return json.loads(path.read_bytes())
    if limit <= 0 or not math.isfinite(limit) or not methods or len(set(methods)) != len(methods):
        raise ValueError('invalid performance contract')
    if file_hash(Path(refpin['path'])) != refpin['sha256']:
        raise ValueError('HF reference hash changed')
    reference = read(Path(refpin['path']))
    manifest = read(manifest_path)
    plan, digest = manifest['plan'], manifest['sha256']
    if (content_key(plan) != digest or
        any(plan[k] != reference[k] for k in ('policies', 'inputs'))):
        raise ValueError('replay inputs, plan or weights differ')
    ops = root/'ops'
    if read(ops/'terminal.json') != dict(exit_code=0,status='declared_stages_finished'):
        raise ValueError('workers did not complete')
    expected = {r['block']:r for r in reference['blocks']}
    if set(expected) != {b['key'] for b in plan['blocks']}:
        raise ValueError('incomplete frozen block population')
    sums = defaultdict(lambda:dict(pairs=0, checked_scores=0, batches=0, differences=0,
                                  standard_seconds=0., candidate_seconds=0.))
    receipts = {}
    for f in sorted(ops.glob('block-worker-*-collected.json')):
        d = read(f)
        if d.get('plan_sha256') != digest or d.get('labels_used') is not False or d.get('npu_executed') is not True:
            raise ValueError('invalid execution evidence')
        receipts[f.name] = file_hash(f)
        for rec in d['diagnostic']:
            ref = expected.pop(rec['block'])
            if any(rec[k] != ref[k] for k in ('arm','dataset','policy_id')):
                raise ValueError('policy or dataset changed')
            for b, rb in zip(rec['batches'],ref['batches'],strict=True):
                if b['neighbors'] != rb['neighbors'] or b['pairs'] != len(rb['neighbors']):
                    raise ValueError('batch inputs changed')
                for ex, rx in zip(b['examples'],rb['examples'],strict=True):
                    if any(ex[k] != rx[k] for k in ('id','index','singleton_logit','singleton_probability')):
                        raise ValueError('independent single-pair reference mismatch')
                if rec['dataset'] != dataset:
                    continue
                for mode in methods:
                    timing = b['examples'][0]['variants'][mode]['seconds']
                    if not all(math.isfinite(t) and t > 0 for t in (timing,b['batch_seconds'])):
                        raise ValueError('invalid forward timing')
                    s=sums[rec['arm'],mode]
                    s['batches']+=1
                    s['pairs']+=b['pairs']
                    s['candidate_seconds']+=timing
                    s['standard_seconds']+=b['batch_seconds']
                    for e in b['examples']:
                        v=e['variants'][mode]
                        if v['seconds']!=timing or v['pairs']!=b['pairs']:
                            raise ValueError('candidate batch changed')
                        s['checked_scores']+=1
                        s['differences']+=int((v['logit'],v['probability'])!=(e['singleton_logit'],e['singleton_probability']))
    if expected or {arm for arm, _ in sums} != set(plan['policies']):
        raise ValueError('missing blocks or policies')
    result = {}
    for (arm,mode),s in sums.items():
        s['ratio']=s['candidate_seconds']/s['standard_seconds']
        s['finite_pass']=s['differences']==0 and s['ratio'] <= limit
        result.setdefault(arm,{})[mode]=s
    return dict(scope='finite forward replay; first-decision latency and full job timing not measured',
                dataset=dataset, reference_sha256=refpin['sha256'], plan_sha256=digest,
                receipts=receipts, endpoints=result, max_time_ratio=limit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','reference','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--mode',action='append',required=True)
    p.add_argument('--limit',type=float,required=True)
    a=p.parse_args()
    commit_json(a.output,audit(a.root,a.manifest,dict(path=a.reference,sha256=a.reference_sha256),
                               a.mode,a.dataset,a.limit))
