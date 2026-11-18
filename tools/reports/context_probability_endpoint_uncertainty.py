"""Selection-only probability-quality intervals at the fixed RL endpoint.

Uses whole source groups, retains recording weights, never clips probabilities
or selects a model. Existing full-run audit remains the admission boundary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tools.reports.context_simple_endpoint_uncertainty import score_pair, run as audit_endpoint
from tools.jev.context_rl_evaluation_plan import sha


def per_query(probabilities, gold):
    width=len(probabilities)
    infinite=any(p==0 if i==gold else p==1 for i,p in enumerate(probabilities))
    nll=None if infinite else -math.fsum(math.log(p) if i==gold else math.log1p(-p)
                                        for i,p in enumerate(probabilities))/width
    brier=math.fsum((p-int(i==gold))**2 for i,p in enumerate(probabilities))/width
    return nll,brier


def compare(initial, final, labels, candidates, policies):
    # Reuse complete pairing, probability validity, source-truth and input checks.
    decision,metrics=score_pair(initial,final,labels,candidates,policies)
    values={};infinite=[0,0]
    for a,b,label in sorted(zip(initial,final,labels,strict=True),key=lambda rows:rows[2]['id']):
        gold=candidates.index(label['intent'])
        x,y=per_query(a['prob'],gold),per_query(b['prob'],gold)
        for index,row in enumerate((x,y)):infinite[index]+=int(row[0] is None)
        values.setdefault(label['source_group'],[]).append((None if x[0] is None or y[0] is None else y[0]-x[0],y[1]-x[1]))
    names=sorted(values);counts=np.asarray([len(values[g]) for g in names],dtype=np.int64)
    # All bootstrap draws match the existing decision endpoint analysis.
    draws=np.random.Generator(np.random.PCG64(317)).integers(0,len(names),size=(2000,len(names)),dtype=np.int64)
    draws_hash=hashlib.sha256(draws.astype('<i8',copy=False).tobytes()).hexdigest()
    if draws_hash!=decision['bootstrap']['draws_sha256']:raise ValueError('bootstrap draws differ')
    results={}
    for column,name in enumerate(('candidate_binary_nll','candidate_brier')):
        if column==0 and any(infinite):
            results[name]=dict(initial=metrics[0][name],final=metrics[1][name],delta=None,
                percentile_95=None,status='undefined_paired_difference_due_to_infinite_nll',
                infinite_queries=dict(initial=infinite[0],final=infinite[1]))
            continue
        sums=np.asarray([math.fsum(row[column] for row in values[g]) for g in names])
        bootstrap=np.concatenate([sums[d].sum(axis=1)/counts[d].sum(axis=1) for d in np.array_split(draws,20)])
        delta=float(sums.sum()/counts.sum())
        if abs(delta-(metrics[1][name]-metrics[0][name]))>1e-12:raise ValueError('point scoring differs')
        results[name]=dict(initial=metrics[0][name],final=metrics[1][name],delta=delta,
            percentile_95=np.quantile(bootstrap,[.025,.975],method='linear').tolist(),status='finite')
    return dict(endpoint='fixed_step634',recordings=len(labels),source_groups=len(names),
        metrics=results,bootstrap=decision['bootstrap'],pairing_sha256=decision['pairing_sha256'],
        direction='final_minus_initial; negative is better for both proper scores',
        scope='Pointwise paired source-group sampling uncertainty conditional on fixed outputs. '
              'No seed variance, simultaneous coverage, selected-checkpoint guarantee, clipping, '
              'TEST tuning or automatic complex-training admission.')


def run(directory,algorithm,release):
    endpoint=audit_endpoint(directory,algorithm,release)
    root=Path(directory);curve_path=root/'curve.json'
    if sha(curve_path)!=endpoint['provenance']['curve_sha256']:raise ValueError('curve changed after audit')
    curve=json.loads(curve_path.read_text());population=Path(release)
    for name in ('selection.jsonl','tasks.json'):
        if sha(population/name)!=endpoint['sources'][str(population/name)]:raise ValueError('release changed after audit')
    labels=[json.loads(line) for line in (population/'selection.jsonl').read_text().splitlines()]
    candidates=[r['id'] for r in json.loads((population/'tasks.json').read_text())['intent']['schema']['candidates']]
    records=[];policies=[]
    for step in (0,634):
        point=next(p for p in curve['independently_audited_points'] if p['step']==step)
        p=Path(point['collection'])/'evaluation/selection/predictions.json'
        if sha(p)!=endpoint['sources'][str(p)]:raise ValueError('prediction changed after audit')
        records.append(json.loads(p.read_text())['records']);policies.append(point['inference_policy_id'])
    return dict(schema='context-probability-fixed-endpoint-uncertainty-v1',algorithm=algorithm,
        decision_endpoint=endpoint,probability=compare(*records,labels,candidates,policies),
        implementation_sha256=sha(__file__),complex_training_admitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('directory','algorithm','release','output'):parser.add_argument('--'+key,required=True)
    args=vars(parser.parse_args());output=Path(args.pop('output'));result=run(**args)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
