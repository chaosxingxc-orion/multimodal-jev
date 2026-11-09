"""Paired speaker-cluster bootstrap for completed frozen development evaluations."""
from collections import defaultdict
import json
import random
from pathlib import Path
from .baseline import file_hash
from .evaluate import evaluate,parsed_prediction
from .reward import score_rollout
from .task_schema import JevExample


def load_completed(run,manifest):
    run=Path(run)
    if json.loads((run/'terminal.json').read_text())['exit_code']!=0:
        raise ValueError('successful evaluation terminal required')
    identity=json.loads((run/'identity.json').read_text())
    if identity['manifest_sha256']!=file_hash(manifest):raise ValueError('different evaluation manifest')
    predictions=run/'predictions.jsonl'
    metrics=json.loads((run/'metrics.json').read_text())
    if metrics['predictions_sha256']!=file_hash(predictions) or metrics['manifest_sha256']!=file_hash(manifest):
        raise ValueError('evaluation artifacts changed')
    records={}
    for line in predictions.open():
        row=json.loads(line);key=row['example_id']
        if key in records:raise ValueError('duplicate prediction ID')
        records[key]=row
    return records,identity


def bootstrap(rows,draws,seed):
    """rows=(cluster, base correctness, candidate correctness); resample clusters."""
    groups=defaultdict(lambda:[0,0,0])
    for cluster,left,right in rows:
        cell=groups[cluster];cell[0]+=1;cell[1]+=left;cell[2]+=right
    n=len(rows);keys=sorted(groups);count=len(keys)
    left=sum(r[1] for r in rows)/n;right=sum(r[2] for r in rows)/n
    result={'count':n,'clusters':count,'base_rate':left,'candidate_rate':right,'delta':right-left,
            'delta_ci95':None,'bootstrap_draws':0}
    if count<2:return result
    rng=random.Random(seed);deltas=[]
    for _ in range(draws):
        sample=[groups[keys[rng.randrange(count)]] for _ in range(count)]
        deltas.append(sum(r-l for _,l,r in sample)/sum(size for size,_,_ in sample))
    deltas.sort()
    def quantile(p):
        index=(len(deltas)-1)*p;lo=int(index);hi=min(lo+1,len(deltas)-1)
        return deltas[lo]+(deltas[hi]-deltas[lo])*(index-lo)
    result.update(delta_ci95=[quantile(.025),quantile(.975)],bootstrap_draws=draws)
    return result


def compare(manifest,base,candidate,draws=2000,seed=17,cluster_field='speaker_id'):
    if type(draws) is not int or draws<100:raise ValueError('at least 100 bootstrap draws required')
    left,li=load_completed(base,manifest);right,ri=load_completed(candidate,manifest)
    # Same generation protocol; model/adapter identity is allowed to differ.
    for field in ('max_new_tokens','device','dtype','seed'):
        if li['config'][field]!=ri['config'][field]:raise ValueError('decoding protocol differs: '+field)
    examples=[JevExample.from_record(json.loads(line)) for line in manifest.open()]
    ids={ex.example_id for ex in examples}
    if len(ids)!=len(examples):raise ValueError('duplicate manifest ID')
    if set(left)!=ids or set(right)!=ids:raise ValueError('complete aligned prediction sets required')
    buckets=defaultdict(list);groups=defaultdict(dict)
    for ex in examples:
        cluster=ex.metadata.get(cluster_field)
        if not isinstance(cluster,str) or not cluster:raise ValueError('missing cluster identity')
        scores=[];formats=[]
        for records in (left,right):
            parsed=parsed_prediction(records[ex.example_id]);formats.append(parsed is not None)
            scores.append(parsed is not None and bool(score_rollout(ex,parsed['value'])['semantic']))
        variant=ex.metadata.get('variant',ex.schema.decision_type)
        partition=ex.metadata.get('partition',ex.split)
        presence='null' if ex.target is None else 'present'
        for key in (f'{partition}/{variant}',f'{partition}/{variant}/{presence}'):
            buckets[key].append((cluster,*scores))
        buckets[f'{partition}/format'].append((cluster,*formats))
        group=ex.metadata['source_group']
        if variant in groups[group]:raise ValueError('duplicate group/variant')
        groups[group][variant]=(cluster,partition,*scores)
    for views in groups.values():
        if len({v[:2] for v in views.values()})!=1:raise ValueError('group crosses speaker or partition')
        for a,b,name in [('native','permutation','permutation'),('native','rebind','rebind'),
                         ('native','missing_candidate','missing_candidate'),('noul.True','noul.False','noul')]:
            if a not in views or b not in views:continue
            cluster,partition,l,r=views[a];_,_,ll,rr=views[b]
            buckets[f'{partition}/pair.{name}'].append((cluster,l and ll,r and rr))
    return {'manifest_sha256':file_hash(manifest),'base_run':str(base),'candidate_run':str(candidate),
            'base_prediction_sha256':file_hash(base/'predictions.jsonl'),
            'candidate_prediction_sha256':file_hash(candidate/'predictions.jsonl'),
            'cluster_field':cluster_field,'seed':seed,'draws':draws,
            'comparison':{key:bootstrap(rows,draws,seed) for key,rows in sorted(buckets.items())},
            'rescored_base':evaluate(manifest,base/'predictions.jsonl'),
            'rescored_candidate':evaluate(manifest,candidate/'predictions.jsonl'),
            'scope':'paired percentile bootstrap over speakers; descriptive development comparison, no multiple-testing correction or final efficacy claim'}
