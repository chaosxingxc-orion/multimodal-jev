"""Reduce all-rank microbatch margin numerators without model calls."""
import argparse,json,math
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json


def summarize(records,objective):
    from multimodal_jev.context_infonce import validate_objective
    validate_objective(objective)
    if objective['kind']!='candidate_infonce_probability_margin_v1':raise ValueError('margin objective required')
    if not records or len({r['update'] for r in records})!=1:raise ValueError('one complete update required')
    keys=('infonce_numerator','probability_margin_numerator','threshold_anchor_numerator',
          'probability_gap_sum','margin_satisfied_count','anchored_count','local_real_queries')
    totals={k:0. for k in keys}
    for r in records:
        arrays={k:r['metrics']['jev/'+k] for k in keys}
        if len({len(v) for v in arrays.values()})!=1:raise ValueError('matching microbatch arrays required')
        for k,v in arrays.items():
            if any(type(x) not in (int,float) or not math.isfinite(x) for x in v):raise ValueError('finite metric sums required')
            totals[k]+=sum(v)
    n=totals['local_real_queries']
    if n<=0 or any(r['real_queries']!=n for r in records):raise ValueError('all-rank real denominator differs')
    if not 0<=totals['anchored_count']<=n or not 0<=totals['margin_satisfied_count']<=n:raise ValueError('invalid coverage count')
    m=objective['probability_margin']
    parts={k:totals[k]/n for k in keys[:3]}
    total=parts['infonce_numerator']+m['gap_weight']*parts['probability_margin_numerator']+m['anchor_weight']*parts['threshold_anchor_numerator']
    native=sum(r['metrics']['actor/pg_loss'] for r in records)/len(records)
    if not math.isclose(total,native,rel_tol=2e-5,abs_tol=2e-6):raise ValueError('native actor loss differs from objective reduction')
    return dict(update=records[0]['update'],real_queries=int(n),loss=total,
        infonce=parts['infonce_numerator'],probability_margin=parts['probability_margin_numerator'],
        threshold_anchor=parts['threshold_anchor_numerator'],mean_gap=totals['probability_gap_sum']/n,
        margin_coverage=totals['margin_satisfied_count']/n,anchor_coverage=totals['anchored_count']/n,
        slowest_rank_seconds=max(r['total_seconds'] for r in records),scope='training_batches_not_full_population_efficacy')


def run(c):
    config=json.loads(Path(c['training_config']).read_text());world=config['batch_config']['training_devices']
    root=Path(c['run']);result=[]
    for step in c['steps']:
        records=[json.loads((root/f'rank-{rank}/update-{step:04d}.json').read_text()) for rank in range(world)]
        result.append(summarize(records,config['instruction_objective']))
    receipt=dict(objective=config['instruction_objective'],updates=result)
    commit_json(c['output'],receipt);return receipt

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);x=a.parse_args();c,_=load_config(x.config,x.local);print(json.dumps(run(c),indent=2))
