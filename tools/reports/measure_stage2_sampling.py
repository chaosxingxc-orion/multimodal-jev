"""Configured full-size synthetic CPU sampler comparison; no efficacy claim."""
import argparse,json,time
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.stage2_training_view import sample,make_schedule
from multimodal_jev.stage2_dataset_release import validate_execution


def run(c):
    validate_execution(c['execution'])
    n=c['population'];k=c['candidates']
    if type(n)is not int or n<1 or type(k)is not int or k<2:raise ValueError('positive population and candidate count required')
    candidates=[str(i) for i in range(k)]
    rows=[dict(id=str(i),intent=str(i%k),role='learning',official_split='train',known_pending_source=False) for i in range(n)]
    probs={}
    for i,r in enumerate(rows):
        p=[.1]*k;p[i%k]=.9
        if i%5==0:p[i%k]=.1;p[(i+1)%k]=.9
        elif i%5==1:p[i%k]=.52;p[(i+1)%k]=.48
        probs[r['id']]=p
    results=[];baseline=None
    for workers in c['cpu_workers_grid']:
        execution=dict(c['execution'],cpu_workers=workers);validate_execution(execution);timings={}
        started=time.monotonic();value=sample(rows,candidates,probs,c['sampling'],execution=execution,timings=timings)
        elapsed=time.monotonic()-started;signature=digest(value)
        if baseline is None:baseline=signature
        elif signature!=baseline:raise ValueError('serial/parallel outputs differ')
        results.append(dict(cpu_workers=workers,wall_seconds=elapsed,sha256=signature,execution=timings))
    batches=make_schedule(rows,value[0],dict(c['batch_config'],source_queries=len(value[0])),c['sampling']['seed'])
    # Independent per-epoch real-ID coverage, not merely aggregate counts.
    expected=set(value[0])
    for epoch in range(c['batch_config']['epochs']):
        ids=[s['id'] for b in batches if b['epoch']==epoch for s in b['slots'] if not s['padding']]
        if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('epoch coverage differs')
    result=dict(scope='synthetic_full_size_cpu_not_real_Stage1_predictions',model_calls=0,npu_executed=False,
        population=n,candidates=k,summary=value[2],updates=len(batches),measurements=results,
        identical_outputs=True,schedule_sha256=digest(batches),config=c)
    commit_json(c['output'],result)
    return {k:v for k,v in result.items() if k not in ('summary','config')}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);x=a.parse_args();c,_=load_config(x.config,x.local);print(json.dumps(run(c),indent=2))
