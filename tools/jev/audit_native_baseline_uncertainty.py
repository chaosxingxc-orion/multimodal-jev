#!/usr/bin/env python3
"""Describe uncertainty of frozen native predictions; never optimize on TEST."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np

from consolidate_intent_presence_references import read,rows,sha,require


def cluster_interval(records, group_key, metric, replicates=2000, seed=317):
    require(bool(records) and type(replicates) is int and replicates>0,'nonempty records and positive replicates required')
    groups=defaultdict(lambda:[0,0])
    for r in records:
        require(isinstance(r[group_key],str) and bool(r[group_key]),'explicit cluster identity required')
        require(type(r[metric]) is bool,'boolean outcome required')
        groups[r[group_key]][0]+=r[metric];groups[r[group_key]][1]+=1
    ids=sorted(groups);counts=np.array([groups[k] for k in ids],dtype=np.int64)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        sampled=counts[rng.integers(0,len(ids),size=len(ids))].sum(axis=0)
        values.append(float(sampled[0]/sampled[1]))
    correct,total=counts.sum(axis=0)
    lo,hi=np.quantile(values,[.025,.975],method='linear')
    return dict(estimate=int(correct)/int(total),numerator=int(correct),denominator=int(total),
        clusters=len(ids),cluster_key=group_key,replicates=replicates,seed=seed,lower95=float(lo),upper95=float(hi),
        quantile_method='linear',estimand='recording-weighted fraction; resample whole clusters with replacement',
        degenerate=bool(lo==hi),scope='Post-hoc descriptive interval; not training-seed uncertainty or a significance test.')


def summarize(records):
    require(len({r['id'] for r in records})==len(records),'duplicate recording')
    classes=defaultdict(Counter)
    for r in records:
        classes[r['gold_intent']]['examples']+=1
        classes[r['gold_intent']]['correct']+=r['correct']
        classes[r['gold_intent']]['malformed']+=r['malformed']
    per_class={k:dict(v,recall=v['correct']/v['examples']) for k,v in sorted(classes.items())}
    return dict(examples=len(records),joint_correct=sum(r['correct'] for r in records),
        malformed=sum(r['malformed'] for r in records),observed_gold_classes=len(classes),
        macro_recall_over_observed_classes=sum(v['recall'] for v in per_class.values())/len(classes),
        per_class=per_class,intervals={g:{m:cluster_interval(records,g,m) for m in ('correct','malformed')}
            for g in ('source_group','speaker_id')},optimization_allowed=False,
        scope='Frozen native TEST diagnostic. Source and speaker marginal cluster intervals are separate, not a joint cross-classified interval. No task/template/checkpoint selection.')


def audit(accepted,output):
    require(not output.exists(),'output exists')
    acceptance=read(accepted);result_path=Path(acceptance['result'])
    require(sha(result_path)==acceptance['result_sha256'],'accepted native score changed')
    score=read(result_path);root=result_path.parent
    require(read(root/'terminal.json')['exit_code']==0 and read(root/'terminal.json')['result_sha256']==sha(result_path),'successful accepted result required')
    for p,h in score['pins'].items():require(sha(p)==h,'native dependency changed: '+p)
    require(sha(root/'population.json')==score['population_sha256'],'population changed')
    require(sha(root/'official-predictions.jsonl')==score['normalized_predictions_sha256'],'predictions changed')
    population=read(root/'population.json')['ids'];require(len(population)==len(set(population)),'duplicate population')
    path=lambda suffix:Path(next(p for p in score['pins'] if p.endswith(suffix)))
    requests={r['id']:r for r in rows(path('test.requests.jsonl'))}
    targets={r['id']:r for r in rows(path('test.targets.jsonl'))}
    predictions_list=rows(root/'official-predictions.jsonl')
    predictions={r['file']:r for r in predictions_list}
    require(len(predictions)==len(predictions_list)==len(population),'complete unique predictions required')
    records=[]
    for uid in population:
        target=targets[uid];metadata=target['metadata'];gold=target['target']
        pred=predictions[metadata['file']]
        require(uid=='slurp:'+metadata['file'],'recording identity mismatch')
        sentinel='__invalid_native_output__'
        require((pred['scenario']==sentinel)==(pred['action']==sentinel),'partial invalid sentinel')
        require(isinstance(metadata.get('speaker_id'),str) and bool(metadata['speaker_id']),'speaker unavailable')
        records.append(dict(id=uid,source_group=requests[uid]['source_group'],speaker_id=metadata['speaker_id'],
            gold_intent=gold['scenario']+'_'+gold['action'],predicted_intent=pred['scenario']+'_'+pred['action'],
            correct=(gold['scenario'],gold['action'])==(pred['scenario'],pred['action']),
            malformed=pred['scenario']==sentinel))
    require({r['id'] for r in records}=={'slurp:'+r['file'] for r in predictions_list},'prediction roster mismatch')
    summary=summarize(records)
    require(summary['examples']==score['examples']==acceptance['examples'],'denominator differs')
    require(summary['joint_correct']==acceptance['joint_correct'] and summary['malformed']==score['malformed'],'accepted counts differ')
    summary.update(schema='native-baseline-uncertainty-v1',accepted_result_sha256=sha(result_path),
        acceptance_path=str(accepted),acceptance_sha256=sha(accepted),implementation_sha256=sha(__file__),
        original_result=str(result_path),new_model_inference=False,training_dispatch_allowed=False,
        bootstrap_design='2000 whole-cluster draws, seed317, percentile95 linear quantiles; registered before computation in this tool, after original result existed. Descriptive only.')
    output.mkdir(parents=True)
    (output/'recording-outcomes.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+chr(10) for r in records))
    summary['recording_outcomes_sha256']=sha(output/'recording-outcomes.jsonl')
    (output/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+chr(10))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--accepted',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.accepted,a.output)
    print(json.dumps({k:r[k] for k in ('examples','joint_correct','malformed','observed_gold_classes','macro_recall_over_observed_classes','intervals')},indent=2))
