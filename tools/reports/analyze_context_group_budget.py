"""Offline group-size sensitivity; no rollout, temperature or config change."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from tools.jev.context_reward_signal import singleton_signal


def at_least_one(probability,count):
    if not 0<=probability<=1 or type(count) is not int or count<1:
        raise ValueError('Valid probability and positive integer count required')
    return 1. if probability==1 else -math.expm1(count*math.log1p(-probability))


def run(report_path):
    report=json.loads(Path(report_path).read_text())
    for p,h in report['sources'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Input dependency changed: '+p)
    observation=next(p for p in report['sources'] if p.endswith('live-v13-raw.json'))
    queries=json.loads(Path(observation).read_text())['progress']['latest'][0]['data']['rollout']['queries']
    details=report['queries']
    if len(queries)!=128 or len(details)!=128:raise ValueError('Complete global128 required')
    rows=[]
    for q,d in zip(queries,details):
        if q['request_id'].split('::epoch')[0]!=d['id']:raise ValueError('Request pairing changed')
        for side in ('behavior','reference'):
            p=q[side+'_record']['prob'];g=d['gold_index'];base=singleton_signal(p,g)
            if base!=d[side]['signal']:raise ValueError('Source distribution differs')
            for group in (16,32,64,128,256):
                signal=singleton_signal(p,g,group_size=group)
                rows.append(dict(id=d['id'],side=side,group=group,
                    wrong=not d[side]['decision']['exact_set'],
                    wrong_high_collision=not d[side]['decision']['exact_set'] and base['group_all_equal_reward_probability']>.95,
                    reward_varies=1-signal['group_all_equal_reward_probability'],
                    any_exact_correct=at_least_one(base['exact_correct_set_probability'],group),
                    any_positive_reward=at_least_one(p[g],group)))
    summaries=[]
    for side in ('behavior','reference'):
        for group in (16,32,64,128,256):
            for stratum in ('all','wrong','wrong_high_collision'):
                subset=[r for r in rows if r['side']==side and r['group']==group and (stratum=='all' or r[stratum])]
                summaries.append(dict(side=side,group=group,stratum=stratum,queries=len(subset),
                    **{key:sum(r[key] for r in subset)/len(subset) if subset else None for key in ('reward_varies','any_exact_correct','any_positive_reward')}))
    return dict(schema='context-offline-group-budget-v1',source_report=str(report_path),
        source_sha256=hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),summaries=summaries,
        scope='Exact fixed-probability sampling distribution, same step74 TRAIN batch; no resampling or model update.',
        limits=['Group-size counterfactual changes neither policy nor decisions; not predicted training improvement.',
                'Reference and behavior strata are defined separately; compare all-query rows for matched population.',
                'No audio-forward multiplier inferred from G; no measured NPU cost or gradient variance.',
                'Whole-run and heldout evaluation remain required; formal G16 configuration unchanged.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    result=run(a.report);result['tool_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps([r for r in result['summaries'] if r['side']=='behavior'],indent=2))
