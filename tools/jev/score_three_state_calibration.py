#!/usr/bin/env python3
"""Score all terminal paired calibration requests; never admit supervision."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from build_blind_semantic_annotation import validate_response
from consolidate_intent_presence_references import read, rows, require, sha
from parse_semantic_annotation_response import parse_response_content


def summarize(cases, answers, arms):
    require(len(arms)==len(set(arms))==2, 'two distinct arms required')
    keys=[(c['request_id'],c['semantic_id']) for c in cases]
    require(len(keys)==len(set(keys)) and bool(keys),'unique nonempty cases required')
    arm_reports={};pairs=[]
    for arm in arms:
        counts=defaultdict(Counter)
        for c in cases:
            predicted=answers[arm].get(c['request_id'],{}).get(c['semantic_id'])
            expected=c['expected_status']
            for bucket in ('all',c['family']+'/'+expected):
                counts[bucket]['targets']+=1
                counts[bucket]['returned']+=predicted is not None
                counts[bucket]['agreement']+=predicted==expected
                counts[bucket][expected+'__'+(predicted or 'missing')]+=1
        arm_reports[arm]={k:dict(v) for k,v in sorted(counts.items())}
    for c in cases:
        values={a:answers[a].get(c['request_id'],{}).get(c['semantic_id']) for a in arms}
        pairs.append(dict(**c,predictions=values,agreements={a:v==c['expected_status'] for a,v in values.items()}))
    return dict(arms=arm_reports,paired_targets=pairs,semantic_admission=0,bulk_dispatch_allowed=False,
                scope='Agreement with hash-selected author references, not independent or population accuracy. All scheduled failures remain in denominators.')


def score(root,output):
    require(not output.exists(),'output exists')
    plan=read(root/'plan.json');launch=read(root/'launch.json');terminal=read(root/'terminal.json')
    require(launch['plan_sha256']==sha(root/'plan.json'),'launch plan changed')
    data=Path(plan['data_root']);report=read(data/'report.json')
    require(sha(data/'report.json')==plan['data_report_sha256'],'calibration changed')
    for n,h in report['files'].items():require(sha(data/n)==h,'calibration file changed')
    cases=rows(data/'cases.offline.jsonl')
    requests={r['request_id']:r for r in rows(data/'requests.jsonl')}
    jobs={(j['arm'],j['request_id']):j for j in plan['jobs']}
    endings={(r['arm'],r['request_id']):r for r in terminal['results']}
    require(len(jobs)==len(plan['jobs'])==plan['total_requests'],'duplicate jobs')
    require(len(endings)==len(terminal['results']) and set(endings)==set(jobs),'all terminal requests required')
    require(set(jobs)=={(a,r) for a in plan['arms'] for r in requests},'complete paired schedule required')
    receipts=[];answers={a:{} for a in plan['arms']};usage={a:Counter() for a in plan['arms']}
    systems={a:set() for a in plan['arms']};transport_configs=set()
    for key,job in sorted(jobs.items()):
        arm,rid=key;p=Path(job['path']);t=endings[key]
        require(read(p/'terminal.json')==t,'per-request terminal differs')
        require(sha(p/'messages.private.json')==job['messages_sha256'] and
                sha(p/'config.private.json')==job['config_sha256'],'frozen request changed')
        messages=read(p/'messages.private.json');cfg=read(p/'config.private.json')
        require(cfg['model']==plan['model'] and cfg['max_tokens']==plan['max_tokens_per_request'],
                'configured model or budget differs from plan')
        transport_configs.add(json.dumps({k:v for k,v in cfg.items() if k not in ('messages','output_root')},sort_keys=True))
        require(len(messages)==2 and messages[0]['role']=='system' and messages[1]['role']=='user','two-message input required')
        systems[arm].add(messages[0]['content'])
        user,_=parse_response_content(messages[1]['content'])
        require(user==requests[rid]['user'],'full panel user changed')
        sent=read(p/'response/request.json')
        expected=dict(model=cfg['model'],messages=messages,max_tokens=cfg['max_tokens'],stream=cfg.get('stream',False))
        if expected['stream']:expected['stream_options']={'include_usage':True}
        if 'thinking' in cfg:expected['thinking']=cfg['thinking']
        require(sent==expected,'sent body changed')
        result=read(p/'response/result.json')
        require(all(t[k]==result.get(k) for k in ('state','http_status','finish_reasons','usage')),'terminal result differs')
        receipt=dict(arm=arm,request_id=rid,panel_id=job['panel_id'],valid=False,
            exit_code=t['exit_code'],http_status=result.get('http_status'),finish_reasons=result.get('finish_reasons'),
            usage=result.get('usage'),request_sha256=sha(p/'response/request.json'),
            result_sha256=sha(p/'response/result.json'),failure='transport_or_process_failure')
        for k in ('prompt_tokens','completion_tokens','total_tokens'):usage[arm][k]+=(result.get('usage') or {}).get(k,0)
        if t['exit_code']==0 and result.get('http_status')==200 and result.get('state')=='response_received':
            raw=read(p/'response/response.json');receipt['response_sha256']=sha(p/'response/response.json')
            require(raw.get('usage')==result.get('usage'),'raw usage differs')
            choices=raw['choices'];require([c.get('finish_reason') for c in choices]==result['finish_reasons'],'raw finish differs')
            receipt['failure']='incomplete_or_multiple_choices'
            if len(choices)==1 and choices[0]['finish_reason']=='stop':
                try:
                    parsed,representation=parse_response_content(choices[0]['message'].get('content',''))
                    validate_response(parsed,user)
                except (ValueError,TypeError,KeyError) as e:receipt['failure']='invalid_output: '+str(e)
                else:
                    receipt.update(valid=True,decisions=parsed['decisions'],representation=representation)
                    receipt.pop('failure')
                    answers[arm][rid]={d['id']:d['status'] for d in parsed['decisions']}
        receipts.append(receipt)
    require(all(len(v)==1 for v in systems.values()),'system prompt varies within arm')
    require(len(transport_configs)==1,'transport configuration differs between jobs')
    require(next(iter(systems['envelope_three_state']))==next(iter(systems['envelope']))+plan['directive'],
            'paired prompt intervention differs')
    result=summarize(cases,answers,plan['arms'])
    result.update(schema='three-state-calibration-observed-v1',plan_sha256=sha(root/'plan.json'),
        terminal_sha256=sha(root/'terminal.json'),data_report_sha256=sha(data/'report.json'),
        implementation_sha256=sha(__file__),usage={a:dict(v) for a,v in usage.items()},receipts=receipts,
        training_dispatch_allowed=False,model_identity=plan['model_identity'])
    for arm in plan['arms']:
        valid=sum(r['valid'] for r in receipts if r['arm']==arm)
        result['arms'][arm]['transport']=dict(requests=len(requests),valid=valid)
        result['arms'][arm]['narrow_calibration_gate_passed']=valid==len(requests) and result['arms'][arm]['all']['agreement']==len(cases)
    result['comparable_full_panel_decisions']=[]
    for rid in sorted(requests):
        if all(rid in answers[a] for a in plan['arms']):
            for q in requests[rid]['user']['questions']:
                result['comparable_full_panel_decisions'].append(dict(request_id=rid,semantic_id=q['id'],
                    predictions={a:answers[a][rid][q['id']] for a in plan['arms']}))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=score(a.root,a.output)
    print(json.dumps({k:r[k] for k in ('arms','usage','semantic_admission','bulk_dispatch_allowed')},indent=2))
