#!/usr/bin/env python3
"""Exact design audit of fixed-capacity semantic rewards; no training runtime."""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path

from consolidate_intent_presence_references import read, sha, require


def coefficients(panels, capacity, caps):
    require(set(capacity)==set(caps), 'family support differs')
    require(all(type(n) is int and n>0 for n in capacity.values()), 'positive fixed capacity required')
    require(all(isinstance(c,F) and c>=0 for c in caps.values()), 'nonnegative exact caps required')
    seen=set();result=[]
    for panel in panels:
        counts={f:0 for f in capacity};row={}
        for q in panel['process_questions']:
            f,k=q['operator'],q['definition_id']
            require(f in capacity,'unknown family')
            require((f,k) not in seen,'duplicate predicate across complete sweep')
            seen.add((f,k));counts[f]+=1;row[f,k]=caps[f]/capacity[f]
        require(all(counts[f]<=capacity[f] for f in capacity),'panel exceeds capacity')
        require(counts==panel['configured_counts'],'configured count mismatch')
        result.append(row)
    return result


def process_reward(weights, correct, masks):
    require(set(weights)==set(correct)==set(masks),'coordinate support mismatch')
    require(all(type(x) is bool for x in [*correct.values(),*masks.values()]),'boolean correctness and masks required')
    return sum((w*correct[k]*masks[k] for k,w in weights.items()),F(0))


def audit(menu, output):
    require(not output.exists(),'output exists')
    report=read(menu/'report.json')
    for n,h in report['files'].items():require(sha(menu/n)==h,'menu changed')
    panels=read(menu/'panels.json')
    capacity={'I_present':4,'S_present':4};caps={f:F(1,10) for f in capacity}
    rows=coefficients(panels,capacity,caps);sweep={};detail=[]
    for panel,weights in zip(panels,rows,strict=True):
        alltrue={k:True for k in weights}
        maxreward=process_reward(weights,alltrue,alltrue)
        require(maxreward<=F(1,5),'raw cap exceeded')
        for k,w in weights.items():sweep[k]=w/len(panels)
        detail.append(dict(panel_id=panel['panel_id'],counts=panel['configured_counts'],
                           max_process_reward=str(maxreward)))
    require(set(sweep.values())=={F(1,680)},'unequal per-predicate raw exposure')
    family={f:sum((w for (ff,k),w in sweep.items() if ff==f),F(0)) for f in capacity}
    require(family=={'I_present':F(13,136),'S_present':F(11,136)},'family exposure mismatch')
    # The same one-question action with a constant Final demonstrates why raw
    # coefficient equalization does not itself enforce group-gradient budgets.
    advantages={}
    for w in [F(1,10),F(1,40)]:
        rewards=[0.]*8+[float(w)]*8;mean=sum(rewards)/16
        std=math.sqrt(sum((r-mean)**2 for r in rewards)/15)
        advantages[str(w)]={'grpo_correct':(float(w)-mean)/(std+1e-6),
                             'rloo_correct':float(w)*8/15}
    result=dict(schema='jev-semantic-fixed-capacity-budget-audit-v1',
        menu_report_sha256=sha(menu/'report.json'),implementation_sha256=sha(__file__),
        per_question_raw_coefficient='1/40',per_predicate_equal_panel_mean='1/680',
        full_sweep_family_coefficients={f:str(v) for f,v in family.items()},
        full_sweep_total_process_coefficient=str(sum(family.values(),F(0))),
        same_input_final_correct_minimum_reward_margin='4/5',panels=detail,
        constant_final_G16_eight_process_correct=advantages,
        scope='Exact reward-coefficient and fixed-example advantage arithmetic; not shared-parameter gradient balance, semantics, algorithm integration or model improvement.',
        training_dispatch_allowed=False,semantic_admission=0)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--menu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.menu,a.output);print(json.dumps({k:v for k,v in r.items() if k!='panels'},indent=2))
