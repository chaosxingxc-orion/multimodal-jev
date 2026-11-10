#!/usr/bin/env python3
"""Audit full learning-label composition and fixed synthetic reward controls."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shards import verify_plan,population
from multimodal_jev.sfc_jev_reward import bundle_rewards


def audit(plan_path,plan_hash):
    if file_hash(plan_path)!=plan_hash:raise ValueError('registered learning plan changed')
    plan=json.loads(Path(plan_path).read_text())
    if plan['split']!='train_id' or plan['partition'] is None or plan['partition']['part']!='learning':
        raise ValueError('complete TRAIN learning population required')
    verify_plan(plan)
    release=Path(plan['release']);all_rows=population(release,'train_id')
    wanted={uid for shard in plan['shards'] for uid in shard['ordered_bundle_ids']}
    learning={g for shard in plan['shards'] for g in shard['source_groups']}
    target=release/'train_id.targets.jsonl';manifest=json.loads((release/'MANIFEST.json').read_text())
    if file_hash(target)!=manifest['splits']['train_id']['targets_sha256']:raise ValueError('TRAIN target file changed')
    counts=Counter();by_level=defaultdict(Counter);histogram=Counter();group_counts=defaultdict(Counter)
    modes=('oracle','consistent_empty','constant_false','malformed')
    totals={m:Counter() for m in modes};rewards={m:[] for m in modes};bundle_means={m:[] for m in modes}
    activity={m:{k:Counter() for k in ('active','inactive')} for m in modes}
    input_digest=hashlib.sha256()
    with target.open() as gold_file:
        for row,line in zip(all_rows,gold_file,strict=True):
            # Export preserves request/target order. Selection target bytes are
            # skipped without JSON parsing or scoring; whole-file hash above
            # still binds the authoritative release.
            if row['id'] not in wanted:continue
            gold=json.loads(line)
            if gold['id']!=row['id']:raise ValueError('learning target order differs')
            input_digest.update(canonical(row));flags={}
            for d in row['input']['decisions']:
                if d['id'].endswith('.calls'):flags[d['id'].rsplit('.',1)[0]]=bool(gold['answers'][d['id']])
            level=row['source_group'].split('/',1)[0]
            c=Counter(bundles=1,decisions=len(row['input']['decisions']),functions=len(flags),
                active_functions=sum(flags.values()),inactive_functions=sum(not v for v in flags.values()),
                no_call_bundles=int(not any(flags.values())))
            counts.update(c);by_level[level].update(c);group_counts[row['source_group']].update(c)
            histogram[str(len(flags))]+=1
            for mode in modes:
                ps=[]
                for d in row['input']['decisions']:
                    text=json.dumps(gold['answers'][d['id']],ensure_ascii=False)
                    if mode=='consistent_empty':text='[]' if d['id'].endswith('.calls') else ('true' if d['id'].endswith('.absent') else 'false')
                    if mode=='constant_false':text='false'
                    if mode=='malformed':text='```json\nfalse\n```'
                    ps.append(dict(example_id=row['id'],decision_id=d['id'],
                        request_sha256=hashlib.sha256(canonical(row)).hexdigest(),raw_text=text))
                scored=bundle_rewards(row,gold,ps,learning_groups=learning,mode='function-conjunction')
                for key in ('correct_function_triples','consistent_function_triples','native_calls_exact','all_decisions_correct'):
                    totals[mode][key]+=scored[key]
                rewards[mode].extend(p['total'] for p in scored['decisions']);bundle_means[mode].append(scored['bundle_mean_reward'])
                for p in scored['decisions']:
                    active=flags[p['decision_id'].rsplit('.',1)[0]];bucket=activity[mode]['active' if active else 'inactive']
                    bucket.update(decisions=1,correct=int(p['semantic']),format_valid=int(p['format_valid']),
                        schema_valid=int(p['schema_valid']))
    if counts['bundles']!=plan['bundles'] or counts['decisions']!=plan['decisions'] or set(group_counts)!=learning:
        raise ValueError('complete learning coverage required')
    reports={}
    oracle=math.fsum(rewards['oracle'])
    for mode in modes:
        reports[mode]=dict(totals[mode],total_reward=math.fsum(rewards[mode]),
            reward_fraction_of_oracle=math.fsum(rewards[mode])/oracle,
            decision_mean_reward=math.fsum(rewards[mode])/counts['decisions'],
            bundle_macro_mean_reward=math.fsum(bundle_means[mode])/counts['bundles'],
            activity=dict(activity[mode]))
    if (reports['oracle']['all_decisions_correct']!=counts['bundles']
            or reports['consistent_empty']['correct_function_triples']!=counts['inactive_functions']
            or reports['consistent_empty']['native_calls_exact']!=counts['no_call_bundles']):
        raise ValueError('synthetic reward control does not match label composition')
    return dict(schema='sfc-jev-complete-learning-reward-audit-v1',plan=str(Path(plan_path).resolve()),
        plan_sha256=plan_hash,partition=plan['partition'],release=plan['release'],manifest_sha256=plan['manifest_sha256'],
        train_targets_sha256=file_hash(target),learning_inputs_sha256=input_digest.hexdigest(),
        counts=dict(counts),source_groups=len(group_counts),by_level={k:dict(v) for k,v in sorted(by_level.items())},
        functions_per_bundle_histogram=dict(sorted(histogram.items(),key=lambda p:int(p[0]))),
        synthetic_controls=reports,source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/audit_sfc_learning_rewards.py','src/multimodal_jev/sfc_jev_reward.py','src/multimodal_jev/sfc_jev_score.py')},
        model_called=False,weight_updates=0,verified_unix=time.time(),
        scope='Complete registered learning population only. Whole TRAIN target bytes hashed; selection lines '
        'skipped without parsing/scoring. No TEST files read. Fixed synthetic outputs, not model performance '
        'or RL benefit. No reward changes or weighting chosen from selection/TEST.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-hash',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior reward audit')
    result=audit(a.plan,a.plan_hash)
    with a.output.open('xb') as handle:handle.write(canonical(result))
    print(json.dumps(dict(counts=result['counts'],source_groups=result['source_groups'],
        synthetic_controls=result['synthetic_controls']),indent=2))
