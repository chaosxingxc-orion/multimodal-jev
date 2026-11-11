#!/usr/bin/env python3
"""Compare exact-credit candidates to authenticated SFT and joint GRPO."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_rule_comparison import compare_predictions
from multimodal_jev.span_source_statistics import paired_source_interval
from compare_span_rule_matched import run as compare_matched
from compare_span_rule_policies import compare
from verify_span_rule_step import verify as verify_step
from verify_span_rule_runtime import verify as verify_runtime
from infer_span_rule_baseline import select_rows


def run(training,evaluation,matched_training,matched_evaluation,initial):
    read=lambda p:json.loads(p.read_text())
    original=compare_matched(matched_training,matched_evaluation,initial)
    if read(training/'terminal.json')['exit_code']!=0 or read(evaluation/'terminal.json')['exit_code']!=0:
        raise ValueError('conditional branches must complete training and evaluation')
    p=read(training/'protocol.json');q=read(matched_training/'protocol.json')
    for key in ('release_manifest_sha256','initial_adapter_id','runtime_admission','arm','request_ids',
                'num_updates','optimizer_start','seed','lr'):
        if p[key]!=q[key]:raise ValueError('conditional and matched controls differ: '+key)
    if p['objectives']!=[['conditional',0.],['conditional',.2]]:
        raise ValueError('both prespecified conditional branches required')
    roots={'initial':initial,**{name:matched_evaluation/name for name in ('sft-aux0','rl-aux0','sft-aux2','rl-aux2')}}
    reports={}
    config=read(initial/'config.json');rows=select_rows(Path(config['release']))
    by_id={r['id']:r for r in rows};trained=[by_id[k] for k in p['request_ids']]
    for name in ('conditional-aux0','conditional-aux2'):
        root=evaluation/name;step=verify_step(training,name,root);runtime=verify_runtime(root/'runtime')
        update=read(training/name/'update.json')
        if (not runtime['passed'] or runtime['adapter_id']!=step['adapter_id']
                or update['initial_policy_id']!=config['base_id']+':'+config['adapter_id']
                or step['protocol_sha256']!=file_hash(training/'protocol.json')):
            raise ValueError('conditional policy failed initializer/runtime identity')
        roots[name]=root;reports[name]=dict(step=step,runtime=runtime)
    metrics=compare(roots)
    predictions={name:[json.loads(s) for s in (root/'run/records.jsonl').read_text().splitlines()]
                 for name,root in roots.items()}
    paired={};intervals={}
    for weight in ('aux0','aux2'):
        name='conditional-'+weight
        for baseline in ('initial','sft-'+weight,'rl-'+weight):
            paired[name+'_vs_'+baseline]=compare_predictions(rows,predictions[baseline],predictions[name],trained)
            intervals[name+'_vs_'+baseline]=paired_source_interval(rows,predictions[baseline],predictions[name])
    return dict(passed=True,table=metrics['table'],identical_inputs_across_policies=True,
        conditional_reports=reports,paired=paired,source_intervals=intervals,runs=metrics['runs'],
        matched_protocol_sha256=original['protocol_sha256'],conditional_protocol_sha256=file_hash(training/'protocol.json'),
        scope='One update, seed17, matched initializer/requests/lr; exact-support candidate differs in credit, standardization and marginal clipping jointly. Learning-only, not compute matched, causal component attribution or generalization evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('training','evaluation','matched-training','matched-evaluation','initial','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=run(a.training,a.evaluation,a.matched_training,a.matched_evaluation,a.initial)
    atomic_json(a.output,r);print(json.dumps(r['table'],indent=2))
