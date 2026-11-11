#!/usr/bin/env python3
"""Full three-method comparison after authenticated fixed-final evaluations."""
import argparse,json
from copy import deepcopy
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_matched_budget import compare_training_exposure
from multimodal_jev.span_transfer_evaluation import cell_metrics
from multimodal_jev.span_rule_population import configured_rule_rows
from compare_span_transfer import compare as verify_evaluation


def compare(roots):
    if set(roots)!={'sft','rl','conditional'}:raise ValueError('exactly three methods required')
    read=lambda p:json.loads(p.read_text());reports={};branches={};finals={};rows=None;reference_initial=None;study=None
    for method,root in roots.items():
        report=verify_evaluation(root);reports[method]=report
        current_study=read(root/'study.json')
        if study is None:study=current_study
        elif current_study!=study:raise ValueError('different full study')
        training=Path(read(root/'evaluation-protocol.json')['training_root']);c=read(training/'config.json')
        if c['method']!=method:raise ValueError('method label differs from actual training')
        steps=[]
        for index in range(1,report['training']['completed_updates']+1):
            update=training/'run'/f'update-{index:04d}'
            steps.append(dict(config=read(update/'config.json'),report=read(update/'run/update.json')))
        # Whole controller duration from its first initial-stage timestamp to terminal.
        # Stage metadata carries no launch timestamp; use source-stable run filesystem
        # creation time only as an explicitly approximate start, never pure throughput.
        first=training/'run/version-0000/config.json'
        end=read(training/'run/terminal.json')['finished_unix']
        branches[method]=dict(study_sha256=report['study_sha256'],
            initial_policy_id=steps[0]['report']['initial_policy_id'],steps=steps,
            wall_seconds_including_gates=end-first.stat().st_mtime)
        initial=[];final=[];current_rows=[]
        for role in ('learning','selection'):
            for policy,records in [('initial',initial),('final',final)]:
                stage=root/(policy+'-'+role)
                records.extend(map(json.loads,(stage/'run/records.jsonl').read_text().splitlines()))
            current_rows.extend(configured_rule_rows(read(root/('initial-'+role)/'config.json')))
        if rows is None:rows=current_rows;reference_initial=initial
        elif rows!=current_rows or initial!=reference_initial:
            raise ValueError('full population or initial predictions differ across methods')
        finals[method]=final
    budget=compare_training_exposure(branches)
    for method in ('rl','conditional'):
        for a,b in zip(finals['sft'],finals[method],strict=True):
            x,y=deepcopy(a['request_identity']),deepcopy(b['request_identity']);x.pop('policy_id');y.pop('policy_id')
            if a['id']!=b['id'] or x!=y:raise ValueError('cross-method final input identity differs')
    paired={}
    for before,after in [('sft','rl'),('sft','conditional'),('rl','conditional')]:
        paired[after+'_vs_'+before]=cell_metrics(study,rows,finals[before],finals[after])
    table=[]
    for method,report in reports.items():
        for cell,data in report['cells'].items():
            for arm in ('final_only','structured','reordered'):
                rules=[r for r in data['after']['per_rule'] if r['arm']==arm];n=sum(r['n'] for r in rules)
                table.append(dict(method=method,cell=cell,arm=arm,requests=n,
                    final_accuracy=sum(r['final_accuracy']*r['n'] for r in rules)/n,
                    macro_rule_ba=sum(r['balanced_accuracy'] for r in rules)/len(rules),
                    all_span_accuracy=sum(r['all_span_accuracy']*r['n'] for r in rules)/n,
                    complement=data['after']['complement_response'][arm]))
    return dict(passed=True,table=table,actual_budget=budget,paired=paired,
        per_method_initial_comparison={k:v['cells'] for k,v in reports.items()},
        evidence={k:dict(root=str(roots[k]),study_sha256=v['study_sha256'],evaluation_protocol_sha256=v['evaluation_protocol_sha256'],training=v['training']) for k,v in reports.items()},
        elapsed_time_note='Approximate observed controller time from initial gate config mtime to terminal; includes gates; not an authenticated monotonic compute measurement',
        scope='One-seed matched-exposure method comparison in prespecified SLURP development cells. Descriptive unadjusted source intervals, no multi-seed or cross-dataset claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('sft','rl','conditional','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=compare({name:getattr(a,name) for name in ('sft','rl','conditional')});atomic_json(a.output,r)
    print(json.dumps(r['table'],indent=2))
