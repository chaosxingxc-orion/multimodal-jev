#!/usr/bin/env python3
"""Compare matched rule updates only after independent version acceptance."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_rule_comparison import compare_predictions
from verify_span_rule_baseline import verify as verify_hf
from verify_span_rule_runtime import verify as verify_runtime
from verify_span_rule_step import verify as verify_step
from compare_span_rule_policies import compare
from infer_span_rule_baseline import select_rows


def run(training,evaluation,initial):
    read=lambda p:json.loads(p.read_text())
    if read(training/'terminal.json')['exit_code']!=0 or read(evaluation/'terminal.json')['exit_code']!=0:
        raise ValueError('complete training and evaluation required')
    protocol=read(training/'protocol.json');verify_hf(initial)
    config=read(initial/'config.json');rows=select_rows(Path(config['release']))
    by_id={r['id']:r for r in rows};trained=[by_id[k] for k in protocol['request_ids']]
    before=[json.loads(s) for s in (initial/'run/records.jsonl').read_text().splitlines()]
    roots={'initial':initial};reports={};predictions={}
    for name in ('sft-aux0','rl-aux0','sft-aux2','rl-aux2'):
        root=evaluation/name
        step=verify_step(training,name,root);runtime=verify_runtime(root/'runtime')
        if not runtime['passed'] or runtime['adapter_id']!=step['adapter_id']:
            raise ValueError('updated rule policy failed full runtime gate')
        if step['protocol_sha256']!=file_hash(training/'protocol.json'):
            raise ValueError('different matched protocol')
        report=read(training/name/'update.json')
        if report['initial_policy_id']!=config['base_id']+':'+config['adapter_id']:
            raise ValueError('comparison initializer differs')
        roots[name]=root
        after=[json.loads(s) for s in (root/'run/records.jsonl').read_text().splitlines()]
        predictions[name]=after
        reports[name]=dict(step=step,runtime=runtime,paired=compare_predictions(rows,before,after,trained))
    # Existing comparator authenticates identical processor inputs and computes
    # per-arm task metrics. Replace its historical unequal-budget scope here.
    metrics=compare(roots)
    return dict(passed=True,identical_inputs_across_policies=metrics['identical_inputs_across_policies'],
        table=metrics['table'],reports=reports,
        rl_vs_sft={name:compare_predictions(rows,predictions['sft-'+name],predictions['rl-'+name],trained)
                   for name in ('aux0','aux2')},
        auxiliary_vs_final={name:compare_predictions(rows,predictions[name+'-aux0'],predictions[name+'-aux2'],trained)
                            for name in ('sft','rl')},
        protocol_sha256=file_hash(training/'protocol.json'),
        scope='Same SFT initializer, fresh optimizer, eight matched structured requests, one update each; learning-only diagnostics, not compute-matched or unseen-source/task generalization')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('training','evaluation','initial','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();report=run(a.training,a.evaluation,a.initial);atomic_json(a.output,report)
    print(json.dumps(report['table'],indent=2))
