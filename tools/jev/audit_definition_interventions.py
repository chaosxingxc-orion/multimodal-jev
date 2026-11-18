#!/usr/bin/env python3
"""Task-design audit of polarity transforms; never authorize model scoring/training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transform(status, config):
    if status not in config['truth_transform']:
        raise ValueError('explicit registered supervision state required')
    return dict(config['truth_transform'][status])


def pair_scores(gold, affirmative, complement):
    if type(gold) is not bool:raise ValueError('known boolean truth required')
    valid=type(affirmative) is bool and type(complement) is bool
    correct_a=type(affirmative) is bool and affirmative==gold
    correct_c=type(complement) is bool and complement!=gold
    opposite=valid and affirmative!=complement
    return dict(both_correct=correct_a and correct_c,valid_opposite=opposite,
                opposite_but_both_wrong=opposite and not correct_a and not correct_c,
                invalid_pair=not valid)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--review-release',type=Path,required=True)
    p.add_argument('--review-report-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report_path=a.review_release/'report.json'
    if sha(report_path)!=a.review_report_sha256:raise ValueError('review identity mismatch')
    report=json.loads(report_path.read_text());cfg=json.loads(a.config.read_text())
    for path,h in report['pins'].items():
        if sha(path)!=h:raise ValueError('changed dependency '+path)
    for name,h in report['files'].items():
        if sha(a.review_release/name)!=h:raise ValueError('changed review file')
    if cfg['training_dispatch_allowed'] is not False or cfg['evaluation_dispatch_allowed'] is not False:
        raise ValueError('design-only configuration required')
    decisions=[json.loads(x) for x in (a.review_release/'decisions.jsonl').read_text().splitlines()]
    bindings=[json.loads(x) for x in (a.review_release/'recording-bindings.jsonl').read_text().splitlines()]
    fixtures=[]
    for r in decisions:
        targets=transform(r['status'],cfg)
        if r['status'] in ('positive','negative') and targets['affirmative'] is not r['proposed_semantic_value']:
            raise ValueError('reference polarity mismatch')
        fixtures.append(dict(source_id=r['source_id'],definition_id=r['entity_type'],
            definition_sha256=r['definition_sha256'],policy_sha256=r['policy_sha256'],
            source_status=r['status'],reference_only_targets=targets,
            optimization_allowed=False,model_scoring_allowed=False,
            limitation='Reference-only transform fixture, not an admitted audio evaluation label.'))
    controls={}
    for label,predict in [('always_false',lambda y:(False,False)),
                          ('template_only',lambda y:(True,False)),
                          ('perfect_but_wrong_complement',lambda y:(not y,y)),
                          ('oracle',lambda y:(y,not y))]:
        scores=[pair_scores(y,*predict(y)) for y in (False,True)]
        controls[label]={k:sum(x[k] for x in scores)/2 for k in scores[0]}
    out=dict(schema='jev-definition-intervention-audit-v1',config=str(a.config),config_sha256=sha(a.config),
        source_report=str(report_path),source_report_sha256=a.review_report_sha256,
        implementation_sha256=sha(__file__),reference_decisions=len(decisions),
        reference_status_counts=dict(Counter(r['status'] for r in decisions)),
        known_reference_pairs=sum(r['status'] in ('positive','negative') for r in decisions),
        masked_reference_pairs=sum(r['status'] in ('unavailable','conflict','unassessed') for r in decisions),
        known_recording_pair_bindings=sum(r['status'] in ('positive','negative') for r in bindings),
        masked_recording_pair_bindings=sum(r['status'] in ('unavailable','conflict','unassessed') for r in bindings),
        synthetic_balanced_controls=controls,fixtures=fixtures,
        scope='Transforms and synthetic control calculations only. No model result, no new audio labels, no training or evaluation dispatch.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('fixtures',)},indent=2))


if __name__=='__main__':main()
