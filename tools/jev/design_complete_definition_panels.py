#!/usr/bin/env python3
"""Design complete definition coverage and fixed per-audio weights; never train."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_panels(orders, width=4):
    if type(width) is not int or width < 1 or set(orders) != {'intent', 'slot'}:
        raise ValueError('positive width and intent/slot families required')
    for values in orders.values():
        if not values or len(set(values)) != len(values):
            raise ValueError('nonempty distinct definition IDs required')
    count = max((len(v)+width-1)//width for v in orders.values())
    return [dict(panel_id=i, final_weight=str(Fraction(1,count)),
        families={f:dict(definitions=v[i*width:(i+1)*width],
                        per_question_weight=str(Fraction(1,len(v)))) for f,v in orders.items()})
        for i in range(count)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--menu-report',type=Path,required=True)
    p.add_argument('--menu-report-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.menu_report)!=a.menu_report_sha256:raise ValueError('menu report identity differs')
    report=json.loads(a.menu_report.read_text())
    for path,digest in report['pins'].items():
        if sha(path)!=digest:raise ValueError(f'changed source: {path}')
    orders=report['orders'];panels=complete_panels(orders)
    slot_path=next(Path(k) for k in report['pins'] if k.endswith('slot-support-audit-v1/report.json'))
    slot=json.loads(slot_path.read_text())
    learning_path=next(Path(k) for k in report['pins'] if Path(k).name=='learning.jsonl')
    rows=[json.loads(line) for line in learning_path.read_text().splitlines()]
    if len(rows)!=report['recordings']:raise ValueError('population mismatch')
    counts={t:sum(r['intent']==t for r in rows) for t in orders['intent']}
    n=len(rows)
    output=dict(schema='slurp-complete-definition-panel-design-v1',training_dispatch_allowed=False,
        menu_report=str(a.menu_report),menu_report_sha256=a.menu_report_sha256,
        implementation_sha256=sha(__file__),panels=panels,
        one_population_sweep=dict(independent_recordings=n,source_groups=report['source_groups'],
            forward_requests=n*len(panels),intent_decisions=n*len(orders['intent']),slot_decisions=n*len(orders['slot']),
            final_decisions=n*len(panels),maximum_auxiliary_questions=max(sum(len(f['definitions']) for f in x['families'].values()) for x in panels)),
        objective=dict(final='mean_over_15_panel_conditioned_Final_losses_per_audio',
            intent='family_budget_I_times_sum_of_valid_intent_CE_divided_by_60',
            slot='family_budget_S_times_sum_of_valid_slot_CE_divided_by_55',
            missing='mask_without_redistributing_weight_to_other_questions_or_panels',
            outer='mean_over_real_audio_queries; grouped_repeated_panels_are_not_independent_audio',
            budgets='family budgets and final coefficient must be frozen before implementation; not inferred here',
            optimizer='weights describe one full-sweep objective; streaming optimizer updates are not mathematically identical to one accumulated update'),
        native_annotation_coverage=dict(intent={t:dict(exposures=n,positive_recordings=v) for t,v in sorted(counts.items())},
            slot={r['entity_type']:dict(exposures=n,positive_recordings=r['annotated_recordings']) for r in slot['per_type']}),
        restrictions=['Not a new learning population or semantic supervision admission.',
            'Complete coverage means each definition asked once per audio across 15 separate calls, not all definitions in one forward.',
            'Do not union panel outputs and report single-call full-set metrics.',
            'Actual processor lengths and training throughput unmeasured for this design.',
            'Changing from per-panel mean-valid normalization to fixed family denominator is an explicit objective choice.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['one_population_sweep']))


if __name__=='__main__':main()
