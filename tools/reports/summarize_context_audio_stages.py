"""Deterministic concise report of independently compared acoustic stages."""
import argparse
import hashlib
import json
from pathlib import Path


STAGES = ['encoder_input', *[f'layer_{i:02d}' for i in range(18)], 'ln_post', 'projected']


def summarize(report):
    if report['status'] != 'actual_model_stage_comparison' or not report['rows']:
        raise ValueError('completed real model comparison required')
    result = dict(queries=len(report['rows']), modes={}, full_model_admitted=False)
    for mode in ('baseline', 'acoustic'):
        batches = report['batch'][mode]
        if any(name not in batches for name in STAGES):
            raise ValueError('complete acoustic stages required')
        first = next((name for name in STAGES if not batches[name].get('exact')), None)
        singles = []
        for row in report['rows']:
            stages = row['comparisons'][mode]
            if any(name not in stages for name in STAGES):
                raise ValueError('complete single-record stages required')
            divergence = next((name for name in STAGES if not stages[name].get('exact')), None)
            if divergence:
                singles.append(dict(id=row['id'], first_stage=divergence))
        scores = {}
        for part in ('records', 'batch-records'):
            values = report['scores'][mode][part]
            if [v['id'] for v in values] != [r['id'] for r in report['rows']]:
                raise ValueError('score population mismatch')
            scores[part] = dict(different_queries=sum(not v['logits']['exact'] for v in values),
                max_logit=max(v['logits']['max_abs'] for v in values),
                max_prob=max(v['prob']['max_abs'] for v in values))
        result['modes'][mode] = dict(single_acoustic_differences=singles,
            batch_first_divergence=first, batch_first_difference=batches[first] if first else None,
            scores=scores)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    from multimodal_jev.local_checkpoint import commit_json
    report = json.loads(a.report.read_text()); result = summarize(report)
    result['report_sha256'] = hashlib.sha256(a.report.read_bytes()).hexdigest()
    commit_json(a.output, result)
    print(json.dumps(result, indent=2))
