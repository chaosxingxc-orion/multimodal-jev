#!/usr/bin/env python3
"""Re-audit every available selection point before summarizing a formal RL run.

Training source, experiment and algorithm are supplied once for the whole curve.
Point inputs only identify collected evaluation artifacts, never alternate runs.
"""
import argparse
import json
from pathlib import Path

if __package__:
    from .audit_context_rl_selection_v2 import audit
    from .context_rl_evaluation_plan import build, sha
    from .context_selection_curve import summarize
else:
    from audit_context_rl_selection_v2 import audit
    from context_rl_evaluation_plan import build, sha
    from context_selection_curve import summarize


def run(*, evaluations, **plan_arguments):
    plan = build(**plan_arguments)
    planned = [p['step'] for p in plan['checkpoints']]
    steps = [p['step'] for p in evaluations]
    if not steps or any(type(s) is not int for s in steps) or steps != planned[:len(steps)]:
        raise ValueError('ordered evaluation prefix from step0 required')
    if any(set(p) != {'step', 'collection', 'transport', 'expected_config_sha256'} for p in evaluations):
        raise ValueError('points cannot override common training source or algorithm')
    roots = [str(Path(p['collection']).resolve()) for p in evaluations]
    handles = [p['transport']['container_id'] for p in evaluations]
    if len(set(roots)) != len(roots) or len(set(handles)) != len(handles):
        raise ValueError('distinct point collections and exact container handles required')
    points, audits = [], []
    for entry, expected in zip(evaluations, plan['checkpoints'], strict=False):
        report = audit(**entry, **plan_arguments)
        # Do not trust a persisted "passed" flag; audit() above rechecks the
        # actual training weights, evaluation records and recomputed metrics.
        if (report['passed'] is not True or report['step'] != expected['step']
                or report['source_policy_id'] != expected['policy_id']
                or report['initial_policy_id'] != plan['initial_policy_id']
                or report['run_id'] != plan['run_id'] or report['algorithm'] != plan['algorithm']
                or report['result']['policy_id'] != report['inference_policy_id']):
            raise ValueError('audited point differs from the common formal run')
        points.append(dict(step=report['step'], result=report['result']))
        audits.append(dict(step=report['step'], source_policy_id=report['source_policy_id'],
            inference_policy_id=report['inference_policy_id'], container_id=report['container_id'],
            collection=str(Path(entry['collection']).resolve()),
            config_sha256=entry['expected_config_sha256'],
            predictions_sha256=report['result']['predictions_sha256']))
    # Serialized export IDs may differ although independently checked tensors
    # are identical. Summary uses actual inference IDs; source identity stays
    # explicit alongside it, rather than relabeling predictions to the source.
    result = summarize(points, plan_arguments['experiment'],
                       initial_policy_id=audits[0]['inference_policy_id'])
    selected = result['selected_checkpoint']
    source_selected = None
    if selected is not None:
        source_selected = next(dict(step=p['step'], policy_id=p['source_policy_id'])
                               for p in audits if p['step'] == selected['step'])
    result.update(run_id=plan['run_id'], algorithm=plan['algorithm'],
        initial_source_policy_id=plan['initial_policy_id'], selected_source_checkpoint=source_selected,
        independently_audited_points=audits, metrics_recomputed=True,
        runtime_provenance_verified=True,
        provenance_scope='Collected policy tensors/processor/adapter, parity, runtime profile and '
            'saved complete-population input receipts verified. Processor tensors are not '
            'independently regenerated. No causal RL, TEST or transfer claim.')
    return result


def from_spec(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    sources = {'spec': dict(path=str(path.resolve()), sha256=sha(path))}
    for name in ('experiment', 'stage1_proof', 'rl_proof'):
        source = Path(spec[name])
        sources[name] = dict(path=str(source.resolve()), sha256=sha(source))
        spec[name] = json.loads(source.read_text())
    for point in spec['evaluations']:
        source = Path(point['transport'])
        sources[f"transport_step{point['step']:04d}"] = dict(path=str(source.resolve()), sha256=sha(source))
        point['transport'] = json.loads(source.read_text())
    report = run(**spec)
    report['input_artifacts'] = sources
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = from_spec(args.spec)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
