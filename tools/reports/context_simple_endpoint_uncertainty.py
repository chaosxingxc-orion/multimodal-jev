"""Read-only paired uncertainty for a completed simple RL run's fixed endpoint.

Does not select checkpoints, authorize complex training, or mutate queues.
"""
import argparse
import json
from pathlib import Path
from tools.reports.compare_context_arithmetic_curves import load_run
from tools.reports.context_attribute_paired_bootstrap import compare_records
from tools.jev.context_rl_evaluation_plan import sha
from multimodal_jev.context_event_metrics import ContextEventMetrics


def score_pair(initial, final, labels, candidates, policies):
    ids = [r['id'] for r in labels]
    if (not ids or len(ids) != len(set(ids)) or len(policies) != 2
            or any(r['role'] != 'selection' for r in labels)
            or any([r['id'] for r in side] != ids for side in (initial, final))):
        raise ValueError('complete ordered selection population required')
    sides = []; metrics = []
    for predictions, policy in zip((initial, final), policies, strict=True):
        scorer = ContextEventMetrics(candidates)
        records = []
        for item, target in zip(predictions, labels, strict=True):
            if item['policy_id'] != policy: raise ValueError('prediction policy differs')
            p = item['prob']; gold = target['intent']
            scorer.add(item['id'], gold, p)
            chosen = [c for c,v in zip(candidates,p,strict=True) if v >= .5]
            records.append(dict(id=item['id'],source_group=target['source_group'],gold=gold,
                structural_slice='exclusive_event',set_f1=2*int(gold in chosen)/(len(chosen)+1),
                exact_set=chosen==[gold],top1_correct=candidates[max(range(len(p)),key=p.__getitem__)]==gold))
        sides.append(records); metrics.append(scorer.result())
    for a,b in zip(initial,final,strict=True):
        if any(a['request_identity'][k] != b['request_identity'][k] for k in ('inputs','positions')):
            raise ValueError('executed inputs differ across endpoints')
    return compare_records(*sides,endpoint='fixed_step634'),metrics


def run(directory, algorithm, release):
    # Reuse the actual full-run auditor, not a stored passed flag.
    audited = load_run(directory, algorithm)
    curve = audited['curve']; root = Path(release)
    manifest = json.loads((root/'MANIFEST.json').read_text())
    sources={str(root/'MANIFEST.json'):sha(root/'MANIFEST.json')}
    if sources[str(root/'MANIFEST.json')] != audited['experiment']['release_sha256']:
        raise ValueError('wrong source release')
    for name in ('selection.jsonl','tasks.json'):
        sources[str(root/name)]=sha(root/name)
        if sources[str(root/name)] != manifest['files'][name]:raise ValueError('release file differs')
    labels=[json.loads(line) for line in (root/'selection.jsonl').read_text().splitlines()]
    if len(labels)!=9226:raise ValueError('full selection required')
    candidates=[r['id'] for r in json.loads((root/'tasks.json').read_text())['intent']['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('Final60 required')
    points=[];predictions=[]
    for step in (0,634):
        point=next(p for p in curve['independently_audited_points'] if p['step']==step)
        path=Path(point['collection'])/'evaluation/selection/predictions.json'
        if sha(path)!=point['predictions_sha256']:raise ValueError('predictions changed')
        sources[str(path)]=sha(path);points.append(point)
        predictions.append(json.loads(path.read_text())['records'])
    uncertainty,metrics=score_pair(*predictions,labels,candidates,[p['inference_policy_id'] for p in points])
    for observed,step in zip(metrics,(0,634),strict=True):
        expected=next(p['metrics'] for p in curve['curve'] if p['step']==step)
        for key in ('query_macro_set_f1','exact_set','top1_event_accuracy','candidate_binary_nll','candidate_brier'):
            a,b=observed[key],expected[key]
            if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-12):
                raise ValueError('endpoint scoring differs from independent curve: '+key)
    for path in (Path(__file__),Path('tools/reports/context_attribute_paired_bootstrap.py')):sources[str(path)]=sha(path)
    return dict(schema='context-simple-fixed-endpoint-uncertainty-v1',algorithm=algorithm,
        provenance=audited['provenance'],sources=sources,uncertainty=uncertainty,
        probability_quality={'initial':{k:metrics[0][k] for k in ('candidate_binary_nll','binary_nll_infinite','candidate_brier')},
                             'final':{k:metrics[1][k] for k in ('candidate_binary_nll','binary_nll_infinite','candidate_brier')}},
        complex_training_admitted=False,
        scope='Fixed step634 minus step0, paired source-group sampling uncertainty only. '
              'NLL/Brier are point values, not bootstrap intervals. No seed variance, chosen-checkpoint coverage, '
              'pure-RL attribution or automatic complex-training approval.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('directory','algorithm','release','output'):parser.add_argument('--'+key,required=True)
    args=vars(parser.parse_args());out=Path(args.pop('output'));result=run(**args)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['uncertainty'],indent=2))
