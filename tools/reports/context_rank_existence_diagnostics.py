"""Saved-output ranking/existence decomposition for exclusive-event tasks.

Diagnostic only: labels identify benchmark membership, not semantic presence.
No model execution, checkpoint selection, probability clipping or tuning.
"""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports.prepare_context_dual_throughput_pair import sha, write


def query_scores(probabilities, gold):
    if (len(probabilities) < 2 or type(gold) is not int or not 0 <= gold < len(probabilities)
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities)):
        raise ValueError('finite complete probabilities and one positive candidate required')
    positive = probabilities[gold]; negatives = [p for i, p in enumerate(probabilities) if i != gold]
    wins = sum(positive > p for p in negatives); ties = sum(positive == p for p in negatives)
    return dict(pair_accuracy=(wins+.5*ties)/len(negatives), pair_tie_fraction=ties/len(negatives),
        positive_nll=None if positive == 0 else -math.log(positive),
        negative_nll=None if 1 in negatives else -math.fsum(math.log1p(-p) for p in negatives)/len(negatives),
        positive_brier=(1-positive)**2,
        negative_brier=math.fsum(p*p for p in negatives)/len(negatives),
        false_reject=float(positive < .5), false_accept_rate=sum(p >= .5 for p in negatives)/len(negatives),
        false_accept_count=sum(p >= .5 for p in negatives),
        positive_probability=positive, hardest_negative_probability=max(negatives))


def summarize(records, labels, candidates, policy_id):
    ids = [r['id'] for r in labels]
    if (not ids or len(set(ids)) != len(ids) or len(set(candidates)) != len(candidates)
            or [r['id'] for r in records] != ids or any(r['role'] != 'selection' for r in labels)):
        raise ValueError('complete ordered selection population and unique candidates required')
    rows = []
    for record, label in zip(records, labels, strict=True):
        if record['policy_id'] != policy_id or len(record['prob']) != len(candidates):
            raise ValueError('fixed policy and full candidate vector required')
        rows.append(query_scores(record['prob'], candidates.index(label['intent'])))
    means = {}; infinite = {}
    for key in rows[0]:
        infinite[key] = sum(row[key] is None for row in rows)
        means[key] = None if infinite[key] else math.fsum(row[key] for row in rows)/len(rows)
    width = len(candidates)
    means['candidate_binary_nll_reconstructed'] = (
        None if means['positive_nll'] is None or means['negative_nll'] is None
        else (means['positive_nll']+(width-1)*means['negative_nll'])/width)
    means['candidate_brier_reconstructed'] = (means['positive_brier']+(width-1)*means['negative_brier'])/width
    return dict(queries=len(rows), candidates=width, means=means,
        infinite_nll_queries={k:infinite[k] for k in ('positive_nll','negative_nll')})


def run(c):
    curve_path = Path(c['curve']); release = Path(c['release'])
    if sha(curve_path) != c['curve_sha256'] or sha(release/'MANIFEST.json') != c['release_sha256']:
        raise ValueError('configured curve and release identities required')
    curve = json.loads(curve_path.read_text()); manifest = json.loads((release/'MANIFEST.json').read_text())
    if curve['complete'] is not True or curve['algorithm'] != c['algorithm']:
        raise ValueError('complete expected algorithm curve required')
    sources = {str(curve_path):c['curve_sha256'], str(release/'MANIFEST.json'):c['release_sha256']}
    for name in ('selection.jsonl','tasks.json'):
        if sha(release/name) != manifest['files'][name]:raise ValueError('release asset changed')
        sources[str(release/name)] = sha(release/name)
    labels = [json.loads(line) for line in (release/'selection.jsonl').read_text().splitlines()]
    candidates = [r['id'] for r in json.loads((release/'tasks.json').read_text())['intent']['schema']['candidates']]
    if len(labels) != 9226 or len(candidates) != 60:raise ValueError('full SLURP selection Final60 required')
    reports = []; previous = None
    for step in (0,634):
        point = next(p for p in curve['independently_audited_points'] if p['step'] == step)
        path = Path(point['collection'])/'evaluation/selection/predictions.json'
        if sha(path) != point['predictions_sha256']:raise ValueError('prediction identity differs')
        sources[str(path)] = point['predictions_sha256']
        records = json.loads(path.read_text())['records']
        report = summarize(records, labels, candidates, point['inference_policy_id'])
        if previous is not None and any(a['request_identity'][k] != b['request_identity'][k]
                for a,b in zip(previous, records, strict=True) for k in ('inputs','positions')):
            raise ValueError('endpoint executed inputs differ')
        previous = records
        metrics = next(p['metrics'] for p in curve['curve'] if p['step'] == step)
        for key in ('candidate_binary_nll','candidate_brier'):
            value = report['means'][key+'_reconstructed']; expected = metrics[key]
            if (value is None) != (expected is None) or (value is not None and abs(value-expected)>1e-12):
                raise ValueError('decomposed proper score differs from curve')
        reports.append(dict(step=step, policy_id=point['inference_policy_id'], **report))
    delta = {k:None if any(r['means'][k] is None for r in reports) else reports[1]['means'][k]-reports[0]['means'][k]
             for k in reports[0]['means']}
    return dict(schema='context-rank-existence-endpoint-diagnostic-v1', algorithm=c['algorithm'],
        endpoints=reports, final_minus_initial=delta, sources=sources, implementation_sha256=sha(__file__),
        ranking_tie_credit=.5, acceptance_threshold=.5, diagnostic_only=True, training_admitted=False,
        full_runtime_reaudited=False,
        scope='Complete saved predictions bound to pinned curve and release; proper scores independently reconstructed. '
              'Pair accuracy is gold-versus-each-negative with half-credit ties, not top1 or calibration. '
              'No runtime rerun, causal attribution, seed uncertainty or semantic-presence claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    a = p.parse_args(); c, provenance = load_config(a.config, a.local)
    result = run(c); result['config_provenance'] = provenance; write(c['output'], result)
    print(json.dumps(result['final_minus_initial'], indent=2))
