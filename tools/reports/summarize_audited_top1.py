"""Report single-label Top-1 scores from previously audited predictions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def verified_rows(path, *, sha256, recordings, decisions, policy_id, hf=False):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('predictions differ from the independent audit')
    body = json.loads(raw)
    if hf and body['policy_id'] != policy_id:
        raise ValueError('HF predictions differ from the frozen policy')
    rows = body['records']
    if type(recordings) is not int or recordings <= 0 or len(rows) != recordings:
        raise ValueError('complete audited population required')
    ids = set()
    for row in rows:
        uid = row['id']
        probabilities = row['prob']
        actual_policy = row['request_identity']['policy_id'] if hf else row['policy_id']
        if (type(uid) is not str or not uid or uid in ids or actual_policy != policy_id
                or type(decisions) is not int or decisions <= 0 or len(probabilities) != decisions
                or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in probabilities)):
            raise ValueError('prediction ID, policy or candidate probabilities differ')
        ids.add(uid)
        yield row


def summarize_predictions(path, *, sha256, recordings, decisions, policy_id):
    rows = verified_rows(path, sha256=sha256, recordings=recordings, decisions=decisions, policy_id=policy_id)
    return math.fsum(max(row['prob']) for row in rows) / recordings


def report(audit, mean_probability, *, hf=False):
    expected_status = 'complete_hf_shards_audited' if hf else 'complete_macro_evaluation_outputs_audited'
    if (audit['status'] != expected_status
            or audit['target_tuning_allowed'] is not False
            or audit['metrics']['queries'] != audit['recordings']
            or not 0 <= mean_probability <= 1):
        raise ValueError('complete independent scoring audit required')
    return dict(dataset=audit['dataset'], step=audit['step'], recordings=audit['recordings'],
                policy_id=audit['policy_id'], predictions_sha256=audit['predictions_sha256'],
                top1_mean_probability=mean_probability,
                top1_accuracy=audit['metrics']['top1_event_accuracy'],
                top1_macro_f1=audit['metrics']['top1_macro_f1'])


def rescore(path, audit, *, seal_root, hf_policy_id=None):
    """Rebuild the argmax confusion counts against the pinned sealed labels."""
    root = Path(seal_root)

    def checked(rel, expected):
        raw = (root/rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('sealed scoring input changed: '+rel)
        return json.loads(raw)

    seal = checked('MANIFEST.json', audit['seal_sha256'])
    dataset = audit['dataset']
    if dataset not in seal['datasets'] or seal['target_tuning_allowed'] is not False:
        raise ValueError('fixed evaluation dataset required')
    panel_path, rows_path, targets_path = (f'inference/{dataset}/panel.json',
        f'inference/{dataset}/rows.json', f'scoring/{dataset}/targets.json')
    panel, roster, targets = (checked(rel, seal['files'][rel]) for rel in (panel_path, rows_path, targets_path))
    candidates = panel['candidate_ids']
    hf = hf_policy_id is not None
    if hf:
        if audit['status'] != 'complete_hf_shards_audited' or audit['runtime'] != 'HF_standalone':
            raise ValueError('complete independent HF shard audit required')
        audit = dict(audit, policy_id=hf_policy_id, recordings=seal['datasets'][dataset]['recordings'],
                     decisions=seal['datasets'][dataset]['candidates'])
    recording_count = audit['recordings']
    if (audit['metrics']['candidate_ids'] != candidates or len(candidates) != audit['decisions']
            or not candidates or len(set(candidates)) != len(candidates)
            or seal['datasets'][dataset]['recordings'] != recording_count
            or len(roster) != recording_count or len(targets) != recording_count):
        raise ValueError('candidate order or population differs from the sealed audit')

    predictions = verified_rows(path, sha256=audit['predictions_sha256'], recordings=recording_count,
        decisions=audit['decisions'], policy_id=audit['policy_id'], hf=hf)
    golds, guesses, hits = Counter(), Counter(), Counter()
    maxima = []
    empty = 0
    threshold = seal['threshold']
    for row, asset, truth in zip(predictions, roster, targets, strict=True):
        if row['id'] != asset['id'] or any(asset[k] != truth[k] for k in ('id', 'audio_sha256')):
            raise ValueError('sealed label or roster differs from the prediction')
        gold = truth['gold']
        if gold not in candidates:
            raise ValueError('sealed label outside candidate panel')
        top = max(range(len(candidates)), key=row['prob'].__getitem__)
        guess = candidates[top]
        probability = row['prob'][top]
        maxima.append(probability)
        golds[gold] += 1
        guesses[guess] += 1
        if guess == gold:
            hits[gold] += 1
        empty += probability < threshold

    result = report(audit, math.fsum(maxima)/recording_count, hf=hf)
    accuracy = sum(hits.values()) / recording_count
    macro_f1 = math.fsum(2*hits[k]/(golds[k]+guesses[k]) if golds[k]+guesses[k] else 0.0
                        for k in candidates) / len(candidates)
    if (accuracy != result['top1_accuracy'] or macro_f1 != result['top1_macro_f1']
            or (hf and result['top1_mean_probability'] != audit['metrics']['mean_top1_probability'])
            or empty/recording_count != audit['metrics']['empty_selection_rate']):
        raise ValueError('independent raw-prediction rescore differs from audit')
    return dict(result, top1_correct=sum(hits.values()), top1_empty=0,
                top1_metrics_independently_recomputed=True, seal_sha256=audit['seal_sha256'],
                targets_sha256=seal['files'][targets_path],
                threshold_diagnostic=dict(threshold=threshold, empty=empty, empty_rate=empty/recording_count))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--seal-root', help='Independently recompute Top-1 against pinned sealed labels')
    p.add_argument('--hf-policy-id', help='Frozen policy identity for independently audited HF shards')
    a = p.parse_args()
    if a.hf_policy_id is not None and not a.seal_root:
        p.error('--hf-policy-id requires --seal-root')
    audited = json.loads(Path(a.audit).read_text())
    if a.seal_root:
        result = rescore(a.predictions, audited, seal_root=a.seal_root, hf_policy_id=a.hf_policy_id)
    else:
        mean = summarize_predictions(a.predictions, sha256=audited['predictions_sha256'],
                                    recordings=audited['recordings'], decisions=audited['decisions'],
                                    policy_id=audited['policy_id'])
        result = report(audited, mean)
    print(json.dumps(result, indent=2))
