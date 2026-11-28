"""Diagnose a shared calibration intercept on frozen source TRAIN predictions, using CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path, expected=None):
    data = path.read_bytes()
    if expected and digest(data) != expected:
        raise ValueError(f"Frozen input mismatch: {path}")
    return json.loads(data)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_bias(z):
    # One positive among all candidate cards per recording.
    levels, counts = np.unique(z, return_counts=True)
    lo, hi = -float(levels.max()) - 50, -float(levels.min()) + 50
    for _ in range(64):
        mid = (lo + hi) / 2
        if sigmoid(levels + mid) @ counts > len(z):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def fit_affine(z, gold):
    levels, counts = np.unique(z, return_counts=True)
    positives = np.bincount(np.searchsorted(levels, z[np.arange(len(z)), gold]), minlength=len(levels))
    design = np.column_stack((levels, np.ones_like(levels)))
    params = np.array([1.0, fit_bias(z)])
    def objective(v):
        t = design @ v
        return counts @ np.logaddexp(0, t) - positives @ t
    for _ in range(80):
        prob = sigmoid(design @ params)
        grad = design.T @ (counts * prob - positives)
        hessian = design.T @ ((counts * prob * (1 - prob))[:, None] * design)
        step = np.linalg.solve(hessian, grad)
        if np.max(np.abs(step)) < 1e-8:
            break
        weight = 1.0
        while weight > 1e-10:
            proposed = params - weight * step
            if proposed[0] > 0 and objective(proposed) <= objective(params):
                break
            weight *= 0.5
        else:
            raise ValueError('Shared positive slope fit did not descend')
        params = proposed
    if np.max(abs(design.T @ (counts * sigmoid(design @ params) - positives))) / counts.sum() > 1e-7:
        raise ValueError('Calibration fit did not converge')
    return params


def metrics(z, gold):
    idx = np.arange(len(z))
    prob = sigmoid(z)
    truth = np.zeros_like(prob)
    truth[idx, gold] = 1
    rival = np.where(truth, -np.inf, prob).max(axis=1)
    pred = prob.argmax(axis=1)
    tp = np.bincount(gold[pred == gold], minlength=z.shape[1])
    den = (np.bincount(gold, minlength=z.shape[1]) + np.bincount(pred, minlength=z.shape[1]))
    pos = prob[idx, gold]
    return dict(brier=float(np.mean((prob-truth)**2)),
                log_loss=float(np.mean(np.logaddexp(0, z) - truth*z)),
                true_card_mean=float(pos.mean()), strongest_negative_mean=float(rival.mean()),
                top1=float(np.mean(pred == gold)), top1_mean_probability=float(prob.max(axis=1).mean()),
                macro_f1_supported_gold=float(np.mean((2*tp/np.maximum(1, den))[truth.sum(axis=0)>0])))


def fit_grouped_affine(z, gold, groups, folds):
    """Validate on disjoint source groups, then refit on the full calibration split."""
    groups = list(groups)
    if (type(folds) is not int or folds < 2 or len(z) != len(gold) or len(z) != len(groups)
            or not groups or any(not isinstance(g, str) or not g for g in groups)):
        raise ValueError('invalid source TRAIN calibration folds')
    assignment = np.array([int(digest(g.encode()), 16) % folds for g in groups])
    oof = np.empty_like(z)
    details = []
    for i in range(folds):
        valid = assignment == i
        if not valid.any() or valid.all():
            raise ValueError('Empty fit or validation group')
        a, b = fit_affine(z[~valid], gold[~valid])
        oof[valid] = a * z[valid] + b
        details.append(dict(recordings=int(valid.sum()), affine_slope=float(a), affine_bias=float(b)))
    a, b = fit_affine(z, gold)
    if not np.array_equal(z.argmax(1), oof.argmax(1)):
        raise ValueError('grouped positive affine calibration changed ranking')
    return dict(affine_slope=float(a), affine_bias=float(b), affine_out_of_fold=metrics(oof, gold),
                calibration_recordings=len(z), groups=len(set(groups)), folds=details, raw=metrics(z, gold))


def high_scores(z, gold, threshold):
    p = sigmoid(z)
    idx = np.arange(len(p))
    high = p > threshold
    true_high = high[idx, gold]
    many = high.sum(axis=1)
    top = p.argmax(axis=1)
    top_high = many > 0
    def mean(values):
        return float(values.mean()) if values.size else None
    return dict(recordings=len(gold), high_pairs=int(high.sum()), true_high=int(true_high.sum()),
                true_coverage=mean(true_high), high_pair_precision=mean(np.broadcast_to(
                    np.arange(p.shape[1]), p.shape)[high] == np.broadcast_to(gold[:, None], p.shape)[high]),
                high_pair_mean_probability=mean(p[high]), recordings_with_high_negative=int((many>true_high.astype(int)).sum()),
                top1_high=int(top_high.sum()), top1_high_correct=int(((top==gold)&top_high).sum()),
                top1_high_accuracy=mean((top==gold)[top_high]), top1_correct_true_not_high=int(((top==gold)&~true_high).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('score', type=Path)
    parser.add_argument('view', type=Path)
    parser.add_argument('--folds', type=int, required=True)
    parser.add_argument('--audit-sha', required=True)
    parser.add_argument('--view-sha', required=True)
    parser.add_argument('--high-probability', type=float)
    args = parser.parse_args()
    audit = read(args.score/'audit.json', args.audit_sha)
    view = read(args.view/'MANIFEST.json', args.view_sha)
    if not audit['passed'] or audit['policy_id'] != view['source_policy'] or audit['checkpoint_sha256'] != view['card_input']['checkpoint_sha256']:
        raise ValueError('Stage1 identity mismatch')
    roster = read(args.score/'entry/full.json', audit['full_sha256'])
    labels = roster['labels']
    if labels != view['candidate_ids'] or labels != audit['labels']:
        raise ValueError('Card order mismatch')
    ids = {r['id']: i for i, r in enumerate(roster['rows'])}
    p = np.full((len(ids), len(labels)), np.nan)
    for device, worker in enumerate(audit['workers']):
        root = args.score/'collected/score'/f'worker-{device}'
        for block in read(root/'result.json', worker['result_sha256'])['blocks']:
            for row in read(root/block['name'], block['sha256'])['payload']['records']:
                i = ids[row['id']]
                if not np.isnan(p[i, 0]):
                    raise ValueError('Duplicate score')
                p[i] = row['prob']
    if len(ids) != audit['recordings'] or not np.all((p > 0) & (p < 1)):
        raise ValueError('Incomplete or saturated scores')
    rows = read(args.view/'rows.json', view['files']['rows.json'])
    calibration = read(args.view/'calibration.json', view['files']['calibration.json'])
    groups = [r['source_group'] for r in calibration]
    if set(groups) & {r['group'] for r in rows.values()} or {r['audio_sha256'] for r in calibration} & {r['audio_sha256'] for r in rows.values()}:
        raise ValueError('Calibration/mining overlap')
    for r in [*calibration, *rows.values()]:
        meta = roster['rows'][ids[r['id']]]
        if any(r[k] != meta[k] for k in ('audio_sha256', 'intent')):
            raise ValueError('Recording or gold mismatch')
    logits = np.log(p) - np.log1p(-p)
    z = logits[[ids[r['id']] for r in calibration]]
    gold = np.array([labels.index(r['intent']) for r in calibration])
    folds = np.array([int(digest(g.encode()), 16) % args.folds for g in groups])
    oof = np.zeros_like(z)
    affine_oof = np.zeros_like(z)
    fold_results = []
    for i in range(args.folds):
        valid = folds == i
        if not valid.any() or valid.all():
            raise ValueError('Empty fit or validation group')
        b = fit_bias(z[~valid])
        oof[valid] = z[valid] + b
        a2, b2 = fit_affine(z[~valid], gold[~valid])
        affine_oof[valid] = a2 * z[valid] + b2
        fold_results.append(dict(recordings=int(valid.sum()), bias=b, affine_slope=a2, affine_bias=b2))
    b = fit_bias(z)
    a2, b2 = fit_affine(z, gold)
    selected = logits[[ids[k] for k in rows]]
    g = np.array([r['gold'] for r in rows.values()])
    rival = np.array([r['competitor'] for r in rows.values()])
    idx = np.arange(len(g))
    def gap(scores):
        return scores[idx, g] - scores[idx, rival]
    before, after = gap(sigmoid(selected)), gap(sigmoid(selected+b))
    after_affine = gap(sigmoid(a2*selected+b2))
    boundary = np.array([r['stratum'] == 'boundary' for r in rows.values()])
    margin = view['probability_gap']['margin']
    if not np.all((before[boundary]>0) & (before[boundary]<margin)):
        raise ValueError('Boundary mismatch')
    result = dict(audit_sha256=args.audit_sha, view_sha256=args.view_sha, checkpoint=audit['checkpoint_sha256'],
                  affine_slope=a2, affine_bias=b2, affine_out_of_fold=metrics(affine_oof,gold),
                  affine_top1_changed=int((z.argmax(1)!=affine_oof.argmax(1)).sum()), calibration_recordings=len(z), groups=len(set(groups)), candidate_cards=len(labels),
                  supported_gold_cards=len(set(gold)), folds=fold_results, fitted_bias=b,
                  raw=metrics(z,gold), out_of_fold=metrics(oof,gold), margin=margin,
                  top1_changed=int((z.argmax(1)!=oof.argmax(1)).sum()),
                  source_view=dict(rows=len(g), wrong=int((before<0).sum()), tied=int((before==0).sum()),
                                   sign_changes=int((np.sign(after)!=np.sign(before)).sum()),
                                   boundary=int(boundary.sum()), boundary_leaving=int((after[boundary]>=margin).sum()),
                                   affine_boundary_leaving=int((after_affine[boundary]>=margin).sum()),
                                   affine_correct_entering=int(((before>=margin)&(after_affine<margin)).sum()),
                                   correct_entering=int(((before>=margin)&(after<margin)).sum())),
                  constant_prior=dict(brier=(len(labels)-1)/len(labels)**2,
                                      log_loss=float(-np.log(1/len(labels))/len(labels)-
                                                     (1-1/len(labels))*np.log1p(-1/len(labels)))))
    if args.high_probability is not None:
        t = args.high_probability
        if not 0 < t < 1:
            parser.error('high probability must be between zero and one')
        result['high_probability'] = dict(strictly_above=t, raw=high_scores(z,gold,t),
            affine_out_of_fold=high_scores(affine_oof,gold,t),
            selected_train_frozen_affine=high_scores(a2*selected+b2,g,t))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
