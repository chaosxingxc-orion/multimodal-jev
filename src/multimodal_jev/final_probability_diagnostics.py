"""Descriptive probability quality of complete native Final60 predictions.

No threshold fitting, checkpoint selection, training reward or new inference.
Categorical probabilities here are distinct from independent Jev applicability.
"""
import math

from .simple_final_metrics import score


def diagnose(rows, classes, records, *, policy_id, bins=10):
    if type(bins) is not int or bins <= 0:
        raise ValueError('positive fixed bin count required')
    records = list(records)
    metrics = score(rows, classes, records, policy_id=policy_id)
    if any(r['status'] != 'completed' or r.get('candidate_logprobs') is None for r in records):
        raise ValueError('complete probability coverage required; do not drop failed records')
    truth = {r['id']: r['intent'] for r in rows}
    indices = {c: i for i, c in enumerate(classes)}
    buckets = [dict(count=0, correct=0, confidence_sum=0.) for _ in range(bins)]
    by_class = {c: dict(support=0, nll_sum=0., brier_sum=0.) for c in classes}
    nll_sum = brier_sum = wrong_confidence_sum = 0.
    wrong_count = confident_wrong = 0
    residual = 0.
    for record in records:
        logp = record['candidate_logprobs']
        probabilities = [math.exp(x) for x in logp]
        total = sum(probabilities)
        residual = max(residual, abs(total - 1.))
        # Renormalize only tolerated storage rounding; no floor/temperature fit.
        probabilities = [p / total for p in probabilities]
        gold = indices[truth[record['id']]]
        nll = -logp[gold] + math.log(total)
        brier = sum((p - (i == gold)) ** 2 for i, p in enumerate(probabilities))
        confidence = max(probabilities)
        correct = record['prediction'] == truth[record['id']]
        bucket = buckets[min(int(confidence * bins), bins - 1)]
        bucket['count'] += 1
        bucket['correct'] += int(correct)
        bucket['confidence_sum'] += confidence
        nll_sum += nll
        brier_sum += brier
        entry = by_class[truth[record['id']]]
        entry['support'] += 1
        entry['nll_sum'] += nll
        entry['brier_sum'] += brier
        if not correct:
            wrong_count += 1
            wrong_confidence_sum += confidence
            confident_wrong += confidence >= .9
    count = len(rows)
    reliability = [dict(lower=i/bins, upper=(i+1)/bins, count=b['count'],
        accuracy=b['correct']/b['count'] if b['count'] else None,
        confidence=b['confidence_sum']/b['count'] if b['count'] else None)
        for i, b in enumerate(buckets)]
    predicted = sorted(metrics['per_class'].items(), key=lambda kv: (-kv[1]['predicted'], kv[0]))
    return dict(recordings=count, accuracy=metrics['accuracy'], macro_F1_fixed60=metrics['macro_F1_fixed60'],
        nll_nats=nll_sum/count, multiclass_brier_sum_over60=brier_sum/count,
        top_label_ece=sum(abs(b['confidence_sum']-b['correct']) for b in buckets)/count,
        reliability_bins=reliability, normalization_residual_max=residual,
        incorrect_mean_confidence=wrong_confidence_sum/wrong_count if wrong_count else None,
        incorrect_confidence_ge_0_9=confident_wrong,
        incorrect_confidence_ge_0_9_population_fraction=confident_wrong/count,
        predicted_classes=sum(v['predicted'] > 0 for v in metrics['per_class'].values()),
        zero_recall_supported_classes=sum(v['support'] > 0 and v['correct'] == 0 for v in metrics['per_class'].values()),
        top2_prediction_fraction=sum(v['predicted'] for _, v in predicted[:2])/count,
        top_predictions=[dict(candidate=k, count=v['predicted']) for k, v in predicted[:5]],
        per_class={c: dict(support=v['support'], nll_nats=v['nll_sum']/v['support'] if v['support'] else None,
                          brier=v['brier_sum']/v['support'] if v['support'] else None) for c, v in by_class.items()},
        scope='Descriptive categorical probability quality; not independent applicability calibration or RL efficacy.')
