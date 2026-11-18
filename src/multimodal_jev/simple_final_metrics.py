"""Complete-population native Final metrics; no partial-run success or label repair."""
from collections import Counter
import math


def score(rows, classes, records, *, policy_id):
    if not rows or len(classes) != 60 or len(set(classes)) != 60 or not policy_id:
        raise ValueError('full native60 population and policy identity required')
    truth = {r['id']: r for r in rows}
    if len(truth) != len(rows):
        raise ValueError('duplicate population identity')
    seen = set(); support = Counter(); predicted = Counter(); correct = Counter()
    invalid = failed = 0
    for record in records:
        rid = record['id']
        if rid not in truth or rid in seen:
            raise ValueError('extra or duplicate prediction')
        seen.add(rid); row = truth[rid]
        if (record.get('policy_id') != policy_id or record.get('audio_sha256') != row['audio_sha256']
                or record.get('view') != 'direct' or record.get('variant') != 'native'):
            raise ValueError('policy/audio/task identity differs')
        if row['intent'] not in classes:
            raise ValueError('gold outside fixed native support')
        gold = row['intent']; support[gold] += 1
        status = record.get('status')
        if status not in ('completed', 'failed'):
            raise ValueError('nonterminal record; do not silently drop it')
        choice = None
        if status == 'failed':
            if not isinstance(record.get('error'), str) or not record['error'].strip():
                raise ValueError('explicit failure required')
            failed += 1
        else:
            choice = record.get('prediction')
            values = record.get('candidate_logprobs')
            if values is not None:
                if (record.get('candidate_values') != classes or not isinstance(values, list)
                        or len(values) != 60 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                        or abs(sum(math.exp(v) for v in values if v <= 0)-1) > 1e-5
                        or any(v > 1e-6 for v in values)):
                    raise ValueError('complete normalized ordered native distribution required')
                expected = classes[max(range(60), key=lambda i: values[i])]
                if choice != expected:
                    raise ValueError('reported prediction disagrees with fixed argmax')
        if type(choice) is not str or choice not in classes:
            invalid += 1
        else:
            predicted[choice] += 1
            correct[gold] += choice == gold
    if seen != set(truth):
        raise ValueError('incomplete population; scoring blocked')
    per = {c: dict(support=support[c], predicted=predicted[c], correct=correct[c],
        recall=correct[c]/support[c] if support[c] else None,
        f1=2*correct[c]/(support[c]+predicted[c]) if support[c]+predicted[c] else 0.) for c in classes}
    return dict(recordings=len(rows), correct=sum(correct.values()), accuracy=sum(correct.values())/len(rows),
        macro_F1_fixed60=sum(v['f1'] for v in per.values())/60,
        macro_recall_supported=sum(v['recall'] for v in per.values() if v['support'])/sum(v['support']>0 for v in per.values()),
        invalid=invalid, failed=failed, invalid_rate=invalid/len(rows), per_class=per,
        scope='Native single Final, complete supplied population. Caller must authenticate release, raw prediction receipts and policy/input provenance.')
