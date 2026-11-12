"""Post-hoc mechanism diagnostics on fixed-policy predictions, not rewards."""
from collections import Counter, defaultdict
import math

from .span_rule_dataset import predicted_fact_rule


def diagnose(rows, predictions):
    expected = {r['id']: r for r in rows}
    actual = {r['id']: r for r in predictions}
    if not rows or len(expected) != len(rows) or len(actual) != len(predictions) or expected.keys() != actual.keys():
        raise ValueError('complete unique fixed evaluation population required')
    groups, pairs = defaultdict(list), defaultdict(dict)
    for uid, row in expected.items():
        arm, provenance = row['arm'], row['provenance']
        pred, gold = actual[uid]['predictions'], row['targets']
        logp = actual[uid]['candidate_logprobs']
        if (arm not in {'final_only', 'structured', 'reordered'} or len(pred) != len(gold)
                or len(pred) != (1 if arm == 'final_only' else 3)
                or len(logp) != len(gold) or not all(row['supervised'])
                or any(type(v) is not bool for v in pred+gold)):
            raise ValueError('complete supervised Boolean decisions required')
        for values in logp:
            if (len(values) != 2 or any(not math.isfinite(v) or v > 1e-6 for v in values)
                    or abs(sum(math.exp(v) for v in values)-1) > 1e-5):
                raise ValueError('normalized finite binary log probabilities required')
        if any(pred[i] != bool(max(range(2), key=lambda j: logp[i][j])) for i in range(len(pred))):
            raise ValueError('predictions differ from first-argmax candidate readout')
        fact_correct, executor = None, None
        if arm != 'final_only':
            order = slice(None) if arm == 'structured' else slice(None, None, -1)
            facts, labels = pred[:2][order], gold[:2][order]
            if labels != provenance['dependency_targets']:
                raise ValueError('intermediate question order differs from dependency labels')
            if predicted_fact_rule(provenance['rule'], dict(zip(provenance['dependency_keys'], labels, strict=True))) != gold[-1]:
                raise ValueError('final gold inconsistent with reviewed dependencies')
            executor = predicted_fact_rule(provenance['rule'], dict(zip(provenance['dependency_keys'], facts, strict=True)))
            fact_correct = facts == labels
        item = dict(final_correct=pred[-1] == gold[-1], fact_correct=fact_correct,
                    executor_correct=None if executor is None else executor == gold[-1],
                    consistent=None if executor is None else executor == pred[-1],
                    fact_decisions_correct=sum(a == b for a,b in zip(pred[:-1], gold[:-1], strict=True)),
                    prediction=pred[-1], gold=gold[-1], true_probability=math.exp(logp[-1][1]),
                    final_nll=-logp[-1][int(gold[-1])])
        groups[arm].append(item)
        key = (arm, provenance['recording_id'], provenance['complement_pair_id'])
        polarity = provenance['rule_polarity']
        if polarity not in {'positive', 'negative'} or polarity in pairs[key]:
            raise ValueError('unique positive/negative rule pairs required')
        pairs[key][polarity] = item
    result = {}
    for arm, items in sorted(groups.items()):
        paired = [v for (a,_,_),v in pairs.items() if a == arm]
        if any(set(p) != {'positive','negative'} or p['positive']['gold'] == p['negative']['gold'] for p in paired):
            raise ValueError('complete complementary gold pairs required')
        both = sum(p['positive']['final_correct'] and p['negative']['final_correct'] for p in paired)
        neither = sum(not p['positive']['final_correct'] and not p['negative']['final_correct'] for p in paired)
        one = len(paired)-both-neither
        flips = sum(p['positive']['prediction'] != p['negative']['prediction'] for p in paired)
        if flips != both+neither: raise ValueError('complement confusion identity broken')
        report = dict(requests=len(items), final_accuracy=sum(i['final_correct'] for i in items)/len(items),
            final_true_rate=sum(i['prediction'] for i in items)/len(items),
            mean_final_nll=sum(i['final_nll'] for i in items)/len(items),
            complement=dict(pairs=len(paired), both_correct=both, both_wrong=neither, exactly_one_correct=one,
                prediction_flips=flips,
                mean_probability_sum_error=sum(abs(p['positive']['true_probability']+p['negative']['true_probability']-1) for p in paired)/len(paired),
                mean_absolute_probability_change=sum(abs(p['positive']['true_probability']-p['negative']['true_probability']) for p in paired)/len(paired)))
        if arm != 'final_only':
            table = Counter((i['fact_correct'], i['final_correct']) for i in items)
            correct = sum(i['fact_correct'] for i in items)
            report.update(fact_pair_accuracy=correct/len(items),
                fact_decision_accuracy=sum(i['fact_decisions_correct'] for i in items)/(2*len(items)),
                facts_by_final=[[table[bool(f),bool(y)] for y in range(2)] for f in range(2)],
                final_errors_with_both_facts_correct=table[True,False],
                final_accuracy_given_both_facts_correct=table[True,True]/correct if correct else None,
                deterministic_executor_accuracy=sum(i['executor_correct'] for i in items)/len(items),
                final_executor_consistency=sum(i['consistent'] for i in items)/len(items))
        result[arm] = report
    return dict(arms=result, scope='Post-hoc fixed-policy error decomposition; no causal attribution, '
        'reward update, checkpoint selection or additional model inference. Deterministic executor is a diagnostic only.')
