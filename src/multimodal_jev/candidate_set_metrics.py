"""Metrics for request-local candidate applicability, not a selection softmax."""
import math


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


class CandidateSetMetrics:
    """Equal query weighting; None truth is missing supervision, never false.

    Candidate IDs may mean different things between requests. Consequently
    macro metrics average requests, not request-local candidate identifiers.
    The caller must authenticate dataset/request identities before supplying
    records. Threshold is fixed by protocol, never fitted here.
    """

    def __init__(self, threshold):
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('fixed threshold strictly between zero and one required')
        self.threshold = threshold
        self.ids = set()
        self.counts = dict(queries=0, candidates=0, observed=0, missing=0, returned=0,
                          returned_unlabeled=0, tp=0, fp=0, fn=0, tn=0,
                          complete=0, exact=0, empty_gold=0, correct_empty=0,
                          predicted_empty_complete=0, nonempty_gold=0, any_hit=0,
                          precision_queries=0, recall_queries=0, f1_queries=0,
                          scored_queries=0)
        self.sums = dict(precision=0., recall=0., f1=0., brier=0., nll=0.)

    def add(self, query_id, truth, probabilities):
        if not isinstance(query_id, str) or not query_id or query_id in self.ids:
            raise ValueError('unique nonempty query ID required')
        if (not isinstance(truth, dict) or not truth or set(truth) != set(probabilities)
                or any(not isinstance(k, str) or not k for k in truth)
                or any(v is not None and type(v) is not bool for v in truth.values())
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in probabilities.values())):
            raise ValueError('complete finite candidate probabilities and Boolean/None labels required')
        # Validate before mutating; a malformed row cannot enter denominators.
        self.ids.add(query_id)
        c, s = self.counts, self.sums
        selected = {k for k, p in probabilities.items() if p >= self.threshold}
        observed = {k for k, y in truth.items() if y is not None}
        positive = {k for k, y in truth.items() if y is True}
        tp = len(selected & positive)
        fp = len((selected & observed) - positive)
        fn = len(positive - selected)
        tn = len(observed) - tp - fp - fn
        for key, value in dict(queries=1, candidates=len(truth), observed=len(observed),
                               missing=len(truth)-len(observed), returned=len(selected),
                               returned_unlabeled=len(selected-observed), tp=tp, fp=fp,
                               fn=fn, tn=tn).items():
            c[key] += value
        for name, num, den in [('precision', tp, tp+fp), ('recall', tp, tp+fn),
                               ('f1', 2*tp, 2*tp+fp+fn)]:
            if den:
                s[name] += num/den
                c[name+'_queries'] += 1
        if observed:
            brier = nll = 0.
            for k in observed:
                p, y = probabilities[k], truth[k]
                brier += (p-int(y))**2
                correct_p = p if y else 1-p
                nll += -math.log(correct_p) if correct_p > 0 else math.inf
            s['brier'] += brier/len(observed)
            s['nll'] += nll/len(observed)
            c['scored_queries'] += 1
        if len(observed) == len(truth):
            c['complete'] += 1
            c['exact'] += selected == positive
            c['predicted_empty_complete'] += not selected
            if positive:
                c['nonempty_gold'] += 1
                c['any_hit'] += bool(selected & positive)
            else:
                c['empty_gold'] += 1
                c['correct_empty'] += not selected

    def result(self):
        c, s = self.counts, self.sums
        if not c['queries']:
            raise ValueError('nonempty evaluation population required')
        nll = _ratio(s['nll'], c['scored_queries'])
        return dict(threshold=self.threshold, threshold_rule='p >= threshold', counts=dict(c),
            micro_precision_observed=_ratio(c['tp'], c['tp']+c['fp']),
            micro_recall_observed=_ratio(c['tp'], c['tp']+c['fn']),
            micro_f1_observed=_ratio(2*c['tp'], 2*c['tp']+c['fp']+c['fn']),
            macro_query_observed={k:_ratio(s[k], c[k+'_queries']) for k in ('precision','recall','f1')},
            exact_set_accuracy_complete=_ratio(c['exact'], c['complete']),
            any_valid_coverage_nonempty_complete=_ratio(c['any_hit'], c['nonempty_gold']),
            empty_set_precision_complete=_ratio(c['correct_empty'], c['predicted_empty_complete']),
            empty_set_recall_complete=_ratio(c['correct_empty'], c['empty_gold']),
            mean_returned_candidates=c['returned']/c['queries'],
            mean_binary_brier_per_observed_query=_ratio(s['brier'], c['scored_queries']),
            mean_binary_nll_per_observed_query=nll if nll is None or math.isfinite(nll) else None,
            binary_nll_infinite=nll is not None and math.isinf(nll),
            scope='Applicability only; missing labels excluded explicitly. No execution-success or calibration guarantee. Undefined ratios are null; macro denominators are reported.')
