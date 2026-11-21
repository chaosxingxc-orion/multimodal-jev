"""Question-specific supervision by agreement across compatible factual worlds.

The worlds are explicitly supplied by a protocol, never inferred from a model
mistake. A missing/unreviewed annotation is masked, not negative evidence.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionEvidence:
    sufficient: bool | None
    answer: bool | None


def adjudicate(answers, *, exhaustive):
    if type(exhaustive) is not bool:
        raise ValueError('explicit completion status required')
    values = tuple(answers)
    if not values or any(v is not None and type(v) is not bool for v in values):
        raise ValueError('nonempty bool/null decision panel required')
    if any(x is True for x in values) and any(x is False for x in values):
        return DecisionEvidence(False, None)  # two constructive counterexamples suffice
    if not exhaustive or None in values:
        return DecisionEvidence(None, None)
    return DecisionEvidence(True, values[0])


def metric(rows, *, threshold):
    """UNKNOWN precision/recall, wrongful rejection, accepted risk and coverage."""
    if not 0<=threshold<=1:
        raise ValueError('frozen probability cutoff required')
    c=dict(total=0,masked=0,unknown=0,rejected=0,correct_rejection=0,rejected_answerable=0,
           accepted=0,accepted_errors=0)
    for label,p_sufficient,answer in rows:
        if type(p_sufficient) not in (float,int) or not 0<=p_sufficient<=1 or type(answer) is not bool:
            raise ValueError('valid numeric sufficiency and binary decision required')
        if label.sufficient is None:
            c['masked']+=1;continue
        if (type(label.sufficient) is not bool or
                ((type(label.answer) is bool) != label.sufficient)):
            raise ValueError('invalid sufficiency supervision')
        c['total']+=1
        c['unknown']+=not label.sufficient
        if p_sufficient<threshold:
            c['rejected']+=1;c['correct_rejection']+=not label.sufficient
            c['rejected_answerable']+=label.sufficient
        else:
            c['accepted']+=1
            c['accepted_errors']+=(not label.sufficient or answer!=label.answer)
    def ratio(n,d):
        return n/d if d else None
    return dict(**c,coverage=ratio(c['accepted'],c['total']),
                unknown_precision=ratio(c['correct_rejection'],c['rejected']),
                unknown_recall=ratio(c['correct_rejection'],c['unknown']),
                false_rejection_rate=ratio(c['rejected_answerable'],c['total']-c['unknown']),
                risk=ratio(c['accepted_errors'],c['accepted']))
