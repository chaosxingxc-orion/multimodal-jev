"""Frozen card assignments. Each recording alternates rules and examples across the registered epochs."""
from dataclasses import replace

from .context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest


def from_panel(panel):
    return ContextDecisionRequest('audio', tuple(ApplicabilityQuestion(**q) for q in panel['questions']),
                                  tuple(panel['requirements']), tuple(panel['history']), panel['version'])


class CardLayout:
    def __init__(self, plan, ids, task, *, epochs):
        self.requests = {key: from_panel(panel) for key, panel in plan['panels'].items()}
        self.assignments = plan['assignments']
        labels = [c['id'] for c in task['schema']['candidates']]
        if (type(epochs) is not int or epochs < 1 or set(self.assignments) != set(ids) or any([q.id for q in r.questions] != labels
                for r in self.requests.values()) or any(len(pair) != epochs or
                any((a == 'rules') == (b == 'rules') for a, b in zip(pair, pair[1:])) or
                any(k not in self.requests or k == 'native' for k in pair) for pair in self.assignments.values())):
            raise ValueError('incomplete alternating card assignment or different readout order')

    def __call__(self, audio, uid, epoch):
        return replace(self.requests[self.assignments[uid][epoch]], audio=audio)
