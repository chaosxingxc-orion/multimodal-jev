"""Frozen card assignments. A recording sees each training arm exactly once."""
from dataclasses import replace

from .context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest


def from_panel(panel):
    return ContextDecisionRequest('audio', tuple(ApplicabilityQuestion(**q) for q in panel['questions']),
                                  tuple(panel['requirements']), tuple(panel['history']), panel['version'])


class CardLayout:
    def __init__(self, plan, ids, task):
        self.requests = {key: from_panel(panel) for key, panel in plan['panels'].items()}
        self.assignments = plan['assignments']
        labels = [c['id'] for c in task['schema']['candidates']]
        if (set(self.assignments) != set(ids) or any([q.id for q in r.questions] != labels
                for r in self.requests.values()) or any(len(pair) != 2 or pair.count('rules') != 1 or
                any(k not in self.requests or k == 'native' for k in pair) for pair in self.assignments.values())):
            raise ValueError('incomplete two-epoch card assignment or different readout order')

    def __call__(self, audio, uid, epoch):
        return replace(self.requests[self.assignments[uid][epoch]], audio=audio)
