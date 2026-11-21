import unittest
from multimodal_jev.decision_evidence import adjudicate, DecisionEvidence, metric


class EvidenceTests(unittest.TestCase):
    def test_counterfactual_and_missing_annotation(self):
        self.assertEqual(adjudicate([True, False, None],exhaustive=False),DecisionEvidence(False,None))
        self.assertEqual(adjudicate([True,None],exhaustive=True),DecisionEvidence(None,None))
        self.assertEqual(adjudicate([False,False],exhaustive=False),DecisionEvidence(None,None))
        self.assertEqual(adjudicate([False,False],exhaustive=True),DecisionEvidence(True,False))

    def test_risk_counts_unsupported_execution_and_accepts_known_negative(self):
        report=metric([(DecisionEvidence(False,None),.9,True),(DecisionEvidence(False,None),.2,True),
                       (DecisionEvidence(True,False),.8,False),(DecisionEvidence(True,True),.1,True),
                       (DecisionEvidence(None,None),1.,False)],threshold=.5)
        self.assertEqual((report['risk'],report['coverage'],report['unknown_recall'],
                          report['false_rejection_rate'],report['masked']),(.5,.5,.5,.5,1))
