import pytest

from tools.reports.card_latency_replay import compare_block


def test_timed_candidate_mismatch_is_not_hidden_by_standard_or_warmup():
    seen = []
    class Scorer:
        candidates = ('a','b')
        def score(self, model, device, provider, policy, rows, check_batch, latency):
            r = list(rows)
            seen.append((self.mode, tuple(r)))
            records = []
            for uid in r:
                latency.visible([{'request_id': uid+':0'},{'request_id':uid+':1'}])
                records.append(dict(id=uid,prob=[.2, .6 if self.mode=='bad' and uid=='a' else .4]))
            return dict(policy_id=policy, records=records)
    b = dict(key='k', ids=['b','a'], widths=[2,2])
    out = compare_block(Scorer(),None,None,None,'p',b,lambda s:s, workers=1,prefetch=2,
                        aligned='bad',reference='hf_singleton',rounds=2,swap=0)
    assert [list(t) for t in out['trials']]==[['standard','candidate'],['candidate','standard']]
    assert all(t['candidate']['differences']==1 and t['standard']['differences']==0 for t in out['trials'])
    assert all(rows == tuple(b['ids']) for _, rows in seen)
    assert all(t['candidate']['first_decision_seconds'] <= t['candidate']['scoring_seconds'] for t in out['trials'])
