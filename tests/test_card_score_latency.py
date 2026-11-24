from multimodal_jev.card_score_latency import ScoreLatency


def test_first_host_score_and_first_complete_recording_can_span_batches():
    timing = ScoreLatency(3, iter([10., 11., 12., 14., 20.]).__next__)
    for group in [('a:2','b:0'), ('c:1','a:0'), ('b:1','b:2','c:2')]:
        timing.visible([dict(request_id=uid) for uid in group])
    assert timing.result() == dict(first_score_seconds=1., first_decision_seconds=4., scoring_seconds=10.)
