from tools.reports.analyze_context_reference_reuse import count_reuse


def batch(epoch,*values):
    return dict(epoch=epoch,slots=[dict(id=uid,padding=pad,weight=0. if pad else 1.) for uid,pad in values])


def test_padding_is_still_scored_but_same_input_only_fills_once():
    steps,seen=count_reuse([batch(0,('a',False),('a',True),('b',False)),
        batch(1,('b',False),('a',False))],{'a':'tensor-a','b':'tensor-b'})
    assert steps[0]['new_reference_forwards']==2
    assert steps[0]['slots']==3 and steps[0]['padding']==1
    assert steps[0]['same_batch_duplicate_slots']==1
    assert steps[1]['new_reference_forwards']==0 and steps[1]['prior_cache_slots']==2
    warm,_=count_reuse([batch(0,('a',False),('b',False))],{'a':'tensor-a','b':'tensor-b'},initial=seen)
    assert warm[0]['new_reference_forwards']==0


def test_encoded_identity_not_recording_name_controls_reuse():
    steps,_=count_reuse([batch(0,('a',False),('b',False),('c',False))],
        {'a':'same-tensors','b':'same-tensors','c':'different-context'})
    assert steps[0]['new_reference_forwards']==2
