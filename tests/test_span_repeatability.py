import pytest
from multimodal_jev.span_repeatability import REPEAT_SEQUENCE, compare_repeats


def records():
    return [dict(phase=p,candidate_logits=[[1.,2.],[3.,4.]]) for p in REPEAT_SEQUENCE]


def test_exact_repeated_logits():
    report=compare_repeats(records())
    assert report['all_logits_exact'] and len(report['pairs'])==10


def test_trace_shift_detected_even_when_normalized_probabilities_match():
    rows=records(); rows[2]['candidate_logits']=[[2.,3.],[4.,5.]]
    report=compare_repeats(rows)
    assert not report['all_logits_exact']
    affected=[r for r in report['pairs'] if 'trace' in (r['left'],r['right'])]
    assert len(affected)==4 and all(r['maximum_logit_difference']==1 for r in affected)
    assert all(r['maximum_logprob_difference']==0 for r in affected)


def test_post_trace_persistent_change_and_order_validation():
    rows=records(); rows[3]['candidate_logits'][0][0]=2.
    assert not compare_repeats(rows)['all_logits_exact']
    with pytest.raises(ValueError,match='sequence'):
        compare_repeats(rows[::-1])
    rows=records(); rows[0]['candidate_logits'][0][0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        compare_repeats(rows)


def test_trace_hooks_surround_only_third_encode(tmp_path):
    import torch
    from types import SimpleNamespace
    from multimodal_jev.span_repeatability import run_repeat_probe
    calls=[]
    def rpc(name,args=()):
        calls.append(name)
        if name=='finish_jev_span_trace_ranked':
            receipts=[]
            for rank in range(4):
                path=tmp_path/f'rank-{rank}.pt';path.write_bytes(b'trace')
                receipts.append(dict(rank=rank,path=str(path)))
            return receipts
    def encode():
        calls.append('encode');return torch.ones(3,2)
    engine=SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    result=run_repeat_probe(engine,encode,tmp_path)
    assert result['comparison']['all_logits_exact']
    assert calls==['encode','encode','begin_jev_span_trace','encode',
                   'finish_jev_span_trace_ranked','encode','encode']
