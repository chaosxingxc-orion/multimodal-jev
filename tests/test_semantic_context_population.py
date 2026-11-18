from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from project_semantic_context_population import audio_tokens, fixed_lengths, summarize


def test_chunk_and_hop_boundaries_and_scope():
    # Full100-frame chunk gives13; the next frame starts a new token.
    assert [audio_tokens(n) for n in (15999,16000,16001,41979,373760)] == [13,13,14,34,304]
    for n in (7999,480001,16000.0):
        with pytest.raises(ValueError):audio_tokens(n)


def test_projection_counts_each_recording_and_panel_without_g_multiplier():
    audio=[dict(id=str(i),samples=n,sample_rate=16000) for i,n in enumerate([16000,16000,16001])]
    r=summarize(audio,{'a':8179,'b':10})
    assert r['requests']==6
    assert r['overflow']['8192']==1  # Equal8192 fits; only8193 overflows.
    assert r['total_input_tokens']==8192*2+8193+23*2+24
    assert r['dense_attention_pair_proxy']==8192**2*2+8193**2+23**2*2+24**2
    with pytest.raises(ValueError):summarize(audio+audio[:1],{'a':10})


def test_anchor_disagreement_not_silently_extrapolated():
    a=dict(panel_id='a',arm='Direct',audio_id='x',audio_samples=16000,audio_tokens=13,tokens=113,positions=[[100]],template_sha256='t')
    b=dict(a,audio_id='y',audio_samples=16001,audio_tokens=14,tokens=114,positions=[[101]])
    assert fixed_lengths([a,b])=={('a','Direct'):100}
    for changed in (dict(b,tokens=115),dict(b,positions=[[102]]),dict(b,template_sha256='other')):
        with pytest.raises(ValueError):fixed_lengths([a,changed])
