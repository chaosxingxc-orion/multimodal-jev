from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
from audit_audio2tool_context_population import groups, samples_and_frames


def test_variants_and_dialogue_have_different_denominators():
    audios = [dict(path='a', turn=0, role='user'), dict(path='b', turn=1, role='agent')]
    assert groups(dict(tier=1, audio_semantics='independent_variants', audios=audios)) == [[audios[0]], [audios[1]]]
    assert groups(dict(tier=7, audio_semantics='ordered_conversation_turns', audios=audios)) == [audios]
    with pytest.raises(ValueError, match='ordered'):
        groups(dict(tier=7, audio_semantics='ordered_conversation_turns', audios=audios[::-1]))


@pytest.mark.parametrize('frames,samples,hops', [(239,160,1),(240,160,1),(241,161,2),(15711378,10474252,65465)])
def test_integer_resampling_and_hop_ceil(frames,samples,hops):
    assert samples_and_frames(dict(frames=frames,sample_rate=24000,channels=1,sample_width_bytes=2)) == (samples,hops)


def test_zero_length_is_not_silently_a_zero_token_request():
    with pytest.raises(ValueError):
        samples_and_frames(dict(frames=0,sample_rate=24000,channels=1,sample_width_bytes=2))
