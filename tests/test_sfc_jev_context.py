import sys
from pathlib import Path
import pytest
from test_sfc_jev import load_tool
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def test_context_audio_segments_frame_coverage_and_exact_budget_boundary():
    module=load_tool('audit_sfc_jev_context')
    ids=[1,9,9,2,9,9,9,3]
    result=module.check_coverage({'audio_sha256':['a','b']},ids,[101,200],[16001,32001],160,9)
    assert result['audio_token_lengths']==[2,3] and not result['context_overflow']
    assert module.check_coverage({'audio_sha256':['a']},[9]+[1]*3071,[3001],[480160],160,9)['context_overflow'] is False
    assert module.check_coverage({'audio_sha256':['a']},[9]+[1]*3072,[3001],[480160],160,9)['context_overflow'] is True


def test_long_audio_is_checked_without_thirty_second_exclusion():
    module=load_tool('audit_sfc_jev_context')
    result=module.check_coverage({'audio_sha256':['a']},[1,9,9,2],[3965],[634412],160,9)
    assert result['observed_frames']==[3965]
    with pytest.raises(ValueError,match='coverage'):
        module.check_coverage({'audio_sha256':['a']},[1,9,9,2],[3000],[634412],160,9)


def test_missing_or_merged_audio_segments_rejected():
    module=load_tool('audit_sfc_jev_context')
    for ids in ([1,2],[1,9,9,9,2]):
        with pytest.raises(ValueError,match='count'):
            module.check_coverage({'audio_sha256':['a','b']},ids,[10,20],[1600,3200],160,9)
