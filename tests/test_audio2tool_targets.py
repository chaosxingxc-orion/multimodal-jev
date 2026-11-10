import sys
from types import SimpleNamespace
import pytest
from multimodal_jev.audio2tool_targets import project,literal_call

OFFICIAL=SimpleNamespace(normalize_tool_name=lambda s:s.split('(')[0].strip().lower())


def row(extras):
    return dict(id=1,query_idx=9,tool_name='a',extracted_params={'x':1},
                expected_tool_call='a(x=2)',additional_tool_calls=extras)


def test_missing_arguments_are_not_empty_arguments():
    target=project(row(['b']),3,OFFICIAL,{'a','b'})
    assert target['full_calls'] is None
    assert target['additional_annotations'][0]['arguments'] is None
    assert target['primary_encoding_consistent'] is False
    assert target['primary']['arguments']=={'x':1}


def test_explicit_empty_call_is_complete_and_duplicates_retained():
    target=project(row(['b()','b()']),3,OFFICIAL,{'a','b'})
    assert len(target['full_calls'])==3
    assert target['full_calls'][1]['arguments']=={}
    assert target['official_expected_tools']==['a','b()','b()']


@pytest.mark.parametrize('value',['b(x=1,x=2)','b(**{})','b(f())','b(x=float("nan"))','b(x={1:2})'])
def test_unsupported_literal_stays_unavailable(value):
    assert literal_call(value) is None


def test_extra_unknown_tool_rejected():
    with pytest.raises(ValueError):project(row(['unknown']),3,OFFICIAL,{'a','b'})
