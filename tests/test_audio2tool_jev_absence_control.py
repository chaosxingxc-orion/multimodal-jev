from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from audit_audio2tool_jev_absence_control import counts


def test_repeated_missing_call_is_one_required_tool_not_an_absent_tool():
    target=dict(primary={'name':'a'},additional_annotations=[dict(name='a',argument_evidence='unavailable'),
        dict(name='b',argument_evidence='explicit_literal_call')])
    assert counts(target,{'a','b','c'})==dict(required_tools=2,absent_tools=1,
        unavailable_call_lists=1,available_required_call_lists=1)


def test_unknown_registry_tool_rejected():
    with pytest.raises(ValueError):counts(dict(primary={'name':'x'},additional_annotations=[]),{'a','b'})
