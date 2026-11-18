"""Whitespace compaction preserves every candidate rule and output contract."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import pytest

TOOLS=Path(__file__).parents[1]/'tools/jev'
sys.path.insert(0,str(TOOLS))
from audit_harness_routing_support import policy,FAMILIES
from probe_harness_routing_context import request_for
from multimodal_jev.task_schema import Candidate,DecisionSchema


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('structured',[False,True])
def test_complete_request_equivalence_with_whitespace_inside_descriptions(family,structured):
    ontology=[f'event-{i}' for i in range(60)]
    task=dict(schema=asdict(DecisionSchema('events','choice','Which event?',
        tuple(Candidate(k,'Keep  spaces, "quotes", and Unicode 语音: '+k) for k in ontology),False)),
        thinking=dict(requirements=['Use audio.']))
    spec=policy(ontology,family,317)
    a=request_for('/audio.wav',task,spec,structured=structured)
    b=request_for('/audio.wav',task,spec,structured=structured,representation='compact_json_v1')
    assert a.questions==b.questions and a.requirements[:-1]==b.requirements[:-1]
    assert json.loads(a.requirements[-1])==json.loads(b.requirements[-1])
    assert len(b.requirements[-1])<len(a.requirements[-1])
    assert a.audio==b.audio and a.version==b.version and a.final_id==b.final_id


def test_unknown_representation_fails_before_compilation():
    with pytest.raises(ValueError,match='representation'):
        request_for('/audio.wav',{}, {}, structured=False,representation='lossy')
