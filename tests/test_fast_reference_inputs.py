import copy
import pytest
from tools.reports.context_fast_reference_inputs import compiler
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.slurp_training import content_hash


@pytest.mark.parametrize('audio',['/a.wav','中文/音频.wav','a"\\b.wav','emoji🎵.wav'])
def test_exact_original_requests(audio):
    fast=compiler();c=[dict(id=str(i),definition='定义 '+str(i)) for i in range(60)];r=['rule']
    fast('/first.wav',c,r,semantics='exclusive_event')
    a=fast(audio,c,r,semantics='exclusive_event').model_input()
    b=compile_candidates(audio,c,r,semantics='exclusive_event').model_input()
    assert a==b and content_hash(a)==content_hash(b)


@pytest.mark.parametrize('fault',['audio','token','reserved','definition','requirements','duplicate','semantics'])
def test_invalid_changed_inputs_still_rejected(fault):
    fast=compiler();c=[dict(id='a',definition='A')];r=['rule'];audio='/a.wav';sem='exclusive_event'
    fast(audio,c,r,semantics=sem)
    if fault=='audio':audio=' '
    if fault=='token':audio='<|jev_decision|>'
    if fault=='reserved':audio='<|jev_probe_reserved_id_151705|>'
    if fault=='definition':c[0]['definition']=''
    if fault=='requirements':r.append('')
    if fault=='duplicate':c.append(dict(c[0]))
    if fault=='semantics':sem='wrong'
    with pytest.raises(ValueError):fast(audio,c,r,semantics=sem)
