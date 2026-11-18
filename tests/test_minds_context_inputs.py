from copy import deepcopy
import json
from pathlib import Path

import pytest

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.slurp_training import content_hash
from tools.reports.prepare_minds_context_inputs import definitions_for, reconstruct


def protocol():
    return json.loads(Path('configs/tasks/minds14/native-descriptor-transfer-v1.json').read_text())


def test_audio_identity_does_not_change_the_questions_or_rendering():
    definitions = definitions_for(protocol())
    a = compile_candidates('/audio/aaa.wav', **definitions)
    b = compile_candidates('/audio/bbb.wav', **definitions)
    assert render_context_request(a) == render_context_request(b)
    assert content_hash(a.model_input()) != content_hash(b.model_input())
    assert len(a.questions) == 14
    assert a.history == ()


@pytest.mark.parametrize('change', ['prose', 'duplicate', 'dispatch', 'missing'])
def test_nonfrozen_descriptors_rejected(change):
    p = deepcopy(protocol())
    if change == 'prose':p['candidates'][0]['definition'] = 'invented narrow definition'
    if change == 'duplicate':p['candidates'][1] = p['candidates'][0]
    if change == 'dispatch':p['evaluation_dispatch_allowed'] = True
    if change == 'missing':p['candidates'].pop()
    with pytest.raises(ValueError):definitions_for(p)


def test_request_binding_and_gold_field_rejected():
    definitions = definitions_for(protocol())
    request = compile_candidates('/audio/abc.wav', **definitions)
    system, text = render_context_request(request)
    row = dict(id='minds14:abc', audio=request.audio, audio_sha256='a',
               request_sha256=content_hash(request.model_input()),
               rendered_sha256=content_hash(dict(system=system, text=text)))
    assert reconstruct(row, definitions) == request
    with pytest.raises(ValueError):reconstruct(dict(row, gold='freeze'), definitions)
    with pytest.raises(ValueError):reconstruct(dict(row, rendered_sha256='changed'), definitions)
