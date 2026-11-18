import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from prepare_fsc_context_transfer import compile_native
from multimodal_jev.context_candidate_task import bind_candidate_targets
from multimodal_jev.context_decision_contract import render_context_request,DECISION_TOKEN


def task():
    return json.loads((Path(__file__).parents[1]/'configs/tasks/fsc-span-transfer-v1.json').read_text())


def test_all_native_commands_bind_without_semantic_projection():
    t=task();r=compile_native('/audio.wav',t)
    assert len(r.questions)==31 and r.version=='jev-context-exclusive_event-v1'
    assert [q.definition for q in r.questions]==[c['description'] for c in t['commands']]
    assert render_context_request(r)[1].count(DECISION_TOKEN)==31
    for index,c in enumerate(t['commands']):
        target=bind_candidate_targets(r,semantics='exclusive_event',exclusive_event=c['id'])
        assert sum(target['labels'])==1 and target['labels'][index] and all(target['supervised'])
    with pytest.raises(ValueError):
        bind_candidate_targets(r,semantics='candidate_applicability',exclusive_event=t['commands'][0]['id'])


def test_definition_binding_follows_permutation_and_rejects_missing_support():
    t=task();original=compile_native('/audio.wav',t)
    t['commands'].reverse();permuted=compile_native('/audio.wav',t)
    assert {q.id:q.definition for q in original.questions}=={q.id:q.definition for q in permuted.questions}
    assert [q.id for q in original.questions]==list(reversed([q.id for q in permuted.questions]))
    t['commands'].pop()
    with pytest.raises(ValueError):compile_native('/audio.wav',t)
