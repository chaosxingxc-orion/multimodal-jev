from copy import deepcopy
from pathlib import Path
import runpy
import pytest

bind=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/bind_intent_definition_fit.py'))['bind']


def fixture():
    packet={'sources':[{'source_id':'1','native_intent':'other','reference':'recipe request',
        'metadata_sha256':'metadata','recordings':[{'id':'audio1','source_group':'g',
        'audio_sha256':'audio','target_sha256':'target','all_source_ids':['1','2']}]}]}
    reviews=[{'source_id':'1','semantic_id':'recipe','definition_fit':'supported',
        'rationale':'Explicit recipe request.','scope':'Reference only.'}]
    return packet,reviews,{'semantic_id':'recipe','definition':'Request a recipe.'}


def test_semantic_fit_does_not_imply_exclusive_truth_or_optimization_admission():
    decisions,bindings=bind(*fixture())
    assert decisions[0]['proposed_definition_fit'] is True
    assert decisions[0]['disagrees_with_native_membership'] is True
    assert decisions[0]['exclusive_I_class_truth'] is None
    assert decisions[0]['optimization_allowed'] is False
    assert bindings[0]['complete_source_review'] is False
    assert bindings[0]['optimization_allowed'] is False


def test_incomplete_and_duplicate_source_reviews_rejected():
    packet,reviews,definition=fixture()
    for bad in ([],reviews+deepcopy(reviews)):
        with pytest.raises(ValueError,match='exact complete'):
            bind(packet,bad,definition)


def test_unresolved_not_converted_to_negative():
    packet,reviews,definition=fixture()
    reviews[0]['definition_fit']='unresolved'
    decisions,_=bind(packet,reviews,definition)
    assert decisions[0]['proposed_definition_fit'] is None
    assert decisions[0]['disagrees_with_native_membership'] is False
