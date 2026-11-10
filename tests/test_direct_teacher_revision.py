import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('revision',
    Path(__file__).parents[1]/'tools/jev/prepare_direct_teacher_revision.py')
revision=importlib.util.module_from_spec(spec);spec.loader.exec_module(revision)


def context():
    views=['native','permutation','rebind','missing_candidate','noul.True','noul.False']
    return {'schema':'jev-direct-audio-ace-feedback-v1','split':'train',
        'selection_or_test_feedback':False,'weight_updates':0,'actor_models':{'audio_model':'pin'},
        'all_group_summary':[{'id':'group:'+v} for v in views],
        'representatives':[{'evidence_id':group+':'+v,'schema':{'question':v},'gold_offline_only':True}
                           for group in ('first','second') for v in views],
        'paired_diagnostic':{'by_view':{v:{'actual_values':{},'zeros_values':{},'actual_indices':{'0':64}}
                                       for v in views}}}


def test_bounded_revision_preserves_all_summary_and_each_view_without_mutating_parent():
    source=context();original=copy.deepcopy(source);derived=revision.derive(source)
    assert source==original
    assert derived['all_group_summary']==source['all_group_summary']
    assert derived['representatives']==source['representatives'][:6]
    assert all(c=={'actual_indices':{'0':64}} for c in derived['paired_diagnostic']['by_view'].values())


@pytest.mark.parametrize('change',[{'split':'test'},{'selection_or_test_feedback':True},
    {'weight_updates':1},{'actor_models':{'audio_model':'a','text_model':'b'}}])
def test_reject_heldout_trained_or_cascade_feedback(change):
    with pytest.raises(ValueError):revision.derive({**context(),**change})
