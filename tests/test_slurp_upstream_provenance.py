from pathlib import Path
import runpy

import pytest

trace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slurp_upstream_provenance.py'))['trace']


def row(answerid='900', text='send mail'):
    return dict(answerid=answerid, answer_normalised=text, answer_annotation=text,
                answer=text, question='collection prompt', scenario='email', intent='sendemail')


def source(text='send mail'):
    return dict(slurp_id=1, sentence=text, scenario='email', action='sendemail')


def test_cross_dataset_ids_never_join_and_duplicate_text_never_collapses():
    records=[row('1','unrelated'),row('900'),row('901')]
    records[-1]['intent']='query'
    result=trace([source()],records,[])[0]
    assert result['status']=='multiple_exact_candidates'
    assert [x['annotated_row']['answerid'] for x in result['candidates']]==['900','901']
    assert [x['native_pair_agreement'] for x in result['candidates']]==[True,False]
    assert not result['optimization_allowed'] and not result['actor_input_allowed']


def test_markup_only_and_strict_no_wake_word_or_case_repair():
    annotated=row(text='send mail to amy');annotated['answer_annotation']='send mail to [person : amy]'
    result=trace([source('send mail to amy')],[annotated],[])[0]
    assert result['candidates'][0]['matched_fields']==['answer_normalised','answer_annotation_stripped']
    assert trace([source('olly send mail to amy')],[annotated],[])[0]['status']=='no_exact_match'
    assert trace([source('Send mail to amy')],[annotated],[])[0]['status']=='no_exact_match'


def test_original_id_requires_content_and_preserves_multiple_candidates():
    result=trace([source()],[row()],[row(),row(text='unrelated')])[0]
    candidates=result['candidates'][0]['original_candidates']
    assert [x['linkage_status'] for x in candidates]==['id_and_content_match','id_only_unverified']


def test_duplicate_source_id_rejected():
    with pytest.raises(ValueError):trace([source(),source()],[row()],[])
