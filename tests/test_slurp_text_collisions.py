from pathlib import Path
import runpy

m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slurp_text_collisions.py'))


def test_normalization_preserves_negation_and_distinguishes_punctuation_modes():
    norm=m['text_key']
    assert norm('  Ａ  query ', 'source_normalized')=='a query'
    assert norm("don't stop",'source_normalized')!=norm('do stop','source_normalized')
    assert norm('hello!', 'source_normalized')!=norm('hello','source_normalized')
    assert norm('hello!', 'punctuation_relaxed')==norm('hello','punctuation_relaxed')


def test_disagreement_and_cross_role_are_independent_checks():
    groups={'same':[dict(source_id='a',role='learning',intent='i',types=['t']),
                    dict(source_id='b',role='learning',intent='j',types=[]),
                    dict(source_id='c',role='selection')]}
    bindings={'a':[dict(id='r1',source_group='g')],'b':[dict(id='r2',source_group='g')]}
    result=m['summarize'](groups,bindings,True)
    assert result['cross_role_collision_groups']==1
    assert result['learning_intent_disagreement_groups']==1
    assert result['learning_type_set_disagreement_groups']==1
    assert result['learning_intent_disagreement_recordings']==2
    assert 'c' not in result['intent_disagreements'][0]['learning_source_ids']


def test_selection_only_group_does_not_expose_content_or_labels():
    groups={'private selection text':[dict(source_id='a',role='selection'),dict(source_id='b',role='selection')]}
    r=m['summarize'](groups,{},True)
    assert r['learning_collision_groups']==r['cross_role_collision_groups']==0
    assert not r['intent_disagreements'] and not r['type_set_disagreements']
