from pathlib import Path
import json
import runpy
import pytest

root=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(root/'tools/jev/audit_definition_interventions.py'))
cfg=json.loads((root/'configs/tasks/definition-interventions-candidate-v1.json').read_text())


def test_unknown_and_conflict_never_become_complement_negatives():
    for status in ('unavailable','conflict','unassessed'):
        assert set(m['transform'](status,cfg).values())=={None}
    with pytest.raises(ValueError):m['transform']('omitted',cfg)


def test_full_state_release_keeps_unassessed_mask_and_denominator(tmp_path, monkeypatch):
    import hashlib
    states = ('positive', 'negative', 'unavailable', 'conflict', 'unassessed')
    decisions = [dict(source_id=str(i), entity_type='role', status=state,
                      proposed_semantic_value=True if state == 'positive' else
                      False if state == 'negative' else None,
                      definition_sha256='fixture-definition', policy_sha256='fixture-policy')
                 for i, state in enumerate(states)]
    # Two recordings per source; unassessed must stay in both denominators.
    bindings = [dict(status=state) for state in states for _ in range(2)]
    files = {}
    for name, rows in [('decisions.jsonl', decisions), ('recording-bindings.jsonl', bindings)]:
        path = tmp_path / name
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = tmp_path / 'report.json'
    report.write_text(json.dumps(dict(pins={}, files=files)))
    output = tmp_path / 'audit.json'
    monkeypatch.setattr('sys.argv', ['audit', '--config', str(root / 'configs/tasks/definition-interventions-candidate-v1.json'),
        '--review-release', str(tmp_path), '--review-report-sha256', hashlib.sha256(report.read_bytes()).hexdigest(),
        '--output', str(output)])
    m['main']()
    result = json.loads(output.read_text())
    assert result['reference_decisions'] == 5
    assert (result['known_reference_pairs'], result['masked_reference_pairs']) == (2, 3)
    assert (result['known_recording_pair_bindings'], result['masked_recording_pair_bindings']) == (4, 6)
    missing = result['fixtures'][-1]
    assert missing['source_status'] == 'unassessed'
    assert set(missing['reference_only_targets'].values()) == {None}
    assert missing['optimization_allowed'] is False
    assert missing['model_scoring_allowed'] is False


def test_known_complement_and_paraphrase_truth_tables():
    for status,y in [('positive',True),('negative',False)]:
        t=m['transform'](status,cfg)
        assert t['affirmative'] is y and t['paraphrase'] is y
        assert t['complement'] is not y
        assert m['pair_scores'](y,y,not y)['both_correct']


def test_template_shortcut_is_detected_by_both_correct():
    results=[m['pair_scores'](y,True,False) for y in (False,True)]
    assert all(r['valid_opposite'] for r in results)
    assert sum(r['both_correct'] for r in results)==1
    wrong=[m['pair_scores'](y,not y,y) for y in (False,True)]
    assert all(r['opposite_but_both_wrong'] for r in wrong)


def test_invalid_outputs_never_count_as_boolean_answers():
    for a,b in [(1,0),('true','false'),(True,None),(None,False)]:
        r=m['pair_scores'](True,a,b)
        assert r['invalid_pair'] and not r['both_correct'] and not r['valid_opposite']
    with pytest.raises(ValueError):m['pair_scores'](None,True,False)


def test_negated_existential_is_not_existential_negation():
    op=cfg['operators']['B_bound']
    prompt=cfg['templates']['complement'].format(**op)
    assert 'proposition false?' in prompt and 'There exists' in prompt
    # Two instances: one bound and one unbound. NOT exists(bound)=False,
    # whereas exists(NOT bound)=True; those questions are not equivalent.
    instances=[True,False]
    assert (not any(instances)) is False
    assert any(not x for x in instances) is True
