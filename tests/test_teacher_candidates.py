import copy
import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.teacher_candidates import strict_json, validate_candidates, stage_candidates


def sample():
    context = {'parent_prompt_sha256': 'parent-hash', 'parent_instruction': 'old prompt',
               'representatives': [{'evidence_id': 'train-evidence-001'}],
               'all_group_summary': [{'id': 'train-evidence-001'}, {'id': 'train-evidence-002'}],
               'split': 'train', 'selection_or_test_feedback': False}
    refs = ['train-evidence-001']
    payload = {'schema': 'jev-teacher-prompts-v1', 'parent_prompt_sha256': 'parent-hash',
               'diagnosis': [{'mechanism': 'JSON format', 'evidence_ids': refs, 'uncertainty': 'small sample'}],
               'candidates': [{'id': 'p1', 'instruction': 'Read the definition and return JSON.',
                               'mechanism': 'format', 'evidence_ids': refs,
                               'counterevidence': 'some outputs valid', 'falsification': 'no heldout gain'}]}
    return context, payload


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"value":NaN}', 'non-JSON'])
def test_invalid_json_rejected_without_repair(text):
    with pytest.raises(ValueError): strict_json(text)


def test_diagnosis_can_cite_supplied_summary_but_candidate_requires_full_evidence():
    context, payload = sample()
    payload['diagnosis'][0]['evidence_ids'] = ['train-evidence-002']
    assert validate_candidates(payload, context)
    payload['candidates'][0]['evidence_ids'] = ['train-evidence-002']
    with pytest.raises(ValueError, match='evidence'): validate_candidates(payload, context)
    payload['candidates'][0]['evidence_ids'] = ['train-evidence-001']
    payload['diagnosis'][0]['evidence_ids'] = ['invented']
    with pytest.raises(ValueError, match='evidence'): validate_candidates(payload, context)


@pytest.mark.parametrize('change', ['parent', 'evidence', 'duplicate', 'long', 'memorize', 'extra', 'empty'])
def test_candidate_boundaries(change):
    context, payload = sample()
    assert len(validate_candidates(payload, context)) == 1
    candidate = payload['candidates'][0]
    if change == 'parent': payload['parent_prompt_sha256'] = 'wrong'
    if change == 'evidence': candidate['evidence_ids'] = ['selection-secret']
    if change == 'duplicate': payload['candidates'].append(copy.deepcopy(candidate))
    if change == 'long': candidate['instruction'] = 'x' * 4001
    if change == 'memorize': candidate['instruction'] += ' train-evidence-002 answers a'
    if change == 'extra': candidate['command'] = 'execute'
    if change == 'empty': payload['candidates'] = []
    with pytest.raises(ValueError): validate_candidates(payload, context)


def request_fixture(tmp_path, finish='stop', done=True):
    context, payload = sample(); root = tmp_path / 'request'; root.mkdir()
    response = root / 'response'; response.mkdir()
    messages = root / 'messages.private.json'
    messages.write_text(json.dumps([{'role': 'system', 'content': 'instructions'},
                                    {'role': 'user', 'content': json.dumps(context)}]))
    config = {'messages': {'path': str(messages), 'sha256': file_hash(messages)},
              'output_root': str(response)}
    (root / 'executor-config.private.json').write_text(json.dumps(config))
    content = json.dumps(payload)
    event = {'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': finish}]}
    (response / 'response.sse').write_text('data: ' + json.dumps(event) + '\n\n' + ('data: [DONE]\n\n' if done else ''))
    (response / 'response.json').write_text(json.dumps({'choices': [
        {'finish_reason': finish, 'message': {'content': content, 'reasoning_content': 'private'}}]}))
    (response / 'result.json').write_text(json.dumps({'state': 'response_received', 'http_status': 200,
        'finish_reasons': [finish], 'response_sha256': file_hash(response / 'response.sse'), 'usage': {'total_tokens': 10}}))
    return root


def test_stage_preserves_candidate_without_approval_or_private_reasoning(tmp_path):
    root = request_fixture(tmp_path); output = tmp_path / 'candidates'
    receipt = stage_candidates(root, output)
    assert receipt['semantic_review'] == 'pending' and receipt['candidate_accepted'] is False
    assert 'private' not in (output / 'proposal.json').read_text()
    assert (output / 'p1.txt').read_text() == sample()[1]['candidates'][0]['instruction']


@pytest.mark.parametrize('finish,done', [('length', True), ('stop', False)])
def test_truncated_response_is_not_imported(tmp_path, finish, done):
    root = request_fixture(tmp_path, finish, done)
    with pytest.raises(ValueError): stage_candidates(root, tmp_path / 'candidates')
    assert not (tmp_path / 'candidates').exists()


def test_request_specific_character_budget_rejects_before_staging(tmp_path):
    root = request_fixture(tmp_path)
    (root/'prepared.json').write_text(json.dumps({'max_candidates': 2, 'max_instruction_characters': 5}))
    with pytest.raises(ValueError, match='registered candidate budget'):
        stage_candidates(root, tmp_path/'candidates')
    assert not (tmp_path/'candidates').exists()
