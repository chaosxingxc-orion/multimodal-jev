from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.omni_probability_batch import build_components, make_probability_batch
from multimodal_jev.sfc_reward_batch import digest


class Tokenizer:
    pad_token_id = 0

    def __len__(self):
        return 100

    def decode(self, ids, skip_special_tokens):
        return ' '.join(str(i) for i in ids if i not in (0, 99))


def fixture():
    requests = [dict(id=name, split='train_id', input=dict(
        decisions=[dict(id='calls')], context=dict(audios=['a.wav', 'b.wav'] if i == 0 else ['c.wav'])))
        for i, name in enumerate(('two-audio', 'one-audio'))]
    prepared = {}
    records = []
    tokenizer = Tokenizer()
    for i, row in enumerate(requests):
        ids = [10, 11, 12] if i == 0 else [13]
        width, count = (5, 2) if i == 0 else (7, 1)
        item = dict(input_ids=torch.tensor(ids), request_sha256=digest(row),
            prompt_sha256=digest(row['id']), input_ids_sha256=digest(ids), waveform_boundary='vllm_hop',
            input_features=torch.arange(count*2*width).reshape(count, 2, width).float(),
            feature_attention_mask=torch.ones(count, width, dtype=torch.long))
        prepared[row['id'], 'calls'] = item
        for rep in (0, 1):
            tokens = [20+i, 99] if rep == 0 else [22+i, 24+i, 25+i]
            records.append(dict(example_id=row['id'], decision_id='calls', replicate=rep,
                policy_sha256='a'*64, instruction_sha256='b'*64,
                request_sha256=item['request_sha256'], prompt_sha256=item['prompt_sha256'],
                input_ids_sha256=item['input_ids_sha256'], token_ids=tokens,
                log_probs=[-float(j+1) for j in range(len(tokens))],
                raw_text=tokenizer.decode(tokens, skip_special_tokens=True),
                finish_reason='stop' if rep == 0 else 'length'))
    # Exercise mixed audio counts and a completion order unrelated to replicate.
    records = [records[i] for i in (3, 0, 2, 1)]
    return dict(processor=SimpleNamespace(tokenizer=tokenizer), requests=requests,
        prepared=prepared, records=records, policy_sha256='a'*64, instruction_sha256='b'*64)


def test_label_free_full_roster_preserves_audio_response_eos_and_old_probabilities():
    args = fixture()
    before = deepcopy(args)
    parts = build_components(**args)
    t = parts['tensors']
    assert t['input_ids'].tolist() == [
        [0, 0, 13, 23, 25, 26], [10, 11, 12, 20, 99, 0],
        [0, 0, 13, 21, 99, 0], [10, 11, 12, 22, 24, 25]]
    assert t['attention_mask'].tolist() == [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0],
                                               [0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
    assert t['position_ids'].tolist()[2] == [1, 1, 0, 1, 2, 1]
    assert t['old_log_probs'].tolist() == [[-1, -2, -3], [-1, -2, 0], [-1, -2, 0], [-1, -2, -3]]
    assert 'token_level_rewards' not in t
    assert parts['meta_info']['diagnostic_only'] and not parts['meta_info']['synthetic_old_log_probs']
    assert args['records'] == before['records']
    for i, record in enumerate(args['records']):
        audio = parts['non_tensors']['multi_modal_inputs'][i]
        original = args['prepared'][record['example_id'], record['decision_id']]
        width = original['input_features'].shape[-1]
        assert torch.equal(audio['input_features'][..., :width], original['input_features'])
        assert torch.count_nonzero(audio['input_features'][..., width:]) == 0
        assert torch.equal(audio['feature_attention_mask'][..., :width], original['feature_attention_mask'])
        assert audio['input_features'].shape[-1] == 7
    assert parts['non_tensors']['record_identity'].tolist() == [
        (r['example_id'], r['decision_id'], r['replicate']) for r in args['records']]


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'policy', 'instruction', 'request',
    'prompt', 'input_ids', 'raw_boundary', 'text', 'nan', 'positive', 'bool_token', 'bool_rep',
    'length', 'reason', 'context', 'test_split', 'audio_count', 'audio_nan', 'mask_hole', 'fp32_overflow'])
def test_rejects_changed_identity_or_incomplete_input(kind):
    args = fixture();r = args['records'][0];item = args['prepared']['one-audio', 'calls']
    if kind == 'missing': args['records'].pop()
    if kind == 'duplicate': args['records'][-1] = deepcopy(r)
    if kind in ('policy', 'instruction', 'request', 'prompt'): r[kind+'_sha256'] = 'c'*64
    if kind == 'input_ids': item['input_ids'][0] = 14
    if kind == 'raw_boundary': item['waveform_boundary'] = 'raw'
    if kind == 'text': r['raw_text'] = 'repaired'
    if kind == 'nan': r['log_probs'][0] = float('nan')
    if kind == 'positive': r['log_probs'][0] = .01
    if kind == 'bool_token': r['token_ids'][0] = True
    if kind == 'bool_rep': r['replicate'] = True
    if kind == 'length': r['log_probs'].pop()
    if kind == 'reason': r['finish_reason'] = 'abort'
    if kind == 'context': args['context_limit'] = 5
    if kind == 'test_split': args['requests'][0]['split'] = 'test_ood'
    if kind == 'audio_count': item['input_features'] = torch.zeros(2, 2, 7)
    if kind == 'audio_nan': item['input_features'][0, 0, 0] = float('nan')
    if kind == 'mask_hole': item['feature_attention_mask'][0, 3] = 0
    if kind == 'fp32_overflow': r['log_probs'][0] = -1e100
    with pytest.raises(ValueError): build_components(**args)


def test_real_dataproto_preserves_identity_through_reordering():
    pytest.importorskip('verl')
    batch = make_probability_batch(**fixture())
    selected = batch[[1, 0]]
    assert selected.batch['responses'].tolist() == [[20, 99, 0], [23, 25, 26]]
    assert selected.non_tensor_batch['record_identity'].tolist() == [
        ('two-audio', 'calls', 0), ('one-audio', 'calls', 1)]
    assert [x['input_features'].shape[0] for x in selected.non_tensor_batch['multi_modal_inputs']] == [2, 1]
