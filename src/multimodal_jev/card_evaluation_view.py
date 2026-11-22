"""Substitute frozen knowledge-card text around verified, unchanged audio tokens."""
import json
from pathlib import Path

from .baseline import file_hash
from .stage1_card_layout import from_panel
from .stage1_layout_cache import split_text


def wrap(spec, processor, candidates, encode):
    import torch
    from .audio_policy import load_audio
    from .context_decision_contract import encode_context_request

    path = Path(spec['path'])
    if file_hash(path) != spec['sha256']:
        raise ValueError('evaluation card specification changed')
    data = json.loads(path.read_bytes())
    sample = data['witness']
    if file_hash(sample['path']) != sample['sha256']:
        raise ValueError('tokenization witness changed')
    audio = load_audio(sample['path'], None)
    encoding = data['encoding']
    audio_id, marker = encoding['audio_token_id'], encoding['decision_token_id']
    templates = {}
    for name, panel in data['panels'].items():
        request = from_panel(panel)
        if [q.id for q in request.questions] != candidates:
            raise ValueError('card candidate order differs')
        batch, _ = encode_context_request(processor, request, audio, **encoding)
        before, after, _ = split_text(batch['input_ids'][0], audio_id)
        templates[name] = (torch.tensor(before), torch.tensor(after))
    original, readouts = templates['native']
    def prepare(asset):
        row, receipt = encode(asset)
        inputs = row['inputs']
        ids = inputs['input_ids'][0]
        end = len(ids) - len(readouts)
        if (not torch.equal(ids[:len(original)], original) or not torch.equal(ids[end:], readouts)
                or end <= len(original) or not torch.all(ids[len(original):end] == audio_id)):
            raise ValueError('input does not match verified native template')
        arm = data['assignments'][asset['id']]
        prefix, suffix = templates[arm]
        new_ids = torch.cat((prefix, ids[len(original):end], suffix))
        pos = (new_ids == marker).nonzero().T.contiguous()
        if (new_ids.numel() > encoding['max_tokens'] or pos.shape != row['positions'].shape
                or int(pos.min()) < len(prefix) + end - len(original)):
            raise ValueError('card token budget or marker layout differs')
        return dict(inputs=dict(inputs,input_ids=new_ids[None,:],attention_mask=torch.ones_like(new_ids[None,:])),
                    positions=pos), receipt
    return prepare
