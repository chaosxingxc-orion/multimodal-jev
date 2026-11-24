from types import SimpleNamespace

import pytest
import torch
from torch import nn

from multimodal_jev.card_acoustic_reuse import score


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_calls = 0
        self.embedding = nn.Embedding(20, 1, dtype=torch.bfloat16)

    def get_input_embeddings(self):
        return self.embedding

    def get_audio_features(self, features, mask, *, return_dict):
        self.audio_calls += 1
        return SimpleNamespace(pooler_output=features.flatten()[:1, None])

    def get_placeholder_mask(self, ids, embeds, acoustic):
        return (ids == 8).unsqueeze(-1).expand_as(embeds)

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))


class Head(nn.Module):
    def forward(self, hidden, positions, mask, *, return_logits):
        return hidden.squeeze(-1).gather(1, positions).float()


@pytest.mark.parametrize('by_length', [True, False])
def test_card_scores_preserve_acoustic_owner_and_original_order(by_length):
    from multimodal_jev.context_audio_collate import collate_context_audio
    model = nn.Module()
    model.backbone = Backbone()
    model.decision_head = Head()
    model.eval()
    sounds = {u:(torch.full((1,1,100), float(i)), (torch.arange(100) < 8)[None,:].long()) for i,u in enumerate('abc',1)}
    cfg = dict(decision_token_id=9, pad_token_id=0, audio_token_id=8,
               audio_chunk_size=100, audio_infer_window=100)
    rows = []
    for uid, idx, prefix in [('b',0,[2,4]), ('a',1,[3]), ('b',1,[3]), ('c',0,[2,4])]:
        ids = torch.tensor([[*prefix,8,9]])
        f,m = sounds[uid]
        rows.append(dict(request_id=f'{uid}:{idx}',policy_id='p', positions=torch.tensor([[ids.numel()-1]]),
                         inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     input_features=f,input_features_mask=m)))
    reference = []
    for row in rows:
        singleton,pos = collate_context_audio([row], policy_id='p', **cfg)
        ids = singleton['input_ids']
        single_emb = model.backbone.embedding(ids).masked_scatter((ids==8).unsqueeze(-1),
                              row['inputs']['input_features'].flatten()[:1].to(torch.bfloat16))
        reference.append(single_emb.cumsum(1)[0,pos.item()].float())
    assert torch.equal(score(model, rows, 'cpu', 'p', cfg, split_lengths=by_length), torch.stack(reference))
    assert model.backbone.audio_calls == len(sounds)
    rows[2]['inputs']['input_features'] = sounds['a'][0]
    with pytest.raises(ValueError, match='share the same recording'):
        score(model, rows, 'cpu', 'p', cfg, split_lengths=by_length)
