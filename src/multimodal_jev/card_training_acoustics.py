"""One frozen acoustic forward per distinct recording in a local Stage1 update."""
import torch
from torch.nn import functional as F

from .context_audio_collate import collate_card_text

ENGINE = 'unique_recording_v1'


def collate(rows, policy_id, **options):
    packed, positions = collate_card_text(rows, policy_id=policy_id, **options)
    by_owner, owners, offsets, gathers = {}, [], [], []
    total = 0
    for row in rows:
        x = row['inputs']
        f, m = x['input_features'], x['input_features_mask']
        key = (id(f), id(m))
        if key not in by_owner:
            by_owner[key] = len(owners)
            owners.append((f, m))
            count = int((x['input_ids'] == options['audio_token_id']).sum())
            offsets.append(torch.arange(total,total+count))
            total += count
        gathers.append(offsets[by_owner[key]])
    width = max(f.shape[-1] for f, _ in owners)
    for i, key in enumerate(('input_features','input_features_mask')):
        packed[key] = torch.cat([F.pad(v[i], (0,width-v[i].shape[-1])) for v in owners])
    packed['audio_owner_tokens'] = torch.cat(gathers)
    return packed, positions


def hidden(backbone, inputs):
    tower, projector = backbone.audio_tower, backbone.multi_modal_projector
    if (tower.training or projector.training or
            any(p.requires_grad for m in (tower,projector) for p in m.parameters())):
        raise ValueError('acoustic reuse requires a deterministic frozen tower and projector')
    with torch.no_grad():
        unique = backbone.get_audio_features(inputs['input_features'], inputs['input_features_mask'],
                                             return_dict=True).pooler_output
        audio = unique.index_select(0,inputs['audio_owner_tokens'])
    ids=inputs['input_ids']
    embeds=backbone.get_input_embeddings()(ids)
    placeholders=backbone.get_placeholder_mask(ids,embeds,audio)
    return backbone(inputs_embeds=embeds.masked_scatter(placeholders,audio),
                    position_ids=inputs['position_ids'],attention_mask=inputs['attention_mask'],
                    use_cache=False,return_dict=True).last_hidden_state
