"""Evaluation-only Qwen3-ASR card scoring with group-local acoustic reuse."""
from collections import defaultdict

import torch

from .context_audio_collate import collate_context_audio


@torch.inference_mode()
def score(model, group, device, policy, collate, *, split_lengths):
    if model.training or not group:
        raise ValueError("acoustic reuse requires nonempty inference inputs")
    # Validate the original unpadded rows and spans exactly as the reference does.
    packed, positions = collate_context_audio(group, policy_id=policy, **collate)
    owners = {}
    sounds = []
    buckets = defaultdict(list)
    for i, row in enumerate(group):
        uid, _ = row["request_id"].rsplit(":", 1)
        inp = row["inputs"]
        features, mask = inp["input_features"], inp["input_features_mask"]
        if uid not in owners:
            out = model.backbone.get_audio_features(
                features.to(device=device, dtype=torch.bfloat16), mask.to(device), return_dict=True)
            owners[uid] = (features, mask, out.pooler_output)
        original, original_mask, audio = owners[uid]
        if features is not original or mask is not original_mask:
            raise ValueError("card siblings must share the same recording tensors")
        if audio.shape[0] != int((inp["input_ids"] == collate["audio_token_id"]).sum()):
            raise ValueError("per-card acoustic token count changed")
        sounds.append(audio)
        width = inp["input_ids"].shape[-1] if split_lengths else packed["input_ids"].shape[-1]
        buckets[width].append(i)

    results = [None] * len(group)
    for width, indices in buckets.items():
        ids = packed["input_ids"][indices, :width].to(device)
        mask = packed["attention_mask"][indices, :width].to(device)
        pos = packed["position_ids"][indices, :width].to(device)
        emb = model.backbone.get_input_embeddings()(ids)
        audio = torch.cat([sounds[i] for i in indices])
        placeholders = model.backbone.get_placeholder_mask(ids, emb, audio)
        emb = emb.masked_scatter(placeholders, audio)
        # Enter the original backbone through __call__: packed-attention hooks
        # still see the full valid mask; no cached hidden states or KV reuse.
        hidden = model.backbone(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                                use_cache=False, return_dict=True).last_hidden_state
        logits = model.decision_head(hidden, positions[indices].to(device), mask.bool(), return_logits=True)
        if logits.shape != (len(indices), 1) or not torch.isfinite(logits).all():
            raise ValueError("invalid card logits")
        for i, value in zip(indices, logits.unbind(), strict=True):
            results[i] = value
    return torch.stack(results)
