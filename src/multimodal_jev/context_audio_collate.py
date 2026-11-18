"""Batch complete audio/context requests for the shared probability branch.

Reuse the existing audio padding/window checks; its token candidate slots are
internal collation metadata only. They never reach the contextual head or model.
Equal decision count is required, but candidate identities may differ by query.
"""
import torch

from .span_audio_collate import collate_span_audio


def collate_context_audio(items, *, decision_token_id, policy_id, **audio_config):
    if (not items or type(decision_token_id) is not int or decision_token_id < 0 or
            not isinstance(policy_id,str) or not policy_id or
            decision_token_id in (audio_config.get('audio_token_id'),audio_config.get('pad_token_id'))):
        raise ValueError('complete requests and distinct registered marker required')
    count = None
    seen = set()
    adapted = []
    for item in items:
        if (set(item) != {'request_id','policy_id','inputs','positions'} or
                item['policy_id'] != policy_id or not isinstance(item['request_id'],str) or
                not item['request_id'] or item['request_id'] in seen):
            raise ValueError('distinct label-free current-policy requests required')
        seen.add(item['request_id'])
        positions = item['positions']
        ids = item['inputs']['input_ids']
        if (positions.ndim != 2 or positions.shape[0] != 1 or positions.shape[1] < 1 or
                positions.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1):
            raise ValueError('one complete request with decision markers required')
        if count is None:
            count = positions.shape[1]
        if positions.shape[1] != count:
            raise ValueError('group equal decision counts before collation')
        actual = (ids[0] == decision_token_id).nonzero().flatten()
        if not torch.equal(actual,positions[0]):
            raise ValueError('marker positions must cover all markers exactly')
        adapted.append(dict(inputs=item['inputs'],positions=positions,
                            candidates=[[0,1] for _ in range(count)]))
    inputs, positions, _ = collate_span_audio(adapted,**audio_config)
    return inputs,positions


def forward_context_audio_batch(model, items, *, device, dtype, decision_token_id, policy_id, **audio_config):
    inputs,positions = collate_context_audio(items,decision_token_id=decision_token_id,
                                            policy_id=policy_id,**audio_config)
    inputs = {k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype)
              for k,v in inputs.items()}
    # One backbone invocation; sigmoid is applied by consumer only for outputs.
    # Enter Module.__call__: DDP/FSDP must see the forward to prepare reducers
    # and gather shards. Calling a delegated score_logits method bypasses them.
    return model(inputs,positions.to(device),return_logits=True)
