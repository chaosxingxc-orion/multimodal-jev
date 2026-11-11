"""Explicit padded Omni DataProto -> verl compute_log_prob adapter.

No optimizer, CPU monkey patch or implicit device fallback. Callers authenticate
model/runtime identity and construct batches through omni_actor_batch.make_batch.
"""
def compute_log_probs(actor, data, *, micro_batch_size, pad_token_id):
    import torch
    if (type(micro_batch_size) is not int or micro_batch_size<1
            or type(pad_token_id) is not int or pad_token_id<0):
        raise ValueError('positive micro batch size and tokenizer pad ID required')
    if (data.meta_info.get('audio_contract')!='sfc-jev-qwen3-omni-padded-v1'
            or data.meta_info.get('requires_remove_padding') is not False
            or data.meta_info.get('requires_sequence_parallel_size')!=1
            or actor.use_remove_padding or actor.use_fused_kernels or actor.use_prefix_grouper
            or actor.use_dynamic_bsz or actor.ulysses_sequence_parallel_size!=1):
        raise ValueError('padded unfused static Omni actor with sequence parallel size one required')
    tensors=data.batch;responses=tensors['responses'];mask=tensors['response_mask']
    old=tensors['old_log_probs']
    if (responses.ndim!=2 or responses.shape!=mask.shape or old.shape!=mask.shape
            or not torch.all((mask==0)|(mask==1)) or not torch.all(mask.sum(-1)>0)
            or not torch.equal(mask,mask.cumprod(-1))
            or not torch.equal(tensors['input_ids'][:,-responses.shape[1]:],responses)
            or not torch.equal(tensors['attention_mask'][:,-responses.shape[1]:],mask)
            or not torch.isfinite(old[mask.bool()]).all() or (old[mask.bool()]>0).any()
            or 'multi_modal_inputs' not in data.non_tensor_batch):
        raise ValueError('aligned right-padded responses and retained sampled probabilities required')
    # Exclude old probabilities and rewards from worker input: recomputation must
    # not replace or mutate rollout provenance. Metadata is independent as well.
    inputs=data.select(batch_keys=['responses','input_ids','attention_mask','position_ids'],
                       non_tensor_batch_keys=['multi_modal_inputs'])
    inputs.meta_info=dict(inputs.meta_info,micro_batch_size=micro_batch_size,
                          temperature=1.,use_dynamic_bsz=False,pad_token_id=pad_token_id)
    output=actor.compute_log_prob(inputs,calculate_entropy=False)
    # Direct actor calls bypass FSDPWorker.compute_log_prob's root cleanup.
    # FSDP1 retains the root full-parameter buffer after a no-grad forward;
    # reproduce the pinned worker's reshard before exporting probabilities.
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    module=actor.actor_module if hasattr(actor,'actor_module') else None
    if isinstance(module,FSDP) and torch.distributed.get_world_size(module.process_group)>1:
        if module._handle is not None:
            module._handle.reshard(True)
    logp=output['log_probs']
    if logp.shape!=mask.shape:raise ValueError('actor response shape differs')
    active=mask.to(device=logp.device,dtype=torch.bool)
    if not torch.isfinite(logp[active]).all() or (logp[active]>0).any():
        raise ValueError('actor returned invalid valid-token probabilities')
    # Padding is excluded explicitly; original valid-token values are unchanged.
    return logp.masked_fill(~active,0)
