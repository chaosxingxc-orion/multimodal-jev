"""Compare four inference microbatches with the actor's actual pre-update forwards."""
import torch


def observe(adapter, wrapped, batch, policy, collation, device, passes, micro, tolerance):
    payloads = batch.non_tensor_batch['multi_modal_inputs']
    if passes < 1 or passes*micro > len(payloads):
        raise ValueError('insufficient finite microbatches')
    forward = adapter.forward_context_audio_batch
    references = {}
    with torch.no_grad():
        for start in range(0, passes*micro, micro):
            items = [{k:p[k] for k in ('request_id','policy_id','inputs','positions')}
                     for p in payloads[start:start+micro]]
            logits = forward(wrapped,items,device=device,dtype=torch.bfloat16,policy_id=policy,**collation)
            for item, z in zip(items,logits.detach().float().cpu(),strict=True):
                references[item['request_id']] = z.sigmoid()
    evidence = dict(inference_passes=passes, compared=0, max_abs=0.)
    def compare(model, items, **kwargs):
        logits = forward(model,items,**kwargs)
        for item, z in zip(items,logits.detach().float().cpu(),strict=True):
            expected = references.pop(item['request_id'],None)
            if expected is not None:
                error = (expected-z.sigmoid()).abs().max().item()
                if not 0 <= error <= tolerance:
                    raise ValueError(f'inference and actor probabilities differ by {error}')
                evidence['compared'] += 1
                evidence['max_abs'] = max(error,evidence['max_abs'])
        if not references:
            adapter.forward_context_audio_batch = forward
        return logits
    adapter.forward_context_audio_batch = compare
    return evidence
