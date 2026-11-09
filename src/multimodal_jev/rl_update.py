"""Single on-policy audio PPO/GRPO update using verl's objective.

Consumes actual vLLM rollout records. Pre-update actor/rollout log-prob parity is
mandatory. Call with a freshly exported actor policy matching records.policy_id.
This helper does not manage distributed workers or an experiment schedule.
"""
from __future__ import annotations
from collections import Counter
import math
from .paired_rollout import compose_rewards
from .verl_audio import make_batch,compute_advantages,policy_loss


def validate_rollouts(examples,records,policy_id,group_size):
    if not policy_id or not examples:raise ValueError('policy and examples required')
    if type(group_size) is not int or group_size<2:raise ValueError('group size must be at least two')
    by_id={ex.example_id:ex for ex in examples}
    if len(by_id)!=len(examples):raise ValueError('duplicate example identity')
    seen=set();counts=Counter()
    for record in records:
        if record.policy_id!=policy_id:raise ValueError('stale rollout policy')
        if record.example_id not in by_id:raise ValueError('unknown rollout example')
        key=(record.example_id,record.replicate)
        if key in seen or type(record.replicate) is not int or not 0<=record.replicate<group_size:raise ValueError('duplicate/invalid rollout replicate')
        if not record.token_ids or len(record.token_ids)!=len(record.log_probs):raise ValueError('empty or unaligned response')
        if any(type(token) is not int or token<0 for token in record.token_ids):raise ValueError('invalid response tokens')
        if any(not math.isfinite(p) or p>1e-5 for p in record.log_probs):raise ValueError('invalid response logprobs')
        if record.finish_reason not in {'stop','length'}:raise ValueError('invalid finish reason')
        seen.add(key);counts[record.example_id]+=1
    if set(counts)!=set(by_id) or any(n!=group_size for n in counts.values()):
        raise ValueError('incomplete rollout groups')
    return by_id


def update_actor(model,processor,optimizer,examples,audios,records,policy_id,group_size,
                 max_logprob_error=0.02,max_grad_norm=1.0,reward_mode='grpo',pair_bonus_weight=0.2):
    import torch
    from verl.utils.model import extract_multi_modal_inputs
    from .audio_policy import response_log_probs
    by_id=validate_rollouts(examples,records,policy_id,group_size)
    for record in records:
        decoded=processor.tokenizer.decode(record.token_ids,skip_special_tokens=True)
        if decoded!=record.text:raise ValueError('reward text differs from trained response tokens')
    if not math.isfinite(max_logprob_error) or max_logprob_error<0:
        raise ValueError('finite nonnegative parity tolerance required')
    if len(audios)!=len(examples):raise ValueError('audio/example alignment mismatch')
    waves=dict(zip((ex.example_id for ex in examples),audios))
    ordered_examples=[by_id[r.example_id] for r in records]
    rewards,reward_details=compose_rewards(ordered_examples,records,reward_mode,pair_bonus_weight)
    batch=make_batch(processor,ordered_examples,[waves[r.example_id] for r in records],
                     [r.token_ids for r in records],[r.log_probs for r in records],rewards,group_size)
    batch=compute_advantages(batch)
    b=batch.batch.to(model.device)
    mm=extract_multi_modal_inputs(batch.non_tensor_batch['multi_modal_inputs'])
    mm={key:value.to(device=model.device,dtype=model.dtype if value.is_floating_point() else value.dtype)
        for key,value in mm.items()}
    # Disable dropout for probability parity both before and during update.
    # eval does not disable autograd.
    model.eval()
    kwargs={'input_ids':b['input_ids'],'attention_mask':b['attention_mask'],'position_ids':b['position_ids'],**mm,'use_cache':False}
    with torch.no_grad():
        output=model(**kwargs)
        actor_logp=response_log_probs(output.logits,b['input_ids'],b['prompts'].shape[-1])
        error=(actor_logp-b['old_log_probs']).abs()[b['response_mask'].bool()].max().item()
    del output,actor_logp
    if not math.isfinite(error) or error>max_logprob_error:
        raise ValueError(f'actor/rollout logprob mismatch: {error} > {max_logprob_error}; no optimizer update')
    # Equal rewards must produce a no-op, not a fabricated learning signal.
    if not b['advantages'].any():
        return {'status':'zero_advantage','updated':False,'rewards':rewards,'max_logprob_error':error,'reward_details':reward_details}
    optimizer.zero_grad(set_to_none=True)
    output=model(**kwargs)
    logp=response_log_probs(output.logits,b['input_ids'],b['prompts'].shape[-1])
    batch.batch=b
    loss,metrics=policy_loss(batch,logp)
    if not torch.isfinite(loss):raise ValueError('nonfinite policy loss')
    loss.backward()
    params=[param for param in model.parameters() if param.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    return {'status':'updated','updated':True,'loss':float(loss.detach()),'grad_norm':float(norm),
            'rewards':rewards,'max_logprob_error':error,'metrics':metrics,'reward_details':reward_details,
            'policy_id':policy_id,'scope':'single audio PPO update using verl GRPO advantages; not efficacy evaluation'}
