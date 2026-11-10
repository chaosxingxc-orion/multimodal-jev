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
                 max_logprob_error=0.02,max_grad_norm=1.0,reward_mode='grpo',pair_bonus_weight=0.2,
                 micro_batch_size=None,reference_model=None,kl_coefficient=0.,kl_estimator='k3',progress=None,
                 response_logits_only=False,norm_adv_by_std=True,training_reservation=None):
    import torch
    from verl.utils.model import extract_multi_modal_inputs
    from .audio_policy import response_log_probs
    from .reference_policy import reference_loss
    if type(response_logits_only) is not bool:raise ValueError('response_logits_only must be boolean')
    if type(kl_coefficient) not in (int,float) or not math.isfinite(kl_coefficient) or kl_coefficient<0:
        raise ValueError('finite nonnegative KL coefficient required')
    if (reference_model is None)!=(kl_coefficient==0):
        raise ValueError('positive KL coefficient and frozen reference must be supplied together')
    if reference_model is not None:
        if reference_model is model or any(p.requires_grad for p in reference_model.parameters()):
            raise ValueError('separate frozen reference required')
        if reference_model.device!=model.device or reference_model.dtype!=model.dtype:
            raise ValueError('reference device/dtype must match actor')
        if kl_estimator not in {'k2','k3'}:raise ValueError('unsupported reference estimator')
        reference_model.eval()
    by_id=validate_rollouts(examples,records,policy_id,group_size)
    for record in records:
        decoded=processor.tokenizer.decode(record.token_ids,skip_special_tokens=True)
        if decoded!=record.text:raise ValueError('reward text differs from trained response tokens')
    if not math.isfinite(max_logprob_error) or max_logprob_error<0:
        raise ValueError('finite nonnegative parity tolerance required')
    if len(audios)!=len(examples):raise ValueError('audio/example alignment mismatch')
    if micro_batch_size is None:micro_batch_size=len(records)
    if type(micro_batch_size) is not int or micro_batch_size<1:
        raise ValueError('micro_batch_size must be a positive integer')
    if not math.isfinite(max_grad_norm) or max_grad_norm<=0:
        raise ValueError('finite positive gradient clipping threshold required')
    waves=dict(zip((ex.example_id for ex in examples),audios))
    ordered_examples=[by_id[r.example_id] for r in records]
    rewards,reward_details=compose_rewards(ordered_examples,records,reward_mode,pair_bonus_weight)
    batch=make_batch(processor,ordered_examples,[waves[r.example_id] for r in records],
                     [r.token_ids for r in records],[r.log_probs for r in records],rewards,group_size)
    batch=compute_advantages(batch,norm_adv_by_std=norm_adv_by_std)
    def report(phase,**values):
        if progress is not None:progress({'phase':phase,**values})
    report('batch_ready',responses=len(records),prompt_tokens=int(batch.batch['prompts'].shape[-1]),
           response_tokens=int(batch.batch['response_mask'].sum()))
    # GRPO grouping/normalization happens on the complete batch. Only model
    # forwards are split; prompt replicates need not fit in one microbatch.
    total_tokens=int(batch.batch['response_mask'].sum())
    from .training_budget import batch_budget
    response_lengths=batch.batch['response_mask'].sum(-1).tolist()
    prompt_lengths=(batch.batch['attention_mask'].sum(-1)-batch.batch['response_mask'].sum(-1)).tolist()
    audio_lengths=[int(mm['input_features_mask'].sum()) for mm in batch.non_tensor_batch['multi_modal_inputs']]
    def budget(gradient_passes,optimizer_steps):
        return batch_budget(prompt_lengths,response_lengths,audio_lengths,gradient_passes=gradient_passes,
                            no_grad_passes=1+int(reference_model is not None),optimizer_steps=optimizer_steps)
    if training_reservation is not None:
        from .budget_control import validate_training_reservation
        validate_training_reservation(budget(1,1),training_reservation)
    def chunks():
        for begin in range(0,len(records),micro_batch_size):
            part=batch[begin:begin+micro_batch_size]
            part.batch=part.batch.to(model.device)
            mm=extract_multi_modal_inputs(part.non_tensor_batch['multi_modal_inputs'])
            mm={key:value.to(device=model.device,dtype=model.dtype if value.is_floating_point() else value.dtype)
                for key,value in mm.items()}
            b=part.batch
            kwargs={'input_ids':b['input_ids'],'attention_mask':b['attention_mask'],
                    'position_ids':b['position_ids'],**mm,'use_cache':False}
            if response_logits_only:kwargs['logits_to_keep']=b['responses'].shape[-1]+1
            yield part,kwargs
    def log_probs(output,b):
        boundary=b['prompts'].shape[-1]
        return response_log_probs(output.logits,b['input_ids'],boundary,
                                  logits_start=boundary-1 if response_logits_only else 0)
    # Disable dropout for probability parity both before and during update.
    # eval does not disable autograd.
    model.eval()
    error=0.
    reference_parts=[]
    with torch.no_grad():
        for index,(part,kwargs) in enumerate(chunks()):
            b=part.batch
            report('parity_forward',chunk=index)
            output=model(**kwargs)
            actor_logp=log_probs(output,b)
            differences=(actor_logp-b['old_log_probs']).abs()
            part_error=differences[b['response_mask'].bool()].max().item()
            report('parity_checked',chunk=index,max_logprob_error=part_error,
                   per_response_error=[float(row[mask.bool()].max()) for row,mask in zip(differences,b['response_mask'])])
            del output,actor_logp
            if not math.isfinite(part_error) or part_error>max_logprob_error:
                raise ValueError(f'actor/rollout logprob mismatch: {part_error} > {max_logprob_error}; no optimizer update')
            error=max(error,part_error)
            if reference_model is not None:
                output=reference_model(**kwargs)
                ref_logp=log_probs(output,b)
                if not torch.isfinite(ref_logp[b['response_mask'].bool()]).all():
                    raise ValueError('nonfinite reference response log-prob; no optimizer update')
                reference_parts.append(ref_logp.detach().cpu())
                del output,ref_logp
    # Equal task rewards provide no policy-gradient signal. A fixed-reference
    # regularizer may still act on an actor that has drifted from its reference.
    if not batch.batch['advantages'].any() and reference_model is None:
        return {'status':'zero_advantage','updated':False,'rewards':rewards,'max_logprob_error':error,
                'reward_details':reward_details,'micro_batch_size':micro_batch_size,'training_budget':budget(0,0)}
    optimizer.zero_grad(set_to_none=True)
    total_loss=0.;metrics={}
    for index,(part,kwargs) in enumerate(chunks()):
        b=part.batch
        report('gradient_forward',chunk=index)
        output=model(**kwargs)
        logp=log_probs(output,b)
        loss,part_metrics=policy_loss(part,logp)
        part_metrics['actor/policy_loss']=float(loss.detach())
        if reference_model is not None:
            kl=reference_loss(logp,reference_parts[index].to(logp.device),b['response_mask'],kl_estimator)
            part_metrics['actor/reference_kl']=float(kl.detach())
            loss=loss+kl_coefficient*kl
        # verl's token-mean loss must be weighted by effective response tokens,
        # not the number of microbatches or examples (lengths can differ).
        weight=int(b['response_mask'].sum())/total_tokens
        if not torch.isfinite(loss):raise ValueError('nonfinite policy loss')
        report('backward',chunk=index,loss=float(loss.detach()))
        (loss*weight).backward()
        total_loss+=float(loss.detach())*weight
        for key,value in part_metrics.items():metrics[key]=metrics.get(key,0.)+float(value)*weight
        del output,logp,loss
    params=[param for param in model.parameters() if param.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=True)
    if norm==0:
        return {'status':'zero_gradient','updated':False,'loss':total_loss,'grad_norm':0.,
                'training_budget':budget(1,0),
                'rewards':rewards,'metrics':metrics,'max_logprob_error':error,'reward_details':reward_details,
                'micro_batch_size':micro_batch_size,'kl_coefficient':kl_coefficient,'kl_estimator':kl_estimator}
    optimizer.step()
    report('optimizer_step',grad_norm=float(norm),loss=total_loss)
    return {'status':'updated','updated':True,'loss':total_loss,'grad_norm':float(norm),
            'training_budget':budget(1,1),
            'rewards':rewards,'max_logprob_error':error,'metrics':metrics,'reward_details':reward_details,
            'policy_id':policy_id,'micro_batch_size':micro_batch_size,'response_tokens':total_tokens,
            'kl_coefficient':kl_coefficient,'kl_estimator':kl_estimator if reference_model is not None else None,
            'scope':'single audio PPO update using verl GRPO advantages; not efficacy evaluation'}
