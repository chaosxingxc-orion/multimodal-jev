"""Context-head adapter to the pinned native verl update/optimizer loop.

One transport row is one original audio query, containing its complete sampled
group. The backbone is evaluated once per microbatch, not once per action.
"""
from contextvars import ContextVar
import inspect
import json
import math
import numpy as np
import torch
import torch.distributed as dist

from .baseline import file_hash
from .context_audio_collate import forward_context_audio_batch
from .context_binary_policy import binary_logprobs
from .context_hybrid_objective import loss
from .context_set_reward import set_f1
from .context_behavior_parity import compare as compare_behavior, validate_bounds
from .span_native_actor import ACTOR_SHA256
from .span_policy import joint_logprobs
from .span_group_advantage import group_advantages, CORE_SHA256
from .span_finite_credit import validate_policy

MODE='jev_context_hybrid_v1'
_active=ContextVar('jev_context_hybrid_active',default=None)


def make_batch(items, *, stage, ranking_coefficient, probability_coefficient,
               algorithm='grpo_v1', kl_coefficient=0., reference_policy_id=None):
    from verl import DataProto
    if not items or stage not in ('instruction','task') or algorithm not in ('grpo_v1','rloo_v1'):
        raise ValueError('nonempty context batch and registered stage/algorithm required')
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in
           (ranking_coefficient,probability_coefficient,kl_coefficient)):
        raise ValueError('finite nonnegative objective coefficients required')
    if ((stage=='instruction' and (kl_coefficient or reference_policy_id is not None)) or
            (kl_coefficient>0 and (not isinstance(reference_policy_id,str) or not reference_policy_id))):
        raise ValueError('task KL requires fixed reference identity')
    real_count=sum(item['real_request'] is True for item in items)
    if real_count<1:raise ValueError('nonempty real query population required')
    payload=[];seen=set();policies=set();width=None;groups=None
    for item in items:
        p={k:item[k] for k in ('request_id','policy_id','inputs','positions','labels','observed',
                              'request_weight','real_request')}
        if (not isinstance(p['request_id'],str) or not p['request_id'] or p['request_id'] in seen or
                not isinstance(p['policy_id'],str) or not p['policy_id'] or
                type(p['real_request']) is not bool or type(p['request_weight']) not in (int,float) or
                not math.isfinite(p['request_weight']) or p['request_weight']<0 or
                (not p['real_request'] and p['request_weight']!=0)):
            raise ValueError('distinct requests, current policy and zero-weight padding required')
        seen.add(p['request_id']);policies.add(p['policy_id'])
        labels,observed=p['labels'],p['observed']
        if (labels.dtype!=torch.bool or observed.dtype!=torch.bool or labels.ndim!=2 or
                labels.shape[0]!=1 or labels.shape[1]<1 or labels.shape!=observed.shape or
                labels.device.type!='cpu' or observed.device.type!='cpu' or
                p['positions'].shape!=labels.shape or p['positions'].dtype!=torch.long):
            raise ValueError('complete CPU Boolean query targets and marker positions required')
        n=labels.shape[1]
        if width is None:width=n*2
        if width!=n*2:raise ValueError('equal decision count per update required')
        if (set(p['inputs'])-{'input_ids','attention_mask','position_ids','input_features','input_features_mask'} or
                p['inputs']['input_ids'].ndim!=2 or p['inputs']['input_ids'].shape[0]!=1):
            raise ValueError('one label-free encoded audio request required')
        p.update(stage=stage,algorithm=algorithm,ranking_coefficient=ranking_coefficient,
            probability_coefficient=probability_coefficient,kl_coefficient=kl_coefficient,
            reference_policy_id=reference_policy_id,global_rows=len(items),global_real_requests=real_count)
        if stage=='instruction':
            if any(item.get(k) is not None for k in ('behavior','actions','reference_policy')):
                raise ValueError('instruction stage must not invent rollout')
            p.update(behavior=None,actions=None,reference_policy=None,old_joint=None,advantages=None)
        else:
            behavior=item['behavior'];actions=item['actions']
            validate_policy(behavior,1,detached=True)
            if (len(behavior)!=n or any(v.shape!=(1,2) or v.device.type!='cpu' or v.dtype!=torch.float32 for v in behavior) or
                    actions.ndim!=3 or actions.shape[0]!=1 or actions.shape[2]!=n or
                    actions.shape[1]<2 or actions.device.type!='cpu'):
                raise ValueError('complete detached CPU binary policy and action group required')
            if groups is None:groups=actions.shape[1]
            if groups!=actions.shape[1]:raise ValueError('same group size across queries required')
            rewards=set_f1(actions,labels,observed)
            advantages=group_advantages(rewards,[p['request_id']],algorithm=algorithm)
            reference=item.get('reference_policy')
            if kl_coefficient:
                if item.get('reference_policy_id')!=reference_policy_id:
                    raise ValueError('reference policy identity differs')
                validate_policy(reference,1,detached=True)
                if len(reference)!=n or any(v.shape!=(1,2) or v.device.type!='cpu' for v in reference):
                    raise ValueError('reference binary support differs')
            elif reference is not None:raise ValueError('unused reference must be absent')
            p.update(behavior=behavior,actions=actions,old_joint=joint_logprobs(behavior,actions),
                     advantages=advantages,rewards=rewards,reference_policy=reference)
        payload.append(p)
    if len(policies)!=1:raise ValueError('one behavior policy per update required')
    old=torch.stack([torch.cat(p['behavior'],-1)[0] if stage=='task' else torch.zeros(width) for p in payload])
    packed=np.empty(len(items),dtype=object);packed[:]=payload
    ordinals=torch.arange(width).repeat(len(items),1)
    return DataProto.from_dict(tensors=dict(responses=ordinals,response_mask=torch.ones_like(old),
        old_log_probs=old,advantages=torch.zeros_like(old),input_ids=torch.zeros(len(items),1,dtype=torch.long),
        attention_mask=torch.ones(len(items),1,dtype=torch.long),position_ids=torch.zeros(len(items),1,dtype=torch.long)),
        non_tensors={'multi_modal_inputs':packed},meta_info=dict(temperature=1.,use_dynamic_bsz=False,pad_token_id=0))


def admit(batch,config):
    summary=dict(error=None,ids=[],real=0,signature=None)
    try:
        if len(batch)!=config.ppo_mini_batch_size or batch.meta_info['temperature']!=1.:
            raise ValueError('complete local native minibatch required')
        for i,p in enumerate(batch.non_tensor_batch['multi_modal_inputs']):
            if p['policy_id']!=config.jev_policy_id:raise ValueError('stale context policy')
            options={k:p[k] for k in ('stage','ranking_coefficient','probability_coefficient','algorithm',
                                      'kl_coefficient','reference_policy_id')}
            if options!=dict(config.jev_context_objective):raise ValueError('objective config differs')
            # Rebuild only per-query semantics, group estimator and transport.
            # A zero-weight tail can be locally all padding; temporary real=True
            # here only permits this validation, never changes global reduction.
            check=make_batch([dict(p,real_request=True)],**options)
            rebuilt=check.non_tensor_batch['multi_modal_inputs'][0]
            for name in ('old_joint','advantages','rewards'):
                if name in p and p[name] is not None and not torch.equal(p[name],rebuilt[name]):
                    raise ValueError('sampled group metadata changed: '+name)
            for name in ('responses','response_mask','old_log_probs','advantages'):
                if not torch.equal(batch.batch[name][i].cpu(),check.batch[name][0]):
                    raise ValueError('context transport changed: '+name)
            if type(p['real_request']) is not bool or (not p['real_request'] and p['request_weight']!=0):
                raise ValueError('nonzero padding weight')
            sig=(p['policy_id'],p['global_rows'],p['global_real_requests'],p['labels'].shape[1],
                 None if p['actions'] is None else p['actions'].shape[1],
                 config.ppo_mini_batch_size,config.ppo_micro_batch_size_per_gpu,json.dumps(options,sort_keys=True))
            if summary['signature'] is not None and summary['signature']!=sig:raise ValueError('mixed objectives')
            summary['signature']=sig;summary['ids'].append(p['request_id']);summary['real']+=int(p['real_request'])
    except Exception as exc:summary['error']=repr(exc)
    peers=[None]*dist.get_world_size();dist.all_gather_object(peers,summary)
    if any(p['error'] for p in peers):raise ValueError('collective context admission: '+str(peers))
    ids=[uid for p in peers for uid in p['ids']];sig=summary['signature']
    if (len({p['signature'] for p in peers})!=1 or len(ids)!=len(set(ids)) or len(ids)!=sig[1] or
            sum(p['real'] for p in peers)!=sig[2] or sig[2]<1):
        raise ValueError('global context population or denominator differs')
    return dict(global_rows=len(ids),global_real_requests=sig[2],world_size=len(peers))


def registered_loss(old_log_prob,log_prob,advantages,response_mask,loss_agg_mode='token-mean',
                    config=None,rollout_is_weights=None,**kwargs):
    state=_active.get()
    if (state is None or state['consumed'] or state['log_probs'] is not log_prob or
            config.policy_loss.loss_mode!=MODE or loss_agg_mode!='token-mean' or rollout_is_weights is not None):
        raise ValueError('one exact context forward per native loss required')
    state['consumed']=True
    total=log_prob.sum()*0;metrics={}
    for i,p in enumerate(state['payload']):
        device=log_prob.device
        move=lambda v:None if v is None else v.to(device)
        value,_=loss(state['logits'][i:i+1],move(p['labels']),move(p['observed']),
            log_prob.new_tensor([p['request_weight']]),torch.tensor([p['real_request']],device=device),
            stage=p['stage'],global_real_requests=p['global_real_requests'],world_size=dist.get_world_size(),
            ranking_coefficient=p['ranking_coefficient'],probability_coefficient=p['probability_coefficient'],
            actions=move(p['actions']),old_joint=move(p['old_joint']),advantages=move(p['advantages']),
            reference_policy=None if p['reference_policy'] is None else tuple(move(v) for v in p['reference_policy']),
            kl_coefficient=p['kl_coefficient'],clip_ratio=config.clip_ratio)
        total=total+value
    metrics['jev/context_loss']=float(total.detach())
    return total*(config.ppo_mini_batch_size//config.ppo_micro_batch_size_per_gpu),metrics


def build_actor(config,module,optimizer,*,collation):
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY,register_policy_loss
    if file_hash(inspect.getfile(DataParallelPPOActor))!=ACTOR_SHA256:raise ValueError('native actor source changed')
    if file_hash(inspect.getfile(core_algos))!=CORE_SHA256:raise ValueError('native loss source changed')
    validate_bounds(dict(config.jev_context_parity))
    size,micro=config.ppo_mini_batch_size,config.ppo_micro_batch_size_per_gpu
    if (any(type(v) is not int or v<1 for v in (size,micro)) or size%micro or
            config.ppo_epochs!=1 or config.use_dynamic_bsz or config.use_remove_padding or config.use_fused_kernels or
            config.use_prefix_grouper or config.ulysses_sequence_parallel_size!=1 or config.use_kl_loss or
            config.calculate_entropy or config.entropy_coeff!=0 or config.calculate_sum_pi_squared or
            not config.use_rollout_log_probs or config.loss_agg_mode!='token-mean' or config.global_batch_info or
            config.policy_loss.loss_mode!=MODE):raise ValueError('static single-update context actor required')
    if any(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p for m in module.modules()):
        raise ValueError('nonzero dropout invalidates parity')
    if MODE in POLICY_LOSS_REGISTRY and POLICY_LOSS_REGISTRY[MODE] is not registered_loss:raise ValueError('registry collision')
    register_policy_loss(MODE)(registered_loss)

    class ContextActor(DataParallelPPOActor):
        def _forward_micro_batch(self,micro_batch,temperature,calculate_entropy=False):
            if temperature!=1. or calculate_entropy:raise ValueError('untransformed context policy required')
            state=_active.get()
            if state is not None and not state['consumed']:raise ValueError('unconsumed previous microbatch')
            payload=list(micro_batch['multi_modal_inputs'])
            inputs=[{k:p[k] for k in ('request_id','policy_id','inputs','positions')} for p in payload]
            logits=forward_context_audio_batch(self.actor_module,inputs,
                device=next(self.actor_module.parameters()).device,dtype=self.param_dtype,
                policy_id=self.config.jev_policy_id,**collation)
            logp=torch.cat(binary_logprobs(logits),-1)
            if payload[0]['stage']=='task':
                report=compare_behavior(logp,micro_batch['old_log_probs'],dict(self.config.jev_context_parity))
                reports=[None]*dist.get_world_size()
                dist.all_gather_object(reports,report)
                if not all(r['passed'] for r in reports):
                    raise ValueError('context behavior parity failed before optimizer: '+str(reports))
                self.last_behavior_parity=reports
                self.behavior_parity_history.append(reports)
            if state is not None:state.update(log_probs=logp,logits=logits,payload=payload,consumed=False)
            return {'log_probs':logp}

        def update_policy(self,data):
            self.behavior_parity_history=[]
            self.last_admission=admit(data,self.config)
            if _active.get() is not None:raise ValueError('nested context update')
            token=_active.set(dict(consumed=True))
            try:
                result=super().update_policy(data)
                if not _active.get()['consumed']:raise ValueError('unconsumed final context forward')
                return result
            finally:_active.reset(token)
    return ContextActor(config,module,optimizer)
