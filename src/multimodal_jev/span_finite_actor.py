"""Finite-support quadrature transported through the pinned native verl actor.

One row is one complete request, columns enumerate its marginal candidates.
Padding has zero quadrature mass. Token placeholders never reach the backbone.
Equal rows per DP rank and one local optimizer minibatch are required.
"""
import inspect
import math
import numpy as np
import torch
import torch.distributed as dist

from .baseline import file_hash
from .span_native_actor import ACTOR_SHA256
from .span_finite_credit import credit,validate_policy
from .span_parity import compare_span_policies


def make_global_batch(items):
    """Build on controller, then split complete rows equally among DP ranks.

    Targets remain outside model payload. The caller authenticates source,
    review masks, exact request identities and behavior-policy artifacts.
    """
    from verl import DataProto
    if not items:raise ValueError('nonempty global requests required')
    ids=[x['request_id'] for x in items];policies={x['policy_id'] for x in items}
    if len(ids)!=len(set(ids)) or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError('distinct exact request identities required')
    if len(policies)!=1 or any(not isinstance(x,str) or not x for x in policies):
        raise ValueError('single immutable policy identity required')
    prepared=[]
    for item in items:
        old=item['behavior'];validate_policy(old,1,detached=True)
        if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in old):
            raise ValueError('controller behavior must be detached CPU FP32')
        candidates=item['candidates'];inputs=item['inputs'];positions=item['positions']
        if (len(old)!=len(candidates) or positions.shape!=(1,len(old))
                or any(len(c)!=p.shape[1] or len(c)!=len(set(c)) for c,p in zip(candidates,old,strict=True))):
            raise ValueError('complete aligned candidate support required')
        if (set(inputs)-{'input_ids','attention_mask','position_ids','input_features','input_features_mask'}
                or inputs['input_ids'].ndim!=2 or inputs['input_ids'].shape[0]!=1):
            raise ValueError('one label-free model input required')
        y=item['targets'];mask=item['supervised'];w=item['span_weights'];rw=item['request_weight']
        if (type(rw) not in (int,float) or not math.isfinite(rw) or rw<0
                or y.shape!=(1,len(old)) or y.dtype!=torch.long or mask.shape!=y.shape
                or mask.dtype!=torch.bool or w.shape!=y.shape or w.requires_grad
                or not w.is_floating_point() or not torch.isfinite(w).all() or (w<0).any()):
            raise ValueError('finite detached reward weights and aligned targets required')
        active=bool((w*mask).sum()>0)
        if active:
            advantages=credit(old,y,mask,w)['advantages']
        else:
            for j,p in enumerate(old):
                known=y[:,j][mask[:,j]]
                if ((known<0)|(known>=p.shape[1])).any():raise ValueError('target outside support')
            advantages=tuple(torch.zeros_like(p) for p in old)
        effective_weight=float(rw) if active else 0.
        payload={k:item[k] for k in ('request_id','policy_id','inputs','positions','candidates')}
        payload.update(behavior=tuple(p.clone() for p in old),
                       advantages=tuple(p.clone() for p in advantages),effective_weight=effective_weight)
        prepared.append(payload)
    total_weight=sum(x['effective_weight'] for x in prepared)
    if total_weight<=0:raise ValueError('nonzero global effective request weight required')
    width=max(sum(p.shape[1] for p in x['behavior']) for x in prepared)
    count=len(prepared);old=torch.zeros(count,width);adv=torch.zeros_like(old);quad=torch.zeros_like(old)
    scale=count*width/total_weight
    for i,payload in enumerate(prepared):
        value=torch.cat(payload['behavior'],-1)[0];n=len(value)
        old[i,:n]=value;adv[i,:n]=torch.cat(payload['advantages'],-1)[0]
        quad[i,:n]=value.exp()*payload['effective_weight']*scale
        payload.update(width=width,global_rows=count,global_weight=total_weight)
    payloads=np.empty(count,dtype=object);payloads[:]=prepared
    responses=torch.arange(width).repeat(count,1)
    return DataProto.from_dict(tensors=dict(responses=responses,response_mask=torch.ones_like(old),
        old_log_probs=old,advantages=adv,rollout_is_weights=quad,input_ids=torch.zeros(count,1,dtype=torch.long),
        attention_mask=torch.ones(count,1,dtype=torch.long),position_ids=torch.zeros(count,1,dtype=torch.long)),
        non_tensors={'multi_modal_inputs':payloads},meta_info=dict(temperature=1.,use_dynamic_bsz=False,
        micro_batch_size=1,pad_token_id=0))


def _transport(payload):
    width=payload['width'];old=torch.cat(payload['behavior'],-1)[0]
    adv=torch.cat(payload['advantages'],-1)[0]
    scale=payload['global_rows']*width/payload['global_weight']
    quad=old.exp()*payload['effective_weight']*scale
    pad=lambda x:torch.nn.functional.pad(x,(0,width-len(x)))
    return dict(old_log_probs=pad(old),advantages=pad(adv),rollout_is_weights=pad(quad))


def admit_batch(batch,config):
    """Collective transport/normalization check before model collectives."""
    summary=dict(error=None,ids=[],weight=0.,rows=len(batch),width=batch.batch['responses'].shape[-1],
                 micro=config.ppo_micro_batch_size_per_gpu,policy=config.jev_policy_id,global_rows=None,global_weight=None)
    try:
        if len(batch)!=config.ppo_mini_batch_size:raise ValueError('one full local minibatch required')
        if batch.meta_info['temperature']!=1. or batch.meta_info['use_dynamic_bsz']:
            raise ValueError('static untransformed policy required')
        payload=list(batch.non_tensor_batch['multi_modal_inputs'])
        for i,p in enumerate(payload):
            if p['policy_id']!=config.jev_policy_id or p['width']!=summary['width']:
                raise ValueError('policy or candidate width differs')
            if i and (summary['global_rows']!=p['global_rows'] or summary['global_weight']!=p['global_weight']):
                raise ValueError('inconsistent global normalization')
            summary.update(global_rows=p['global_rows'],global_weight=p['global_weight'])
            if not torch.equal(batch.batch['responses'][i].cpu(),torch.arange(p['width'])):
                raise ValueError('candidate ordinals changed')
            if not torch.all(batch.batch['response_mask'][i]==1):raise ValueError('quadrature mask changed')
            for key,value in _transport(p).items():
                if not torch.equal(batch.batch[key][i].detach().cpu(),value):
                    raise ValueError('finite candidate transport changed: '+key)
            summary['ids'].append(p['request_id']);summary['weight']+=p['effective_weight']
    except Exception as exc:summary['error']=str(exc)
    peers=[None]*dist.get_world_size();dist.all_gather_object(peers,summary)
    if any(p['error'] for p in peers):raise ValueError('collective finite admission failed: '+str([p['error'] for p in peers]))
    shape={(p['rows'],p['width'],p['micro'],p['policy'],p['global_rows'],p['global_weight']) for p in peers}
    if len(shape)!=1:raise ValueError('inconsistent rank shape or global normalization')
    ids=[uid for p in peers for uid in p['ids']]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate rank requests')
    if (sum(p['rows'] for p in peers)!=summary['global_rows']
            or not math.isclose(sum(p['weight'] for p in peers),summary['global_weight'],rel_tol=1e-12,abs_tol=1e-12)):
        raise ValueError('global denominator does not match actual rank population')
    return dict(passed=True,world_size=len(peers),global_requests=len(ids),
                global_effective_weight=summary['global_weight'],rank_effective_weights=[p['weight'] for p in peers])


def finite_forward(actor,micro_batch,temperature,calculate_entropy=False):
    if temperature!=1. or calculate_entropy:raise ValueError('untransformed finite policy required')
    outputs=[];device=next(actor.actor_module.parameters()).device
    for p in micro_batch['multi_modal_inputs']:
        if p['policy_id']!=actor.config.jev_policy_id:raise ValueError('stale policy')
        inputs={k:v.to(device=device,dtype=actor.param_dtype if v.is_floating_point() else v.dtype) for k,v in p['inputs'].items()}
        logp=actor.actor_module(inputs,p['positions'].to(device),p['candidates'])[1]
        if len(logp)!=len(p['behavior']) or any(a.shape!=b.shape for a,b in zip(logp,p['behavior'],strict=True)):
            raise ValueError('model support differs')
        audit=compare_span_policies([x[0].detach().cpu().tolist() for x in logp],
                                    [x[0].tolist() for x in p['behavior']])
        if not audit['passed']:raise ValueError('complete candidate parity failed')
        joined=torch.cat(logp,-1)
        outputs.append(torch.nn.functional.pad(joined,(0,p['width']-joined.shape[-1])))
    return dict(log_probs=torch.cat(outputs,0))


def build_actor(config,module,optimizer):
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    if file_hash(inspect.getfile(DataParallelPPOActor))!=ACTOR_SHA256:raise ValueError('native actor source changed')
    size=config.ppo_mini_batch_size;micro=config.ppo_micro_batch_size_per_gpu
    if (any(type(x) is not int or x<1 for x in (size,micro)) or size%micro
            or config.ppo_epochs!=1 or config.use_dynamic_bsz or config.use_remove_padding
            or config.use_fused_kernels or config.use_prefix_grouper or config.ulysses_sequence_parallel_size!=1
            or config.use_kl_loss or config.calculate_entropy or config.entropy_coeff!=0
            or config.calculate_sum_pi_squared or not config.use_rollout_log_probs
            or config.loss_agg_mode!='token-mean' or config.policy_loss.loss_mode!='vanilla' or config.global_batch_info):
        raise ValueError('one static finite-support update with original behavior probabilities required')
    if any(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p for m in module.modules()):
        raise ValueError('dropout breaks behavior parity')
    class FiniteActor(DataParallelPPOActor):
        def _forward_micro_batch(self,micro_batch,temperature,calculate_entropy=False):
            return finite_forward(self,micro_batch,temperature,calculate_entropy)
        def update_policy(self,data):
            self.last_admission=admit_batch(data,self.config)
            return super().update_policy(data)
    return FiniteActor(config,module,optimizer)
