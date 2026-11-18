"""Dual-credit DataProto/loss adapter over the pinned native verl actor loop.

Only transport admission and objective hooks differ. The original actor source,
forward collation, gradient accumulation, clipping and optimizer remain intact.
This module does not dispatch training or certify the live model identity.
"""
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.distributed as dist

from multimodal_jev.context_dual_credit import ALGORITHMS,ARMS,verify,current_terms,hybrid_loss,log_probabilities
from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.context_logits_wire import validate_logits_record,WIRE

ACTOR_SHA='067297bc726d2b607776d40cf6e155f98649ba9405f1d025b43d803e9c1215b0'
MODE='jev_context_dual_credit_candidate_v1'
OPTION_KEYS=('algorithm','arm','ranking_coefficient','probability_coefficient','kl_coefficient','reference_policy_id')
_loaded=None


def options_check(options):
    if (set(options)!=set(OPTION_KEYS) or options['algorithm'] not in ALGORITHMS
            or options['arm'] not in ARMS or options['ranking_coefficient']!=.1
            or options['probability_coefficient']!=.3 or options['kl_coefficient']!=.01
            or type(options['reference_policy_id'])!=str or not options['reference_policy_id']):
        raise ValueError('fixed dual task objective required')


def validate_item(p,options):
    options_check(options)
    record=verify(p['dual_credit']);b=record['binding'];ident=p['dual_identity']
    if (set(ident)!={'policy_id','input_id','query_id','update'}
            or any(ident[k]!=b[k] for k in ident)
            or p['request_id']!=b['query_id'] or p['policy_id']!=b['policy_id']
            or list(p['candidate_ids'])!=b['candidates'] or b['algorithm']!=options['algorithm']
            or b['group_size']!=16 or b['seed']!=317
            or type(p['real_request'])!=bool or p['request_weight']!=(1. if p['real_request'] else 0.)):
        raise ValueError('dual behavior/query/options binding differs')
    labels=p['labels'];observed=p['observed'];positions=p['positions'];n=len(b['candidates'])
    if (labels.dtype!=torch.bool or observed.dtype!=torch.bool or labels.shape!=(1,n)
            or observed.shape!=(1,n) or labels.device.type!='cpu' or observed.device.type!='cpu'
            or labels[0].tolist()!=b['labels'] or not observed.all()
            or positions.dtype!=torch.long or positions.shape!=(1,n) or positions.device.type!='cpu'
            or set(p['inputs'])-{'input_ids','attention_mask','position_ids','input_features','input_features_mask'}
            or p['inputs']['input_ids'].ndim!=2 or p['inputs']['input_ids'].shape[0]!=1):
        raise ValueError('complete CPU truth/positions and label-free input required')
    # Recompute actual tensor binding; a self-consistent credit record alone
    # must not permit pairing another audio query with its probability vector.
    from multimodal_jev.span_request_identity import tensor_identity
    identity=dict(policy_id=p['policy_id'],positions=positions[0].tolist(),
        inputs={k:tensor_identity(v) for k,v in sorted(p['inputs'].items())})
    raw=dict(policy_id=p['policy_id'],request_identity=identity,request_id=b['input_id'],
             prob=b['probability'],logits=b['logits'],output_protocol=WIRE)
    validate_logits_record(raw)
    bind_rollout(raw,p['inputs'],positions,policy_id=p['policy_id'],group_size=2,seed=0)
    ref=p['reference_probability']
    if (p['reference_policy_id']!=options['reference_policy_id'] or ref.shape!=(n,)
            or ref.device.type!='cpu' or ref.requires_grad):
        raise ValueError('detached fixed reference required')
    log_probabilities(ref)
    return log_probabilities(torch.tensor(b['probability'],dtype=torch.float32)).flatten()


def make_batch(items,**options):
    from verl import DataProto
    options_check(options)
    if not items:raise ValueError('nonempty complete batch required')
    real=sum(p['real_request'] is True for p in items)
    if not real:raise ValueError('nonempty real population required')
    payload=[];old=[];signatures=set();ids=set()
    for item in items:
        value=validate_item(item,options)
        if item['request_id'] in ids:raise ValueError('duplicate scheduled query')
        ids.add(item['request_id'])
        signatures.add((item['policy_id'],tuple(item['candidate_ids']),item['dual_identity']['update']))
        payload.append(dict(item,dual_options=dict(options),global_rows=len(items),global_real_requests=real,stage='task'))
        old.append(value)
    if len(signatures)!=1:raise ValueError('one policy/candidate order/update per batch required')
    old=torch.stack(old);packed=np.empty(len(payload),dtype=object);packed[:]=payload
    return DataProto.from_dict(tensors=dict(responses=torch.arange(old.shape[1]).repeat(len(items),1),
        response_mask=torch.ones_like(old),old_log_probs=old,advantages=torch.zeros_like(old),
        input_ids=torch.zeros(len(items),1,dtype=torch.long),attention_mask=torch.ones(len(items),1,dtype=torch.long),
        position_ids=torch.zeros(len(items),1,dtype=torch.long)),non_tensors={'multi_modal_inputs':packed},
        meta_info=dict(temperature=1.,use_dynamic_bsz=False,pad_token_id=0))


def admit(batch,config):
    summary=dict(error=None,ids=[],real=0,signature=None)
    try:
        options=dict(config.jev_context_objective);options_check(options)
        if len(batch)!=config.ppo_mini_batch_size or batch.meta_info['temperature']!=1.:
            raise ValueError('complete native local minibatch required')
        for i,p in enumerate(batch.non_tensor_batch['multi_modal_inputs']):
            old=validate_item(p,options)
            if p['policy_id']!=config.jev_policy_id or p['dual_options']!=options or p['stage']!='task':
                raise ValueError('stale policy or changed objective')
            expected=dict(old_log_probs=old,response_mask=torch.ones_like(old),advantages=torch.zeros_like(old),
                          responses=torch.arange(len(old)))
            if any(not torch.equal(batch.batch[k][i].cpu(),v) for k,v in expected.items()):
                raise ValueError('dual transport changed')
            if any(type(p[k])!=int or p[k]<1 for k in ('global_rows','global_real_requests')):
                raise ValueError('integer global denominators required')
            sig=(p['policy_id'],p['global_rows'],p['global_real_requests'],tuple(p['candidate_ids']),
                p['dual_identity']['update'],config.ppo_mini_batch_size,config.ppo_micro_batch_size_per_gpu,
                json.dumps(options,sort_keys=True))
            if summary['signature'] not in (None,sig):raise ValueError('mixed update')
            summary['signature']=sig;summary['ids'].append(p['request_id']);summary['real']+=int(p['real_request'])
    except Exception as exc:summary['error']=repr(exc)
    peers=[None]*dist.get_world_size();dist.all_gather_object(peers,summary)
    if any(p['error'] for p in peers):raise ValueError('collective dual admission: '+str(peers))
    ids=[uid for p in peers for uid in p['ids']];sig=summary['signature']
    if (len({p['signature'] for p in peers})!=1 or len(ids)!=len(set(ids))
            or len(ids)!=sig[1] or sum(p['real'] for p in peers)!=sig[2]):
        raise ValueError('global dual query population/denominator differs')
    return dict(global_rows=len(ids),global_real_requests=sig[2],world_size=len(peers))


def check_dual_forward(logits,payload):
    report=dict(passed=True,error=None)
    try:
        for z,p in zip(logits,payload,strict=True):
            current_terms(z.detach(),p['dual_credit'],**p['dual_identity'])
    except Exception as exc:report=dict(passed=False,error=repr(exc))
    reports=[None]*dist.get_world_size();dist.all_gather_object(reports,report)
    if not all(r['passed'] for r in reports):raise ValueError('collective dual behavior parity: '+str(reports))
    return reports


def load(source_root):
    """Isolated hooks; the hash-pinned build_actor/verl update implementation is reused."""
    global _loaded
    path=Path(source_root)/'src/multimodal_jev/context_native_actor.py'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ACTOR_SHA:raise ValueError('original actor source changed')
    if _loaded is not None:return _loaded
    text=raw.decode()
    before="                report=compare_behavior(logp,micro_batch['old_log_probs'],dict(self.config.jev_context_parity))\n                reports=[None]*dist.get_world_size()\n                dist.all_gather_object(reports,report)\n                if not all(r['passed'] for r in reports):\n                    raise ValueError('context behavior parity failed before optimizer: '+str(reports))"
    if text.count(before)!=1:raise ValueError('actor forward parity hook changed')
    text=text.replace(before,'                reports=check_dual_forward(logits,payload)')
    # A rank-local saturated sigmoid can throw before peers enter their parity
    # collective. All ranks must check actual logits first, then form logp.
    before="            logp=torch.cat(binary_logprobs(logits),-1)\n            if payload[0]['stage']=='task':\n                reports=check_dual_forward(logits,payload)"
    after="            reports=check_dual_forward(logits,payload)\n            logp=torch.cat(binary_logprobs(logits),-1)\n            if payload[0]['stage']=='task':"
    if text.count(before)!=1:raise ValueError('dual pre-logarithm parity hook changed')
    text=text.replace(before,after)
    module=types.ModuleType('multimodal_jev._dual_credit_actor_candidate')
    module.__package__='multimodal_jev';exec(compile(text,'<dual-credit-actor>','exec'),module.__dict__)
    from tools.reports.context_probability_policy_v2 import binary_logprobs
    module.MODE=MODE;module._active=ContextVar('jev_dual_credit_active',default=None)
    module.binary_logprobs=binary_logprobs;module.make_batch=make_batch;module.admit=admit
    module.check_dual_forward=check_dual_forward
    def registered_loss(old_log_prob,log_prob,advantages,response_mask,loss_agg_mode='token-mean',
                        config=None,rollout_is_weights=None,**kwargs):
        state=module._active.get()
        if (state is None or state['consumed'] or state['log_probs'] is not log_prob
                or config.policy_loss.loss_mode!=MODE or loss_agg_mode!='token-mean'
                or rollout_is_weights is not None or config.clip_ratio!=.2):
            raise ValueError('one exact dual forward per native loss required')
        state['consumed']=True;p=state['payload']
        total,detail=hybrid_loss(state['logits'],[v['dual_credit'] for v in p],
            [v['dual_identity'] for v in p],torch.stack([v['reference_probability'] for v in p]),
            torch.tensor([v['real_request'] for v in p],dtype=torch.bool),
            global_real_requests=p[0]['global_real_requests'],world_size=dist.get_world_size(),arm=p[0]['dual_options']['arm'])
        metrics={'jev/context_loss':float(total.detach()),'jev/dual_rl':detail['rl'],
                 'jev/dual_auxiliary':detail['auxiliary'],'jev/dual_kl_unweighted':detail['kl']}
        # Native verl divides by accumulation count. Objective is already
        # normalized by real global queries, so compensate exactly once.
        return total*(config.ppo_mini_batch_size//config.ppo_micro_batch_size_per_gpu),metrics
    module.registered_loss=registered_loss
    _loaded=module
    return module
