"""Typed-action bridge to pinned verl's native actor update loop.

The native response axis has length one: it is a complete categorical action,
not a text token. Audio and Span positions remain input-only. This module keeps
upstream compute_log_prob/update_policy/optimizer stepping unchanged, replacing
only the token-language-model microbatch forward with typed candidate readout.
"""
import inspect
import numpy as np
import torch

from .baseline import file_hash
from .span_policy import joint_logprobs
from .span_actor_update import admit_actor

ACTOR_SHA256 = '207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce'


def make_batch(items, group_size, *, algorithm='grpo_v1'):
    """Build actual DataProto; derive GRPO advantages before any slicing.

    Each item supplies prepared input tensors, positions, candidate IDs, one
    authenticated SpanRollout and detached rewards. External provenance and
    gold-review admission are the caller's responsibility.
    """
    from verl import DataProto
    from .span_group_advantage import group_advantages
    if type(group_size) is not int or group_size < 2 or not items:
        raise ValueError('nonempty complete fixed-size request groups required')
    rows=[]; old=[]; rewards=[]; uids=[]; policies=set(); seen=set()
    for item in items:
        rollout=item['rollout']; reward=item['rewards']; inputs=item['inputs']
        if (len(rollout.request_ids)!=1 or rollout.actions.shape!=(1,group_size,len(item['candidates']))
                or reward.shape!=(1,group_size) or reward.requires_grad or not torch.isfinite(reward).all()):
            raise ValueError('one complete sampled request with detached rewards required')
        uid=rollout.request_ids[0]
        if uid in seen:raise ValueError('duplicate exact request')
        seen.add(uid);policies.add(rollout.policy_id)
        if set(inputs)-{'input_ids','attention_mask','position_ids','input_features','input_features_mask'}:
            raise ValueError('known label-free actor tensors required')
        if inputs['input_ids'].ndim!=2 or inputs['input_ids'].shape[0]!=1:
            raise ValueError('one complete encoded prompt per request required')
        if item['positions'].shape!=(1,len(item['candidates'])):
            raise ValueError('aligned Span positions required')
        # Validate saved complete action probabilities even before model forward.
        expected=joint_logprobs(rollout.candidate_logprobs,rollout.actions)
        if not torch.equal(expected,rollout.old_joint_logprobs):raise ValueError('old action probabilities changed')
        for sample in range(group_size):
            rows.append(dict(item=item,sample_index=sample,group_size=group_size))
            old.append(rollout.old_joint_logprobs[0,sample].detach().cpu())
            rewards.append(reward[0,sample].detach().cpu());uids.append(uid)
    if len(policies)!=1:raise ValueError('one immutable behavior policy per update required')
    old=torch.stack(old).reshape(-1,1);rewards=torch.stack(rewards).reshape(-1,1)
    mask=torch.ones_like(rewards)
    advantages=group_advantages(rewards.reshape(-1,group_size),uids[::group_size],
                                algorithm=algorithm).reshape(-1,1)
    # Transport placeholders are never passed to the backbone. The full exact
    # encoded audio prompt resides in multi_modal_inputs, retained by native
    # DataProto.select/split. responses stores sample ordinal, not token IDs.
    ordinals=torch.tensor([r['sample_index'] for r in rows],dtype=torch.long)[:,None]
    payload=np.empty(len(rows),dtype=object)
    payload[:]=rows
    return DataProto.from_dict(tensors=dict(responses=ordinals,response_mask=mask,
        input_ids=torch.zeros_like(ordinals),attention_mask=torch.ones_like(ordinals),position_ids=torch.zeros_like(ordinals),
        old_log_probs=old,advantages=advantages),non_tensors={'multi_modal_inputs':payload},
        meta_info=dict(temperature=1.,micro_batch_size=group_size,use_dynamic_bsz=False,pad_token_id=0,
                       jev_advantage_algorithm=algorithm))


def typed_forward(actor, micro_batch, temperature, calculate_entropy=False):
    if temperature!=1. or calculate_entropy:raise ValueError('untransformed typed policy without entropy term required')
    payload=list(micro_batch['multi_modal_inputs']); size=actor.config.jev_group_size
    if not payload or len(payload)%size:raise ValueError('microbatch splits a request group')
    outputs=[];seen=set()
    # Native update_policy switches train(); reject dropout at construction and
    # retain it here. No stochastic text decoding or answer-prefix injection.
    device=next(actor.actor_module.parameters()).device
    for start in range(0,len(payload),size):
        group=payload[start:start+size];item=group[0]['item'];rollout=item['rollout'];uid=rollout.request_ids[0]
        if rollout.policy_id!=actor.config.jev_policy_id:
            raise ValueError('stale actor policy identity')
        if uid in seen or [r['sample_index'] for r in group]!=list(range(size)):
            raise ValueError('duplicate or incomplete ordered sample group')
        seen.add(uid)
        if any(r['group_size']!=size or r['item'] is not item for r in group):
            raise ValueError('group payload identity differs')
        if micro_batch['responses'][start:start+size,0].tolist()!=list(range(size)):
            raise ValueError('sample ordinal transport changed')
        inputs={k:v.to(device=device,dtype=actor.param_dtype if v.is_floating_point() else v.dtype) for k,v in item['inputs'].items()}
        probabilities=actor.actor_module(inputs,item['positions'].to(device),item['candidates'])[1]
        admit_actor(rollout,probabilities,rollout.policy_id,rollout.request_ids)
        if 'old_log_probs' in micro_batch and not torch.equal(
                micro_batch['old_log_probs'][start:start+size].detach().cpu(),rollout.old_joint_logprobs.reshape(-1,1)):
            raise ValueError('native old action logprob transport changed')
        outputs.append(joint_logprobs(probabilities,rollout.actions.to(device)).reshape(size,1))
    return {'log_probs':torch.cat(outputs,0)}


def build_actor(config,module,optimizer=None):
    """Construct native subclass after caller initializes runtime/process group.

    Equal complete groups per rank are required by the planned DP launcher;
    distributed equivalence is a separate gate, not implied by construction.
    """
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    if file_hash(inspect.getfile(DataParallelPPOActor))!=ACTOR_SHA256:
        raise ValueError('pinned native actor source differs')
    size=config.ppo_mini_batch_size;micro=config.ppo_micro_batch_size_per_gpu;group=config.jev_group_size
    if (any(type(v) is not int or v<1 for v in (size,micro,group)) or group<2
            or size%micro or micro%group or micro>size or config.ppo_epochs!=1
            or config.use_dynamic_bsz or config.use_remove_padding or config.use_fused_kernels
            or config.ulysses_sequence_parallel_size!=1 or config.use_prefix_grouper
            or config.use_kl_loss or config.calculate_entropy or config.entropy_coeff!=0
            or config.calculate_sum_pi_squared or not config.use_rollout_log_probs
            or config.loss_agg_mode!='token-mean' or config.policy_loss.loss_mode!='vanilla'
            or config.global_batch_info):
        raise ValueError('one static complete-group minibatch and original rollout logprobs required')
    for layer in module.modules():
        if isinstance(layer,torch.nn.modules.dropout._DropoutNd) and layer.p!=0:
            raise ValueError('nonzero dropout invalidates actor/rollout parity')
    class NativeTypedSpanActor(DataParallelPPOActor):
        def _forward_micro_batch(self,micro_batch,temperature,calculate_entropy=False):
            return typed_forward(self,micro_batch,temperature,calculate_entropy)
        def update_policy(self,data):
            if len(data)!=self.config.ppo_mini_batch_size:
                raise ValueError('exactly one full native minibatch required')
            if data.meta_info.get('jev_advantage_algorithm','grpo_v1') != self.config.get('jev_advantage_algorithm','grpo_v1'):
                raise ValueError('configured group advantage differs from behavior batch')
            return super().update_policy(data)
    return NativeTypedSpanActor(config,module,optimizer)
