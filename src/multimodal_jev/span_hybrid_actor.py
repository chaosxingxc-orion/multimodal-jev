"""Explicit hybrid loss registration in pinned verl's unchanged update loop.

ContextVar carries supervised metadata between the typed forward and the loss
callback on the same actor thread. Labels never enter the backbone inputs.
Native verl still owns accumulation, backward, clipping and optimizer stepping.
"""
from contextvars import ContextVar
import inspect
import math
import numpy as np
import torch
import torch.distributed as dist

from .baseline import file_hash
from .span_native_actor import ACTOR_SHA256
from .span_hybrid_objective import loss, supervised_numerator
from .span_finite_credit import validate_policy
from .span_parity import compare_span_policies

MODE = 'jev_candidate_hybrid_v1'
_active = ContextVar('jev_hybrid_active', default=None)


def make_batch(items, *, stage, coefficient=0.):
    from verl import DataProto
    if not items or stage not in ('instruction', 'task'):
        raise ValueError('nonempty explicit two-stage batch required')
    if (type(coefficient) not in (float, int) or not math.isfinite(coefficient) or coefficient < 0
            or (stage == 'instruction' and coefficient != 0)):
        raise ValueError('explicit nonnegative task-stage coefficient required')
    payload = []
    for item in items:
        p = {k: item[k] for k in ('request_id', 'policy_id', 'inputs', 'positions', 'candidates',
                                  'targets', 'supervised', 'span_weights', 'request_weight')}
        if (set(p['inputs'])-{'input_ids', 'attention_mask', 'position_ids', 'input_features', 'input_features_mask'}
                or p['inputs']['input_ids'].ndim != 2 or p['inputs']['input_ids'].shape[0] != 1
                or p['positions'].shape != (1, len(p['candidates']))
                or p['positions'].dtype != torch.long
                or any(len(c) < 2 or len(c) != len(set(c)) for c in p['candidates'])):
            raise ValueError('one label-free complete typed request required')
        rw = p['request_weight']
        if type(rw) not in (int, float) or not math.isfinite(rw) or rw < 0:
            raise ValueError('finite nonnegative request weight required')
        # Synthetic distributions validate metadata only, never act as rollout.
        shapes = tuple(torch.full((1, len(c)), -math.log(len(c)), dtype=torch.float32) for c in p['candidates'])
        _, weight = supervised_numerator(shapes, p['targets'], p['supervised'], p['span_weights'], torch.tensor([rw]))
        p['effective_weight'] = float(weight)
        p['behavior'] = item.get('behavior')
        if stage == 'instruction':
            if p['behavior'] is not None:
                raise ValueError('instruction stage must not invent behavior')
        else:
            validate_policy(p['behavior'], 1, detached=True)
            if any(v.device.type != 'cpu' or v.dtype != torch.float32 or v.shape != s.shape
                   for v, s in zip(p['behavior'], shapes, strict=True)):
                raise ValueError('authenticated CPU FP32 full candidate behavior required')
        payload.append(p)
    ids = [p['request_id'] for p in payload]; policies = {p['policy_id'] for p in payload}
    if len(ids) != len(set(ids)) or len(policies) != 1 or any(not isinstance(v, str) or not v for v in ids+list(policies)):
        raise ValueError('distinct request slots and one immutable policy identity required')
    total = sum(p['effective_weight'] for p in payload)
    if total <= 0:
        raise ValueError('positive global effective weight required')
    width = max(sum(len(c) for c in p['candidates']) for p in payload)
    old = torch.zeros(len(payload), width)
    for i, p in enumerate(payload):
        p.update(global_rows=len(payload), global_weight=total, width=width, stage=stage, coefficient=float(coefficient))
        if stage == 'task':
            values = torch.cat(p['behavior'], -1)[0]; old[i, :len(values)] = values
    packed = np.empty(len(payload), dtype=object); packed[:] = payload
    # These tensors satisfy native actor transport. The custom registered loss
    # does not interpret instruction placeholders as action probabilities.
    return DataProto.from_dict(tensors=dict(responses=torch.arange(width).repeat(len(payload), 1),
        response_mask=torch.ones_like(old), old_log_probs=old, advantages=torch.zeros_like(old),
        input_ids=torch.zeros(len(payload), 1, dtype=torch.long),
        attention_mask=torch.ones(len(payload), 1, dtype=torch.long),
        position_ids=torch.zeros(len(payload), 1, dtype=torch.long)),
        non_tensors={'multi_modal_inputs': packed}, meta_info=dict(temperature=1., use_dynamic_bsz=False,
        micro_batch_size=1, pad_token_id=0))


def admit(batch, config):
    """All ranks reject malformed global normalization before model collectives."""
    record = dict(error=None, ids=[], local_weight=0., signature=None)
    try:
        if len(batch) != config.ppo_mini_batch_size or batch.meta_info['temperature'] != 1.:
            raise ValueError('one complete local optimizer batch required')
        for i, p in enumerate(batch.non_tensor_batch['multi_modal_inputs']):
            if p['policy_id'] != config.jev_policy_id:
                raise ValueError('stale policy')
            # Validate zero-effective rows without demanding positive local mass.
            shapes = tuple(torch.full((1, len(c)), -math.log(len(c))) for c in p['candidates'])
            _, effective = supervised_numerator(shapes, p['targets'], p['supervised'], p['span_weights'],
                                                torch.tensor([p['request_weight']]))
            if abs(float(effective)-p['effective_weight']) > 1e-7:
                raise ValueError('effective source weight changed')
            old = torch.zeros(p['width'])
            if p['stage'] == 'task':
                validate_policy(p['behavior'], 1, detached=True)
                values = torch.cat(p['behavior'], -1)[0]; old[:len(values)] = values
            elif p['stage'] != 'instruction' or p['behavior'] is not None or p['coefficient'] != 0:
                raise ValueError('instruction transport changed')
            if (not torch.equal(batch.batch['old_log_probs'][i].cpu(), old)
                    or not torch.all(batch.batch['advantages'][i] == 0)
                    or not torch.all(batch.batch['response_mask'][i] == 1)
                    or not torch.equal(batch.batch['responses'][i].cpu(), torch.arange(p['width']))):
                raise ValueError('hybrid transport changed')
            sig = (p['global_rows'], p['global_weight'], p['width'], p['stage'], p['coefficient'],
                   p['policy_id'], len(batch), config.ppo_micro_batch_size_per_gpu)
            if record['signature'] is not None and record['signature'] != sig:
                raise ValueError('inconsistent local metadata')
            record['signature'] = sig; record['ids'].append(p['request_id']); record['local_weight'] += p['effective_weight']
    except Exception as exc:
        record['error'] = repr(exc)
    peers = [None]*dist.get_world_size(); dist.all_gather_object(peers, record)
    if any(p['error'] for p in peers):
        raise ValueError('collective hybrid admission failed: '+str([p['error'] for p in peers]))
    ids = [uid for p in peers for uid in p['ids']]
    signatures = {p['signature'] for p in peers}; sig = record['signature']
    if (len(signatures) != 1 or len(ids) != len(set(ids)) or len(ids) != sig[0]
            or not math.isclose(sum(p['local_weight'] for p in peers), sig[1], rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError('global hybrid population/denominator differs')
    return dict(global_rows=len(ids), global_weight=sig[1], stage=sig[3], coefficient=sig[4],
                rank_weights=[p['local_weight'] for p in peers])


def registered_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode='token-mean',
                    config=None, rollout_is_weights=None, **kwargs):
    state = _active.get()
    if state is None or state['log_probs'] is not log_prob or state['consumed']:
        raise ValueError('hybrid loss must consume its exact typed forward once')
    if config.policy_loss.loss_mode != MODE or rollout_is_weights is not None or loss_agg_mode != 'token-mean':
        raise ValueError('unexpected hybrid loss dispatch')
    state['consumed'] = True
    total = log_prob.sum()*0; ce = 0.; rl = 0.
    for i, p in enumerate(state['payload']):
        start = 0; policy = []
        for c in p['candidates']:
            policy.append(log_prob[i:i+1, start:start+len(c)]); start += len(c)
        device = log_prob.device
        value, metrics = loss(tuple(policy), p['targets'].to(device), p['supervised'].to(device),
            p['span_weights'].to(device), torch.tensor([p['request_weight']], device=device),
            stage=p['stage'], global_effective_weight=p['global_weight'], world_size=dist.get_world_size(),
            behavior=None if p['behavior'] is None else tuple(x.to(device) for x in p['behavior']),
            coefficient=p['coefficient'], clip_ratio=config.clip_ratio)
        total = total+value; ce += metrics['ce']; rl += metrics['rl']
    # Native update_policy divides by accumulation. Undo that division because
    # each microbatch already uses the common global request denominator.
    accumulation = config.ppo_mini_batch_size//config.ppo_micro_batch_size_per_gpu
    return total*accumulation, {'jev/ce_contribution': ce, 'jev/rl_contribution': rl}


def build_actor(config, module, optimizer):
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, register_policy_loss
    if file_hash(inspect.getfile(DataParallelPPOActor)) != ACTOR_SHA256:
        raise ValueError('pinned native actor changed')
    size, micro = config.ppo_mini_batch_size, config.ppo_micro_batch_size_per_gpu
    if (any(type(x) is not int or x < 1 for x in (size, micro)) or size % micro
            or config.ppo_epochs != 1 or config.use_dynamic_bsz or config.use_remove_padding
            or config.use_fused_kernels or config.use_prefix_grouper or config.ulysses_sequence_parallel_size != 1
            or config.use_kl_loss or config.calculate_entropy or config.entropy_coeff != 0
            or config.calculate_sum_pi_squared or not config.use_rollout_log_probs
            or config.loss_agg_mode != 'token-mean' or config.policy_loss.loss_mode != MODE or config.global_batch_info):
        raise ValueError('one static native hybrid update required')
    if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p for m in module.modules()):
        raise ValueError('dropout breaks deterministic execution')
    if MODE in POLICY_LOSS_REGISTRY and POLICY_LOSS_REGISTRY[MODE] is not registered_loss:
        raise ValueError('hybrid registry collision')
    register_policy_loss(MODE)(registered_loss)

    class HybridActor(DataParallelPPOActor):
        def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
            if temperature != 1. or calculate_entropy:
                raise ValueError('untransformed typed forward required')
            state = _active.get(); outputs = []
            if state is not None and state.get('log_probs') is not None and not state['consumed']:
                raise ValueError('previous hybrid forward not consumed')
            device = next(self.actor_module.parameters()).device
            for p in micro_batch['multi_modal_inputs']:
                if p['policy_id'] != self.config.jev_policy_id:
                    raise ValueError('stale policy')
                inputs = {k: v.to(device=device, dtype=self.param_dtype if v.is_floating_point() else v.dtype)
                          for k, v in p['inputs'].items()}
                logp = self.actor_module(inputs, p['positions'].to(device), p['candidates'])[1]
                if p['stage'] == 'task' and not compare_span_policies(
                        [v[0].detach().cpu().tolist() for v in logp], [v[0].tolist() for v in p['behavior']])['passed']:
                    raise ValueError('complete behavior parity failed')
                joined = torch.cat(logp, -1)
                outputs.append(torch.nn.functional.pad(joined, (0, p['width']-joined.shape[-1])))
            joined = torch.cat(outputs, 0)
            if state is not None:
                state.update(log_probs=joined, payload=list(micro_batch['multi_modal_inputs']), consumed=False)
            return {'log_probs': joined}

        def update_policy(self, data):
            self.last_admission = admit(data, self.config)
            if _active.get() is not None:
                raise ValueError('nested hybrid update')
            token = _active.set(dict(log_probs=None, consumed=True))
            try:
                result = super().update_policy(data)
                if not _active.get()['consumed']:
                    raise ValueError('unconsumed final microbatch')
                return result
            finally:
                _active.reset(token)

    return HybridActor(config, module, optimizer)
