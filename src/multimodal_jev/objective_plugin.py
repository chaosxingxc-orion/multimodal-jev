"""Explicit file-based Stage2 objective extension with content identity.

Plugins are trusted project Python, not sandboxed code. The runtime retains
behavior verification and owns backward/optimizer/checkpoint operations.
"""
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace

import torch
from . import context_dual_credit as original


def continuous_advantages(rewards, algorithm):
    if (rewards.ndim != 2 or rewards.shape[1] < 2 or rewards.dtype != torch.float32
            or rewards.requires_grad or not torch.isfinite(rewards).all()):
        raise ValueError('finite detached FP32 component x group rewards required')
    if algorithm == original.ALGORITHMS[0]:
        return rewards-(rewards.sum(-1, keepdim=True)-rewards)/(rewards.shape[1]-1)
    if algorithm != original.ALGORITHMS[1]:
        raise ValueError('registered baseline algorithm required')
    active=original.grpo_settings() if hasattr(original,'grpo_settings') else None
    floor=.25 if active is None else active['advantage_std_floor']
    return (rewards-rewards.mean(-1, keepdim=True))/rewards.std(-1, unbiased=False, keepdim=True).clamp_min(floor)


class ObjectivePlugin:
    def __init__(self, config):
        if set(config)-{'numerics'} != {'path','sha256','options'} or not isinstance(config['options'],dict):
            raise ValueError('explicit objective file, hash and options required')
        path=Path(config['path'])
        if not path.is_file() or path.is_symlink():raise ValueError('regular objective file required')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config['sha256']:
            raise ValueError('objective file changed; publish a new experiment hash')
        self.options=json.loads(json.dumps(config['options'],allow_nan=False))
        self.identity=dict(api='jev-stage2-objective-v1',sha256=config['sha256'],options=self.options)
        if 'numerics' in config:
            from .grpo_numerics import validate
            self.identity['numerics']=validate(json.loads(json.dumps(config['numerics'],allow_nan=False)))
        scope={'__name__':'jev_custom_stage2_objective','__file__':str(path)}
        exec(compile(raw,str(path),'exec'),scope)
        if (scope.get('API_VERSION')!=self.identity['api']
                or any(not callable(scope.get(k)) for k in ('compute_rewards','compute_loss'))):
            raise ValueError('objective API and both hooks required')
        self.reward_hook=scope['compute_rewards'];self.loss_hook=scope['compute_loss']

    def hybrid_loss(self, logits, records, identities, reference_probability, real_requests, *,
                    global_real_requests, world_size=1, arm='joint', trace_observer=None):
        # Verify the original behavior/action identities regardless of plugin.
        if logits.ndim!=2 or len(records)!=logits.shape[0] or len(identities)!=len(records):
            raise ValueError('complete query records required')
        # A replacement loss may omit baseline_loss entirely. Keep reference
        # admission outside that optional helper, even when KL has weight zero.
        if (reference_probability.shape!=logits.shape or reference_probability.requires_grad
                or reference_probability.device.type!='cpu'):
            raise ValueError('detached CPU reference probabilities matching complete panel required')
        original.log_probabilities(reference_probability)
        for z,record,identity in zip(logits,records,identities,strict=True):
            original.current_terms(z,record,**identity)
        if (real_requests.dtype!=torch.bool or real_requests.shape!=(len(records),)
                or type(world_size)is not int or world_size<1 or arm not in original.ARMS
                or type(global_real_requests)is not int or global_real_requests<max(1,int(real_requests.sum()))):
            raise ValueError('real query mask and global DP denominator required')
        if trace_observer is not None:
            from .objective_trace import inputs,pack,receipt
            before=inputs(logits,records,identities,reference_probability,real_requests,
                global_real_requests=global_real_requests,world_size=world_size,arm=arm)
        def baseline_loss(rewards):
            if not isinstance(rewards,list) or len(rewards)!=len(records):
                raise ValueError('one component reward mapping per query required')
            indexed={id(record):reward for record,reward in zip(records,rewards,strict=True)}
            if len(indexed)!=len(records):raise ValueError('distinct query record objects required')
            def terms(z,record,**identity):
                result=original.current_terms(z,record,**identity)
                reward=indexed[id(record)]
                if set(reward)!=set(result):raise ValueError('complete reward branches required')
                for name,value in reward.items():
                    expected=record['branches'][name]['rewards']
                    if not isinstance(value,torch.Tensor) or value.shape!=expected.shape or value.device!=expected.device:
                        raise ValueError('reward shape/device differs from sampled actions')
                    result[name]['advantages']=continuous_advantages(value,record['binding']['algorithm'])
                return result
            rl=FunctionType(original.rl_loss.__code__,dict(original.rl_loss.__globals__,current_terms=terms))
            rl.__kwdefaults__=original.rl_loss.__kwdefaults__
            hybrid=FunctionType(original.hybrid_loss.__code__,dict(original.hybrid_loss.__globals__,rl_loss=rl))
            hybrid.__kwdefaults__=original.hybrid_loss.__kwdefaults__
            return hybrid(logits,records,identities,reference_probability,real_requests,
                global_real_requests=global_real_requests,world_size=world_size,arm=arm)
        ctx=SimpleNamespace(logits=logits,records=records,identities=identities,
            reference_probability=reference_probability,real_requests=real_requests,
            global_real_requests=global_real_requests,world_size=world_size,arm=arm,
            options=json.loads(json.dumps(self.options)),baseline_loss=baseline_loss)
        # Older sealed runtimes have no numerical extension; preserve their exact API.
        from contextlib import nullcontext
        scope=nullcontext()
        if 'numerics' in self.identity:
            from .grpo_numerics import use
            scope=use(self.identity['numerics'])
        with scope:
            rewards=self.reward_hook(ctx)
            if trace_observer is not None:reward_trace=pack(rewards)
            loss,metrics=self.loss_hook(ctx,rewards)
        if trace_observer is not None:
            if (pack(rewards)!=reward_trace or before!=inputs(logits,records,identities,
                    reference_probability,real_requests,global_real_requests=global_real_requests,
                    world_size=world_size,arm=arm)):
                raise ValueError('objective mutated traced inputs or rewards')
        if (not isinstance(loss,torch.Tensor) or loss.ndim!=0 or loss.device!=logits.device
                or not torch.isfinite(loss) or (logits.requires_grad and not loss.requires_grad)
                or not isinstance(metrics,dict)):
            raise ValueError('finite differentiable scalar loss and metrics required')
        # Existing actor metrics remain stable; new plugin metrics are exported too.
        for key in ('rl','auxiliary','kl'):
            if type(metrics.get(key)) not in (int,float) or not math.isfinite(metrics[key]):
                raise ValueError('finite standard metric required: '+key)
        json.dumps(metrics,allow_nan=False)
        metrics=dict(metrics,objective_sha256=self.identity['sha256'])
        if trace_observer is not None:
            trace_observer(receipt(self.identity,before,reward_trace,loss,metrics))
        return loss,metrics
