#!/usr/bin/env python3
"""CPU-only post-hoc Adam direction diagnostic from authenticated FP32 shards.

The clipped gradient is reconstructed from first moments, with FP32 rounding
error; it is not an independently recorded backward gradient or a causal proof.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json, file_hash


def diagnose(previous, current):
    if (previous['rank'] != current['rank'] or previous['world_size'] != current['world_size']
            or current['updates'] != previous['updates']+1 or previous['names'] != current['names']
            or previous['optimizer']['param_groups'] != current['optimizer']['param_groups']):
        raise ValueError('consecutive identical-topology optimizer states required')
    groups=current['optimizer']['param_groups']
    if len(groups)!=1:raise ValueError('one AdamW parameter group required')
    group=groups[0];b1,b2=group['betas'];step=current['updates']
    if group['weight_decay']!=0 or group.get('amsgrad') or group.get('maximize'):
        raise ValueError('unmodified zero-decay AdamW descent required')
    keys=group['params'];names=current['names']
    if len(keys)!=len(names) or set(previous['parameters'])!=set(names) or set(current['parameters'])!=set(names):
        raise ValueError('complete parameter mapping required')
    stats=dict(elements=0,gradient_squared_norm=0.,delta_squared_norm=0.,gradient_dot_delta=0.,
        current_component_dot_gradient=0.,history_component_dot_gradient=0.,
        maximum_second_moment_residual=0.,maximum_delta_reconstruction_error=0.,
        bf16_changed_elements=0,fp32_changed_elements=0)
    for name,key in zip(names,keys,strict=True):
        p,q=previous['parameters'][name],current['parameters'][name]
        a,b=previous['optimizer']['state'][key],current['optimizer']['state'][key]
        if a['step'].item()!=step-1 or b['step'].item()!=step:raise ValueError('Adam cursor differs')
        tensors=[p,q,a['exp_avg'],b['exp_avg'],a['exp_avg_sq'],b['exp_avg_sq']]
        if any(v.dtype!=torch.float32 or v.shape!=p.shape or not torch.isfinite(v).all() for v in tensors):
            raise ValueError('finite aligned FP32 state required')
        if not p.numel():continue
        p,q,m0,m1,v0,v1=[v.double() for v in tensors]
        gradient=(m1-b1*m0)/(1-b1)
        denom=v1.sqrt()/(1-b2**step)**.5+group['eps']
        scale=-group['lr']/(1-b1**step)
        history=scale*b1*m0/denom;fresh=scale*(1-b1)*gradient/denom
        delta=q-p;predicted=history+fresh
        stats['elements']+=p.numel()
        stats['gradient_squared_norm']+=gradient.square().sum().item()
        stats['delta_squared_norm']+=delta.square().sum().item()
        stats['gradient_dot_delta']+=(gradient*delta).sum().item()
        stats['current_component_dot_gradient']+=(gradient*fresh).sum().item()
        stats['history_component_dot_gradient']+=(gradient*history).sum().item()
        stats['maximum_second_moment_residual']=max(stats['maximum_second_moment_residual'],
            (v1-(b2*v0+(1-b2)*gradient.square())).abs().max().item())
        stats['maximum_delta_reconstruction_error']=max(stats['maximum_delta_reconstruction_error'],(delta-predicted).abs().max().item())
        stats['bf16_changed_elements']+=int((p.bfloat16()!=q.bfloat16()).sum())
        stats['fp32_changed_elements']+=int((p!=q).sum())
    return stats


def summarize(ranks):
    result={k:(max(r[k] for r in ranks) if k.startswith('maximum_') else sum(r[k] for r in ranks)) for k in ranks[0]}
    denominator=(result['gradient_squared_norm']*result['delta_squared_norm'])**.5
    result['cosine_gradient_delta']=result['gradient_dot_delta']/denominator if denominator else None
    result['first_order_descent']=result['gradient_dot_delta']<0
    return result


def selftest():
    p=torch.nn.Parameter(torch.tensor([.01,-.02,.03]));opt=torch.optim.AdamW([p],lr=1e-4,weight_decay=0.,foreach=False,fused=False)
    def state(step):return dict(rank=0,world_size=1,updates=step,names=['p'],parameters={'p':p.detach().clone()},optimizer=copy.deepcopy(opt.state_dict()))
    p.grad=torch.tensor([1.,-1.,1.]);opt.step();before=state(1)
    actual=torch.tensor([-.01,.01,-.01]);p.grad=actual;opt.step();after=state(2)
    result=diagnose(before,after)
    direct=float((actual*(after['parameters']['p']-before['parameters']['p'])).double().sum())
    assert abs(result['gradient_dot_delta']-direct)<1e-10
    assert result['gradient_dot_delta']>0 and result['current_component_dot_gradient']<0
    assert result['history_component_dot_gradient']>0
    assert result['maximum_delta_reconstruction_error']<3e-9
    bad=copy.deepcopy(after);bad['updates']=4
    try:diagnose(before,bad)
    except ValueError:pass
    else:raise AssertionError('nonconsecutive states accepted')
    return dict(passed=True,momentum_opposition=result)


def run(config):
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    torch.set_num_threads(2)
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('diagnostic state changed')
    test=selftest();reports=[]
    for entry in config['updates']:
        previous,current=Path(entry['previous']),Path(entry['current']);ranks=[]
        for rank in range(4):
            paths=[r/'run'/f'rank-{rank}'/'optimizer-local.pt' for r in (previous,current)]
            for p in paths:
                if config['pins'].get(str(p))!=file_hash(p):raise ValueError('all shard states must be pinned')
            a,b=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
            if a['rank']!=rank or a['world_size']!=4 or b['updates']!=entry['update']:raise ValueError('fixed DP4 cursor required')
            ranks.append(diagnose(a,b))
        reports.append(dict(update=entry['update'],ranks=ranks,global_statistics=summarize(ranks)))
    return dict(passed=True,selftest=test,reports=reports,scope='Post-hoc reconstructed clipped gradient and actual FP32 master displacement. '
        'FP32 moments induce reconstruction error. First-order loss direction does not guarantee finite-step BF16 reward, held-out gain or identify a unique cause.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text())))
        atomic_json(args.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
