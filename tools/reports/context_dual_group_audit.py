"""Recompute saved dual credits independently from scheduled truth and wire.

No model calls. This proves consistency of recorded evidence; actual execution
and checkpoint/runtime identity must be checked by the enclosing run audit.
"""
import hashlib
import json

import torch

from multimodal_jev.context_dual_credit import ALGORITHMS
from multimodal_jev.context_logits_wire import validate_logits_record


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor(actual,expected,name):
    if (not isinstance(actual,torch.Tensor) or actual.device.type!='cpu' or actual.requires_grad or
            actual.dtype!=expected.dtype or actual.shape!=expected.shape or not torch.equal(actual,expected)):
        raise ValueError('dual audit tensor differs: '+name)


def audit(groups,receipts,targets,*,candidates,algorithm,arm,update,policy_id,reference_policy_id):
    if algorithm not in ALGORITHMS or arm not in ('joint','ranking_only','existence_only'):
        raise ValueError('registered dual objective required')
    ids=[g['request_id'] for g in groups];rids=[r['request_id'] for r in receipts]
    if (not ids or len(ids)!=len(set(ids)) or len(rids)!=len(set(rids)) or
            set(ids)!=set(rids) or set(ids)!=set(targets) or len(set(candidates))!=len(candidates)):
        raise ValueError('complete distinct scheduled population required')
    entries={r['request_id']:r for r in receipts};real=sum(t['real_request'] for t in targets.values())
    if real<1:raise ValueError('nonempty real population required')
    statistics={name:[] for name in ('ranking','existence')}
    options=dict(algorithm=algorithm,arm=arm,ranking_coefficient=.1,probability_coefficient=.3,
                 kl_coefficient=.01,reference_policy_id=reference_policy_id)
    for g in groups:
        uid=g['request_id'];t=targets[uid];entry=entries[uid];y=torch.tensor(t['labels'],dtype=torch.bool)
        if y.shape!=(len(candidates),):raise ValueError('complete target panel required')
        tensor(g['labels'],y[None],'labels');tensor(g['observed'],torch.ones_like(y)[None],'observed')
        if (g['policy_id']!=policy_id or g['candidate_ids']!=list(candidates) or
                type(g['real_request'])!=bool or g['real_request'] is not t['real_request'] or
                g['request_weight']!=t['weight'] or t['weight']!=float(t['real_request']) or
                g['global_rows']!=len(groups) or g['global_real_requests']!=real or
                g['dual_options']!=options or g['reference_policy_id']!=reference_policy_id):
            raise ValueError('group truth, denominator or objective differs')
        record=validate_logits_record(entry['behavior_record']);reference=entry['reference_record']
        for raw,pid in ((record,policy_id),(reference,reference_policy_id)):
            ident=raw['request_identity']
            if raw['policy_id']!=pid or ident['policy_id']!=pid or raw['request_id']!=digest(ident):
                raise ValueError('wire input/policy identity differs')
        a,b=record['request_identity'],reference['request_identity']
        if a['inputs']!=b['inputs'] or a['positions']!=b['positions'] or len(a['positions'])!=len(candidates):
            raise ValueError('reference and behavior must consume identical input')
        ref=torch.tensor(reference['prob'],dtype=torch.float32)
        if ref.shape!=y.shape or not torch.isfinite(ref).all() or not ((ref>0)&(ref<1)).all():
            raise ValueError('finite interior fixed reference required')
        tensor(g['reference_probability'],ref,'reference_probability')
        identity=dict(policy_id=policy_id,input_id=record['request_id'],query_id=uid,update=update)
        credit=g['dual_credit'];arithmetic=credit['binding'].get('arithmetic')
        if arithmetic not in ('dual-credit-cpu-pair-fp32-v1','dual-credit-cpu-pair-fp64-v2'):
            raise ValueError('unregistered credit arithmetic')
        pair_dtype=torch.float64 if arithmetic.endswith('fp64-v2') else torch.float32
        binding=dict(identity,algorithm=algorithm,group_size=16,seed=317,
            candidates=list(candidates),labels=t['labels'],logits=record['logits'],probability=record['prob'],
            arithmetic=arithmetic)
        if (g['dual_identity']!=identity or credit['binding']!=binding or
                credit['binding_sha256']!=digest(binding) or entry['credit_binding_sha256']!=digest(binding) or
                set(credit['branches'])!={'ranking','existence'}):
            raise ValueError('credit not bound to actual recorded behavior and scheduled truth')
        # Independent pair enumeration: no production prepare/components/
        # advantages/verify calls. Both draw streams remain explicitly G16.
        z=torch.tensor(record['logits'],dtype=torch.float32);p=torch.tensor(record['prob'],dtype=torch.float32)
        ordered=sorted(range(len(candidates)),key=lambda i:candidates[i])
        pairs=[(i,j) for at,i in enumerate(ordered) for j in ordered[at+1:] if y[i]!=y[j]]
        positive=int(y.sum());negative=len(y)-positive
        for name in ('ranking','existence'):
            rows=pairs if name=='ranking' else [(i,) for i in ordered]
            keys=tuple(tuple(candidates[i] for i in row) for row in rows)
            # CPU scalar and vector sigmoid kernels can differ by one ULP.
            # Preserve the protocol's complete FP64 ranking operation without
            # relaxing exact probability/draw checks or reusing production code.
            if name=='ranking':
                differences=torch.stack([z[i].to(pair_dtype)-z[j].to(pair_dtype) for i,j in rows]) if rows else torch.empty(0,dtype=pair_dtype)
                probability=differences.sigmoid()
            else:
                probability=torch.stack([p[row[0]] for row in rows])
            target=torch.tensor([bool(y[row[0]]) for row in rows],dtype=torch.bool)
            if name=='ranking':weight=torch.full((len(rows),),1./len(rows)) if rows else torch.empty(0)
            else:weight=torch.tensor([(.5/(positive if y[row[0]] else negative)) if positive and negative else 1./len(y) for row in rows])
            if not ((probability>0)&(probability<1)).all():raise ValueError('saturated component probability')
            actions=[]
            for key,q in zip(keys,probability):
                seed=int(digest([317,update,uid,name,key])[:16],16)%(2**63)
                actions.append((torch.rand(16,generator=torch.Generator().manual_seed(seed),dtype=probability.dtype)<q).long())
            actions=torch.stack(actions) if rows else torch.empty((0,16),dtype=torch.long)
            reward=(actions==target[:,None]).float()
            if algorithm==ALGORITHMS[0]:
                adv=torch.stack([reward[:,i]-torch.cat((reward[:,:i],reward[:,i+1:]),1).mean(1) for i in range(16)],1)
            else:
                centered=reward-reward.mean(1,keepdim=True)
                adv=centered/(centered.square().mean(1,keepdim=True).sqrt().clamp_min(.25))
            old=torch.stack((torch.log1p(-probability),torch.log(probability)),-1).gather(1,actions)
            expected=dict(probability=probability,target=target,weight=weight,actions=actions,rewards=reward,old_logprob=old)
            branch=credit['branches'][name]
            if set(branch)!=set(expected)|{'keys','advantages'} or branch['keys']!=keys:
                raise ValueError('component keys or fields differ')
            for field,v in expected.items():tensor(branch[field],v,name+'/'+field)
            actual=branch['advantages']
            if (actual.dtype!=torch.float32 or actual.shape!=adv.shape or actual.device.type!='cpu' or
                    actual.requires_grad or not torch.allclose(actual,adv,atol=2e-7,rtol=0)):
                raise ValueError('independently recomputed advantage differs')
            if t['real_request']:
                statistics[name].append(dict(reward=float((weight*reward.mean(1)).sum()),
                    zero_components=int((reward==reward[:,:1]).all(1).sum()),components=len(rows)))
    return dict(algorithm=algorithm,arm=arm,update=update,checked_slots=len(groups),real_queries=real,
        padding_slots=len(groups)-real,branches={name:dict(
            mean_weighted_reward=sum(v['reward'] for v in rows)/real,
            zero_signal_components=sum(v['zero_components'] for v in rows),
            components=sum(v['components'] for v in rows)) for name,rows in statistics.items()},
        scope='Independent recorded component/sample/reward/advantage audit; not model execution proof')
