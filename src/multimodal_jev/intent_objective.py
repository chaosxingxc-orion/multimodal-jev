"""Versioned objective identity, separate from inference policy identity."""
import hashlib
import json
import math


def specification(config):
    value=config.get('objective',{'name':'correctness-v1'})
    if not isinstance(value,dict) or 'name' not in value:
        raise ValueError('explicit supported objective specification required')
    name=value['name']
    if name=='correctness-v1' and set(value)=={'name'}:
        return dict(name=name)
    if name=='smoothed-utility-v1' and set(value)=={'name'}:
        if config.get('arm')!='D':raise ValueError('utility requires single Final D arm')
        return dict(name=name,tau='1/K^2',scale='U_prime(p_old)',
                    normalization='original-effective-request-weight',version=1)
    if name=='uniform-credit-v1' and set(value)=={'name','scale'}:
        scale=value['scale']
        if config.get('arm')!='D':raise ValueError('uniform control requires single Final D arm')
        if type(scale) not in (int,float) or not math.isfinite(scale) or not 0<scale<=1000:
            raise ValueError('finite fixed positive scale at most 1000 required')
        return dict(name=name,scale=float(scale),
                    normalization='original-effective-request-weight',version=1)
    raise ValueError('unknown objective or unregistered objective options')


def identity(config):
    spec=specification(config)
    encoded=json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(specification=spec,sha256=hashlib.sha256(encoded).hexdigest())


def apply(config,items):
    objective=identity(config);spec=objective['specification']
    if spec['name']=='correctness-v1':
        return items,dict(objective=objective,credit=None)
    from .span_smoothed_utility import transform
    transformed,credit=transform(items)
    if spec['name']=='uniform-credit-v1':
        # Reuse the strict single-Final admission, but apply the fixed control
        # scale, never a batch-derived scale selected from target TEST outcomes.
        transformed=[dict(row,span_weights=(row['span_weights']*spec['scale']).detach()) for row in items]
        credit=dict(schema='jev-final-uniform-credit-v1',scale=spec['scale'],
                    request_ids=[row['request_id'] for row in items])
    return transformed,dict(objective=objective,credit=credit)


def supervision_records(items):
    return [dict(request_id=row['request_id'],targets=row['targets'].tolist(),
        supervised=row['supervised'].tolist(),span_weights=row['span_weights'].tolist(),
        request_weight=row['request_weight']) for row in items]


def verify_credit(config,selected,behavior,receipt,supervision):
    """Reconstruct from independent old probabilities and released supervision."""
    import torch
    old_rows={row['id']:row for row in behavior['records']}
    if len(old_rows)!=len(behavior['records']):raise ValueError('duplicate behavior request')
    wire='D' if config['arm']=='D' else 'Q'
    raw=[]
    for item in selected:
        row=old_rows[item['id']+'::'+wire];target=item['target']
        raw.append(dict(request_id=row['request_id'],
            behavior=tuple(torch.tensor([p],dtype=torch.float32) for p in row['candidate_logprobs']),
            targets=torch.tensor([target['targets']],dtype=torch.long),
            supervised=torch.tensor([target['supervised']],dtype=torch.bool),
            span_weights=torch.tensor([target['span_weights']],dtype=torch.float32),
            request_weight=target['request_weight']))
    transformed,expected=apply(config,raw)
    if receipt!=expected:raise ValueError('objective credit receipt differs')
    if supervision!=supervision_records(transformed):raise ValueError('applied objective weights differ')
