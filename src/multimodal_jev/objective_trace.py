"""Finite-admission objective traces and deterministic CPU replay.

Records the small decision tensors, never audio features or base weights.
Replay authenticates formula evaluation only; transport and model execution
must still be proved by the enclosing independent training auditor.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

DTYPES={str(t):t for t in (torch.bool,torch.int32,torch.int64,torch.float32,
                           torch.float64,torch.bfloat16)}


def pack(value):
    # Tagged containers prevent arbitrary plugin dicts colliding with tensors.
    if isinstance(value,torch.Tensor):
        if str(value.dtype) not in DTYPES or not torch.isfinite(value).all():
            raise ValueError('finite supported decision tensor required')
        return ['tensor',str(value.dtype),list(value.shape),value.detach().cpu().tolist()]
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('string trace keys required')
        return ['dict',[[k,pack(v)] for k,v in sorted(value.items())]]
    if isinstance(value,(list,tuple)):
        return ['tuple' if isinstance(value,tuple) else 'list',[pack(v) for v in value]]
    if value is None or type(value) in (str,bool,int):return ['scalar',value]
    if type(value) is float and math.isfinite(value):return ['scalar',value]
    raise ValueError('objective trace requires finite tensors and JSON values')


def unpack(value):
    kind=value[0]
    if kind=='tensor':
        _,dtype,shape,data=value
        return torch.tensor(data,dtype=DTYPES[dtype]).reshape(shape)
    if kind=='dict':return {k:unpack(v) for k,v in value[1]}
    if kind=='list':return [unpack(v) for v in value[1]]
    if kind=='tuple':return tuple(unpack(v) for v in value[1])
    if kind=='scalar':return value[1]
    raise ValueError('unknown trace type')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def inputs(logits,records,identities,reference_probability,real_requests,**kwargs):
    return pack(dict(logits=logits,records=records,identities=identities,
        reference_probability=reference_probability,real_requests=real_requests,**kwargs))


def receipt(identity,before,rewards,loss,metrics):
    body=dict(schema='jev-objective-trace-v1',objective=pack(identity),inputs=before,
              rewards=rewards,loss=float(loss.detach()),metrics=pack(metrics))
    return dict(body,sha256=digest(body))


class FiniteTraceWriter:
    """Buffer microbatches in memory; commit once per rank/update after backward."""
    def __init__(self,root):
        from .local_runtime_storage import check_paths
        self.root=Path(root);check_paths({'objective_trace':self.root})
        self.rows=[];self.update=None

    def begin(self,update):
        if self.update is not None:raise ValueError('uncommitted objective trace update')
        self.update=update;self.rows=[]

    def __call__(self,row):
        if self.update is None:raise ValueError('objective trace outside update')
        actual=unpack(row['inputs'])['identities']
        if any(v['update']!=self.update for v in actual):
            raise ValueError('trace belongs to another update')
        self.rows.append(row)

    def finish(self):
        from .local_checkpoint import commit_json
        if self.update is None or not self.rows:raise ValueError('missing objective traces')
        self.root.mkdir(parents=True,exist_ok=True)
        target=self.root/f'objective-{self.update:04d}.json'
        commit_json(target,dict(schema='jev-objective-update-traces-v1',update=self.update,rows=self.rows))
        self.update=None;self.rows=[]
        return str(target)


def replay(row,plugin,*,loss_atol=0.,replay_device='cpu'):
    """Exact reward replay, explicit loss tolerance; not an optimizer audit."""
    if type(loss_atol) not in (int,float) or not math.isfinite(loss_atol) or loss_atol<0:
        raise ValueError('finite nonnegative loss tolerance required')
    body={k:v for k,v in row.items() if k!='sha256'}
    if row.get('schema')!='jev-objective-trace-v1' or digest(body)!=row.get('sha256'):
        raise ValueError('objective trace digest differs')
    if row['objective']!=pack(plugin.identity):raise ValueError('objective identity differs')
    args=unpack(row['inputs'])
    # Replay sigmoid and auxiliary loss on the actor backend. Behavior and
    # reference tensors remain on CPU, exactly as in actual training.
    args['logits']=args['logits'].to(replay_device).requires_grad_()
    captured=[]
    loss,metrics=plugin.hybrid_loss(**args,trace_observer=captured.append)
    if len(captured)!=1 or captured[0]['rewards']!=row['rewards']:
        raise ValueError('actual custom rewards differ on replay')
    if abs(float(loss.detach())-row['loss'])>loss_atol:
        raise ValueError('custom loss differs on replay')
    if pack(metrics)!=row['metrics']:
        raise ValueError('objective metrics differ on replay')
    return dict(status='objective_formula_replayed',objective_sha256=plugin.identity['sha256'],
                loss_atol=loss_atol,model_execution_verified=False,optimizer_verified=False)


def audit_update(documents,groups_by_rank,plugin,*,update,microbatch,loss_atol=0.,replay_device='cpu'):
    """Bind traces to separately authenticated scheduled actor payloads.

    Caller must first audit groups against input/behavior/reference journals.
    Never derive expected coverage from the traces themselves.
    """
    if (type(update) is not int or update<0 or type(microbatch) is not int or microbatch<1
            or not groups_by_rank or len(documents)!=len(groups_by_rank)):
        raise ValueError('explicit complete rank population required')
    world=len(groups_by_rank)
    groups=[g for rank in groups_by_rank for g in rank]
    ids=[g['dual_identity']['query_id'] for g in groups]
    real=sum(g['real_request'] for g in groups)
    if (not real or len(ids)!=len(set(ids)) or any(not rank or len(rank)%microbatch for rank in groups_by_rank)):
        raise ValueError('distinct complete scheduled queries and microbatches required')
    count=0
    for doc,rank_groups in zip(documents,groups_by_rank,strict=True):
        if (set(doc)!={'schema','update','rows'} or doc['schema']!='jev-objective-update-traces-v1'
                or doc['update']!=update or len(doc['rows'])!=len(rank_groups)//microbatch):
            raise ValueError('complete rank/update trace population required')
        for start,row in zip(range(0,len(rank_groups),microbatch),doc['rows'],strict=True):
            args=unpack(row['inputs']);expected=rank_groups[start:start+microbatch]
            if (set(args)!={'logits','records','identities','reference_probability','real_requests',
                            'global_real_requests','world_size','arm'}
                    or args['identities']!=[g['dual_identity'] for g in expected]
                    or any(v['update']!=update for v in args['identities'])
                    or pack(args['records'])!=pack([g['dual_credit'] for g in expected])
                    or pack(args['reference_probability'])!=pack(torch.stack([g['reference_probability'] for g in expected]))
                    or pack(args['real_requests'])!=pack(torch.tensor([g['real_request'] for g in expected],dtype=torch.bool))
                    or args['global_real_requests']!=real or args['world_size']!=world
                    or any(args['arm']!=g['dual_options']['arm'] for g in expected)):
                raise ValueError('trace differs from authenticated rank population or reference')
            replay(row,plugin,loss_atol=loss_atol,replay_device=replay_device);count+=1
    return dict(status='objective_update_formulas_replayed',update=update,ranks=world,
        queries=len(groups),real_queries=real,microbatches=count,
        objective_sha256=plugin.identity['sha256'],model_execution_verified=False,
        optimizer_verified=False)
