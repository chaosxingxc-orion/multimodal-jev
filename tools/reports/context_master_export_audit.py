"""Read-only assembly of original-parameter FSDP shards against BF16 export."""
import argparse
import json
from pathlib import Path
import torch

from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
from multimodal_jev.local_config import load_config
from tools.jev.collect_context_rl_training import sha
from tools.reports.prepare_context_dual_throughput_pair import write


def audit(states,execution,*,world_size=4):
    if type(world_size)!=int or not 1 <= world_size <= 8 or len(states)!=world_size or not execution:
        raise ValueError('complete explicit declared-width shards and execution required')
    names=states[0]['names'];contract=states[0]['contract'];step=states[0]['completed_updates']
    if len(names)!=len(set(names)):raise ValueError('distinct original parameter names required')
    normalized={n:'.'.join(part for part in n.split('.') if part!='_fsdp_wrapped_module') for n in names}
    if len(set(normalized.values()))!=len(names) or set(normalized.values())!=set(execution):
        raise ValueError('complete unambiguous assembled parameter set required')
    scopes=[]
    for rank,state in enumerate(states):
        if (state['rank']!=rank or state['world_size']!=world_size or state['names']!=names or
                set(state['parameters'])!=set(names) or state['contract']!=contract or
                state['completed_updates']!=step):
            raise ValueError('same cursor/contract/parameter order across all ranks required')
        scopes.append(validate_trainable_scope([(n,torch.nn.Parameter(v)) for n,v in state['parameters'].items()]))
    numel=0
    for name,key in normalized.items():
        parts=[s['parameters'][name] for s in states];target=execution[key]
        if any(v.dtype!=torch.float32 or v.device.type!='cpu' or not torch.isfinite(v).all() for v in parts):
            raise ValueError('finite CPU FP32 original-parameter shards required')
        if target.dtype!=torch.bfloat16 or target.device.type!='cpu' or not torch.isfinite(target).all():
            raise ValueError('finite CPU BF16 execution required')
        # FSDP use_orig_params exposes each parameter's unpadded portion on
        # ranks in flat-shard order, including empty views on nonowners.
        joined=torch.cat([v.reshape(-1) for v in parts])
        if joined.numel()!=target.numel() or not torch.equal(joined.bfloat16().reshape(target.shape),target):
            raise ValueError('FP32 master shards do not reproduce BF16 execution: '+key)
        numel+=target.numel()
    return dict(exact_bf16_from_master_shards=True,rank_shards=world_size,parameters=len(names),
        numel=numel,completed_updates=step,checkpoint_scopes=scopes,
        scope='Original-parameter FSDP rank-order reconstruction; not a gradient correctness proof')


def inspect_committed(c):
    root=Path(c['run_root']);step=c['step'];world_size=c.get('world_size',4)
    if type(world_size)!=int or not 1 <= world_size <= 8:raise ValueError('explicit DP width required')
    if type(step)!=int or step<1:raise ValueError('positive checkpoint step required')
    commit_path=root/f'checkpoint-{step:04d}-committed.json'
    if sha(commit_path)!=c['commit_sha256']:raise ValueError('checkpoint commit identity changed')
    commit=json.loads(commit_path.read_text())
    names=[f'rank-{rank}/checkpoint-{step:04d}.pt' for rank in range(world_size)]
    execution=root/f'execution-{step:04d}.pt'
    if (commit['updates']!=step or set(commit['rank_shards'])!=set(names) or
            commit['execution_file']!=execution.name or sha(execution)!=commit['execution_sha256']):
        raise ValueError('complete committed execution and all rank shards required')
    paths=[root/n for n in names]
    for p,n in zip(paths,names):
        if p.is_symlink() or sha(p)!=commit['rank_shards'][n]:raise ValueError('committed shard changed')
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    if any(s['contract']!=commit['contract'] or s['completed_updates']!=step for s in states):
        raise ValueError('master contract/cursor differs from commit')
    result=audit(states,torch.load(execution,map_location='cpu',weights_only=True),world_size=world_size)
    result.update(files={str(p.resolve()):sha(p) for p in [*paths,execution,commit_path]},
        implementation_sha256=sha(__file__),scope='Actual committed FP32 shards reproduce BF16 execution; no gradient, speed or efficacy claim')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);torch.set_num_threads(1)
    result=inspect_committed(c);result['config_provenance']=provenance;write(c['output'],result)
    print(json.dumps({k:result[k] for k in ('exact_bf16_from_master_shards','parameters','numel','completed_updates')}))
