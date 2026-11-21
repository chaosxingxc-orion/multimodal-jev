"""Two pinned, label-free TP1 workers for one-macro joint TPO references."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.model_provider import resolve_provider
from multimodal_jev.context_fsdp_export import load_assembled_execution
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.span_rollout_rpc import receive, send


def read(path):
    return json.loads(Path(path).read_bytes())


def path_for(root, replica):
    return root/f'reference-{replica}.sock'


def worker(args):
    root=args.output
    cfg=read(args.parent)
    plan=read(args.plan)
    source=Path(cfg['source_policy'])
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    provider = resolve_provider(cfg)
    if (file_hash(args.parent)!=plan['parent_sha'] or
        file_hash(source/'MANIFEST.json')!=plan['source_manifest_sha256'] or
        checkpoint_identity(cfg['model'])[0]!=cfg['base_id']):
        raise ValueError('worker source changed')
    snapshot=read_execution_bundle(source,expected_policy_id=cfg['source_policy_id'],
        expected_contract=cfg['source_contract'],expected_base_id=cfg['base_id'])
    devices=plan['training_devices']+plan['inference_devices']
    card=plan['inference_devices'][args.replica]
    local=root/f'reference-{args.replica}'
    local.mkdir()
    app=dict(cfg,snapshot=str(source),policy_id=cfg['source_policy_id'],contract=cfg['source_contract'],
        physical_devices=[card],
        max_model_len=cfg['view_execution']['max_input_tokens'],process_device_scope=dict(
        kind='same_container_runtime_visibility_v1',mounted_physical=devices,visible_ordinal=str(devices.index(card))))
    torch.use_deterministic_algorithms(True)
    model,device,base,handle=provider.load_reference(app,snapshot,local)
    p = provider.processor(snapshot['processor'])
    collate = provider.collation(base,p,model)
    world=len(plan['training_devices']); replicas=len(plan['inference_devices'])
    version=0
    address=path_for(root,args.replica)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(address)); server.listen(world)
        commit_json(local/'ready.json',dict(card=card,source_policy=plan['source_policy'],pid=os.getpid()))
        for step in range(args.steps):
            expected=set(range(args.replica,world,replicas))
            policy=plan['source_policy']
            if step:
                policy=None
            while expected:
                conn,_=server.accept()
                with conn:
                    request=receive(conn)
                    if request==dict(operation='close',sequence=0):
                        send(conn,dict(sequence=0,ok=True,result=None))
                        return
                    if policy is None:
                        update=root/f'activation-{step:04d}.pt'
                        identity=read(root/f'activation-{step:04d}.json')
                        if identity['update']!=step or file_hash(update)!=identity['sha256']:
                            raise ValueError('incorrect recent actor execution')
                        load_assembled_execution(model,torch.load(update,map_location='cpu',weights_only=True))
                        policy=identity['sha256']; version=step
                    rank=request['rank']
                    if rank not in expected or request['step']!=step or request['version']!=version:
                        raise ValueError('stale or duplicate reference request')
                    path=root/f'rank-{rank}'/f'ref-input-{step:04d}.pt'
                    if file_hash(path)!=request['sha256']:
                        raise ValueError('reference input content changed')
                    rows=torch.load(path,map_location='cpu',weights_only=True)
                    if len(rows)!=cfg['batch_config']['per_dp_replica_batch']*2:
                        raise ValueError('need both complete views')
                    micro=plan['microbatch']
                    values=[]
                    started=time.monotonic()
                    with torch.inference_mode():
                        for start in range(0,len(rows),micro):
                            chunk=rows[start:start+micro]
                            if any(set(r)!={'request_id','inputs','positions'} for r in chunk):
                                raise ValueError('only label-free inputs permitted')
                            z=provider.forward(model,[dict(r,policy_id=policy) for r in chunk],
                                device=device,dtype=torch.bfloat16,policy_id=policy,**collate).float().cpu()
                            if not torch.isfinite(z).all(): raise ValueError('invalid reference logits')
                            values.extend(z.tolist())
                    expected.remove(rank)
                    send(conn,dict(rank=rank,step=step,version=version,policy=policy,card=card,values=values,
                                   seconds=time.monotonic()-started))
        # Keep the workers available until the exact actor process has finished validation/export.
        conn,_=server.accept()
        with conn:
            if receive(conn)!=dict(operation='close',sequence=0): raise ValueError('expected close request')
            send(conn,dict(sequence=0,ok=True,result=None))


def score(root, rank_dir, items, *, rank, step, version, policy, devices, timeout):
    replica=rank%len(devices)
    path=rank_dir/f'ref-input-{step:04d}.pt'
    payload=[{k:r[k] for k in ('request_id','inputs','positions')} for r in items]
    commit_file(path,lambda f: torch.save(payload,f))
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(path_for(root,replica)))
            send(conn,dict(rank=rank,step=step,version=version,sha256=file_hash(path)))
            response=receive(conn)
        if (any(response[k]!=v for k,v in dict(rank=rank,step=step,version=version,policy=policy,card=devices[replica]).items())
                or len(response['values'])!=len(items)):
            raise ValueError('reference reply belongs to another macro or policy')
        return [torch.tensor(v,dtype=torch.float32) for v in response['values']],response['seconds']
    finally:
        path.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parent','plan','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--replica',type=int,required=True)
    p.add_argument('--steps',type=int,required=True)
    worker(p.parse_args())
