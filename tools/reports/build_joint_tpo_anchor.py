"""Produce immutable fixed logits for paired requests using the common initial policy."""
import argparse
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.model_provider import resolve_provider
from multimodal_jev.context_choice_inputs import identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.tpo_anchor_layout import part_name, partition
from tools.reports.build_tpo_view import read, verified
from tools.reports.joint_tpo_inputs import load, stream


def run(args):
    plan=read(args.plan); data=args.plan.parent/'targets'
    c,m,task,rows,pop,batches=load(args.parent,data,plan)
    torch.use_deterministic_algorithms(True)
    if checkpoint_identity(c['model'])[0]!=c['base_id']: raise ValueError('anchor model identity differs')
    bundle=read_execution_bundle(c['source_policy'],expected_policy_id=plan['source_policy'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    provider = resolve_provider(c)
    processor = provider.processor(bundle['processor'])
    root=args.output
    root.mkdir()
    devices=plan.get('offline_reference_devices', plan['inference_devices'])
    card=devices[args.replica]
    app=dict(c,snapshot=c['source_policy'],policy_id=c['source_policy_id'],contract=c['source_contract'],
        physical_devices=[card],
        max_model_len=c['view_execution']['max_input_tokens'],process_device_scope=dict(
            kind='same_container_runtime_visibility_v1',mounted_physical=devices,visible_ordinal=str(args.replica)))
    model,device,base,handle=provider.load_reference(app,bundle,root)
    collate=provider.collation(base,processor,model)
    fresh = plan.get('initialization') == 'pre_task_policy'
    old = None if fresh else plan['old_choice_anchor']
    if not fresh and read(Path('/jev-control/reference/receipt.json')) != old:
        raise ValueError('Choice anchor control changed')
    native = {} if fresh else read(data/'native-probs.json')
    @torch.inference_mode()
    def forward(items):
        return provider.forward(model,[{k:x[k] for k in
            ('request_id','inputs','positions','policy_id')} for x in items],
            device=device,dtype=torch.bfloat16,policy_id=source,**collate).float().cpu()
    source=plan['source_policy']
    micro=plan['microbatch']
    result={}
    stop = args.steps if args.steps is not None else len(batches)
    if not 0 < stop <= len(batches): raise ValueError('invalid fixed anchor window')
    world = len(plan['training_devices'])
    split = 'offline_reference_devices' in plan
    parts = (partition(world, stop, len(devices))[args.replica] if split else
             [(rank, 0, stop) for rank in range(args.replica, world, len(devices))])
    for rank, first, end in parts:
        historic=Path(f'/jev-control/reference/rank-{rank}/choice.pt')
        by_input={} if fresh else torch.load(verified(historic,old['rank_files'][rank]),map_location='cpu',weights_only=True)
        anchor=[]
        with stream(c,task,rows,pop,batches,processor,collate,provider=provider,rank=rank,start=first,stop=end) as consume:
            for step in range(first, end):
                b = batches[step]
                n,choice,meta,slots=consume(step,source)
                views=[]
                for start in range(0,len(choice),micro):
                    group=choice[start:start+micro]
                    keys=[identity(x,source) for x in group]
                    if fresh:
                        batch_ids=[r['id'] for r in meta[start:start+micro]]
                        if any(uid not in native for uid in batch_ids):
                            nz=forward(n[start:start+micro])
                            if not torch.isfinite(nz).all(): raise ValueError('invalid native anchor')
                            for uid, z0 in zip(batch_ids,nz,strict=True): native[uid]=z0.clone()
                        v=forward(group)
                    elif b['prior_mode']=='open':
                        v=torch.stack([by_input[k] for k in keys])
                    else:
                        with torch.inference_mode():
                            v=provider.forward(model,[{k:x[k] for k in
                                ('request_id','inputs','positions','policy_id')} for x in group],
                                device=device,dtype=torch.bfloat16,policy_id=source,**collate).float().cpu()
                    if not torch.isfinite(v).all(): raise ValueError('nonfinite fixed Choice')
                    for i,(k,z) in enumerate(zip(keys,v.unbind(),strict=True),start):
                        uid=meta[i]['id']
                        prob=native[uid] if fresh else torch.tensor(native[uid],dtype=torch.float32)
                        if z.numel()!=choice[i]['positions'].numel():
                            raise ValueError('fixed Choice geometry differs')
                        views.append((uid,rows[uid]['request_key'],k,prob,z.clone()))
                anchor.append(views)
        key = part_name(rank, first, end) if split else str(rank)
        path = root / (key if split else f'rank-{rank}.pt')
        commit_file(path,lambda f:torch.save(anchor,f))
        result[key]=file_hash(path)
    files = dict(parts=result) if split else dict(rank_files=result)
    commit_json(root/'receipt.json',dict(**files,source_policy=source,target_sha=plan['target_sha'],joint_sha=file_hash(args.plan),card=card,updates=stop))


def main():
    p=argparse.ArgumentParser()
    for key in ('parent','plan','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--replica',type=int,required=True)
    p.add_argument('--steps',type=int)
    run(p.parse_args())

if __name__=='__main__': main()
