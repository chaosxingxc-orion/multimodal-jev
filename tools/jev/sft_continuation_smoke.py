#!/usr/bin/env python3
"""Check shared-initializer logits and a real supervised continuation update.

One fixed TRAIN example, no inference/effectiveness score. The actual sft.py
initializer is exercised; a full matched-budget SFT comparison remains separate.
"""
import argparse
import gc
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    import torch
    from multimodal_jev.sft import initialization_identity,initialize_adapter
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import teacher_forcing,load_audio,response_log_probs
    from multimodal_jev.task_schema import JevExample
    config=json.loads(args.config.read_text())
    identity=initialization_identity(config)
    if identity is None:raise ValueError('shared initializer required')
    rows=[JevExample.from_record(row) for line in Path(config['train_manifest']).open()
          if (row:=json.loads(line))['example_id']==config['example_id']]
    if len(rows)!=1 or rows[0].split!='train':raise ValueError('one fixed TRAIN example required')
    ex=rows[0]
    if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('audio changed')
    atomic_json(args.output/'identity.json',{'config':config,'initialization':identity,
        'manifest_sha256':file_hash(config['train_manifest']),'audio_sha256':file_hash(ex.audio)})
    torch.set_num_threads(4);torch.manual_seed(17)
    wave=load_audio(ex.audio)
    processor,merged=load_runtime(config['initial_checkpoint'],'cpu','float32')
    batch,boundary=teacher_forcing(processor,ex,wave)
    atomic_json(args.output/'progress.json',{'phase':'merged_reference','heartbeat_unix':time.time()})
    with torch.inference_mode():
        reference=response_log_probs(merged(**batch,use_cache=False).logits,batch['input_ids'],boundary).clone()
    del merged;gc.collect()
    actor_processor,base=load_runtime(config['model_path'],'cpu','float32')
    actor_batch,actor_boundary=teacher_forcing(actor_processor,ex,wave)
    assert boundary==actor_boundary and batch.keys()==actor_batch.keys()
    assert all(torch.equal(batch[k],actor_batch[k]) for k in batch)
    model=initialize_adapter(base,config).eval()
    params=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    assert params and all('lora_' in name for name,p in params)
    before={name:p.detach().clone() for name,p in params}
    optimizer=torch.optim.AdamW([p for _,p in params],lr=config['lr'],weight_decay=0.,foreach=False,fused=False)
    assert not optimizer.state
    atomic_json(args.output/'progress.json',{'phase':'adapter_forward','heartbeat_unix':time.time()})
    output=model(**actor_batch,use_cache=False)
    actual=response_log_probs(output.logits,actor_batch['input_ids'],boundary)
    torch.testing.assert_close(actual.detach(),reference,atol=3e-4,rtol=1e-5)
    error=float((actual.detach()-reference).abs().max())
    loss=output.loss
    assert torch.isfinite(loss)
    atomic_json(args.output/'progress.json',{'phase':'backward','initial_logprob_error':error,'heartbeat_unix':time.time()})
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
    optimizer.step()
    changed=sum(not torch.equal(p,before[name]) for name,p in params)
    assert changed>0 and optimizer.state and float(norm)>0
    model.save_pretrained(args.output/'adapter')
    assert initialization_identity(config)==identity,'initializer changed during update'
    return {'status':'passed','initial_policy_id':identity['policy_id'],'initial_logprob_error':error,
        'atol':3e-4,'rtol':1e-5,'loss':float(loss.detach()),'grad_norm':float(norm),
        'changed_lora_tensors':changed,'optimizer_states':len(optimizer.state),
        'example_id':ex.example_id,'initializer_unchanged':True,
        'scope':'shared SFT initializer loading and one actual supervised update; not full SFT runner or matched-budget efficacy'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:atomic_json(args.output/'result.json',run(args))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
    atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
