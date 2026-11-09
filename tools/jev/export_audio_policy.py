#!/usr/bin/env python3
"""Merge completed LoRA into an immutable HF checkpoint and verify reload parity."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import torch
from peft import PeftModel
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import teacher_forcing,load_audio,response_log_probs
from multimodal_jev.task_schema import JevExample
from multimodal_jev.baseline import file_hash,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    result=json.loads((args.run/'result.json').read_text())
    terminal=json.loads((args.run/'terminal.json').read_text())
    if terminal['exit_code']!=0 or result['status']!='passed':raise ValueError('successful completed training required')
    config=result['config'];identity=json.loads((args.run/'identity.json').read_text())
    if file_hash(Path(config['model_path'])/'model.safetensors')!=identity['weights_sha256']:
        raise ValueError('base checkpoint changed')
    if identity.get('base_files') is not None:
        from multimodal_jev.policy_identity import checkpoint_identity
        if checkpoint_identity(config['model_path'])[1]!=identity['base_files']:
            raise ValueError('base processor/configuration changed')
    adapter=args.run/'adapter'
    adapter_files={p.name:file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if result.get('adapter_files') is not None and result['adapter_files']!=adapter_files:
        raise ValueError('completed adapter files changed')
    release=hashlib.sha256(json.dumps({'adapter':adapter_files,'base':identity['weights_sha256']},sort_keys=True).encode()).hexdigest()
    destination=args.output_root/release
    if destination.exists():raise FileExistsError('export exists')
    args.output_root.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.export-',dir=args.output_root))
    torch.set_num_threads(4);start=time.monotonic()
    try:
        processor,base=load_runtime(config['model_path'],'cpu','float32')
        model=PeftModel.from_pretrained(base,adapter,is_trainable=False).eval()
        # Fixed TRAIN-only examples; this is numerical export verification, not evaluation.
        wanted=set(identity['example_ids'][:2]);examples=[]
        if file_hash(config['train_manifest'])!=identity['manifest_sha256']:raise ValueError('training manifest changed')
        for line in Path(config['train_manifest']).open():
            row=json.loads(line)
            if row['example_id'] in wanted:examples.append(JevExample.from_record(row))
        if len(examples)!=len(wanted):raise ValueError('verification examples missing')
        if not examples or any(ex.split!='train' for ex in examples):
            raise ValueError('nonempty TRAIN verification examples required')
        inputs=[];reference=[]
        for ex in examples:
            if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('verification audio changed')
            batch,boundary=teacher_forcing(processor,ex,load_audio(ex.audio))
            inputs.append((batch,boundary))
            with torch.inference_mode():
                reference.append(response_log_probs(model(**batch,use_cache=False).logits,batch['input_ids'],boundary).clone())
        merged=model.merge_and_unload(safe_merge=True).eval()
        merge_errors=[]
        for (batch,boundary),ref in zip(inputs,reference):
            with torch.inference_mode():
                logp=response_log_probs(merged(**batch,use_cache=False).logits,batch['input_ids'],boundary)
            torch.testing.assert_close(logp,ref,atol=3e-4,rtol=1e-5)
            merge_errors.append(float((logp-ref).abs().max()))
        merged.save_pretrained(stage,safe_serialization=True);processor.save_pretrained(stage)
        del model,base,merged;gc.collect()
        _,reloaded=load_runtime(stage,'cpu','float32')
        reload_errors=[]
        for (batch,boundary),ref in zip(inputs,reference):
            with torch.inference_mode():
                logp=response_log_probs(reloaded(**batch,use_cache=False).logits,batch['input_ids'],boundary)
            torch.testing.assert_close(logp,ref,atol=3e-4,rtol=1e-5)
            reload_errors.append(float((logp-ref).abs().max()))
        files={p.name:file_hash(p) for p in sorted(stage.iterdir()) if p.is_file()}
        receipt={'status':'passed','release_id':release,'training_run':str(args.run),
                 'training_scope':result.get('scope','unspecified'),
                 'adapter_files':adapter_files,'base_sha256':identity['weights_sha256'],
                 'export_files':files,'merge_max_logprob_error':max(merge_errors),
                 'reload_max_logprob_error':max(reload_errors),'atol':3e-4,'rtol':1e-5,
                 'example_ids':[ex.example_id for ex in examples],
                 'elapsed_seconds':time.monotonic()-start,'scope':'HF merged/reloaded policy parity; vLLM engine verification still required'}
        atomic_json(stage/'EXPORT.json',receipt)
        # tempfile starts with 0700; published shared checkpoints must be readable
        # by the workspace user and later inference containers.
        for path in stage.rglob('*'):
            if path.is_file():path.chmod(path.stat().st_mode | 0o444)
            elif path.is_dir():path.chmod(path.stat().st_mode | 0o555)
        stage.chmod(stage.stat().st_mode | 0o555)
        os.rename(stage,destination)
        print(json.dumps({'path':str(destination),**receipt},indent=2))
    except BaseException as exc:
        # Retain failed stage and terminal receipt for diagnosis; never publish it.
        atomic_json(stage/'FAILURE.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
        raise


if __name__=='__main__':main()
