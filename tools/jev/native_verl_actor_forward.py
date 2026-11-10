#!/usr/bin/env python3
"""Call pinned verl's actual actor microbatch method on real TRAIN audio.

Single-process Gloo initializes the native actor's rank query only. This is
not a Ray/FSDP training run, rollout test, or optimizer update.
"""
import argparse
import inspect
import json
from pathlib import Path
import time
from contextlib import nullcontext

from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.device import get_device_name
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio, teacher_forcing, answer_ids, response_log_probs
    from multimodal_jev.sft import training_batches
    from multimodal_jev.verl_audio import make_batch

    torch.set_num_threads(4)
    if get_device_name() != 'cpu':
        raise ValueError('this registered probe requires CPU; no accelerator fallback')
    config=json.loads(args.config.read_text())
    groups=training_batches(config)
    # Fixed first two recording groups; distinct schema lengths and audio.
    examples=[groups[0][4],groups[1][0]]
    if any(ex.split!='train' for ex in examples) or examples[0].audio==examples[1].audio:
        raise ValueError('two distinct TRAIN recordings required')
    for ex in examples:
        if file_hash(ex.audio)!=ex.metadata['audio_sha256']:
            raise ValueError('audio hash mismatch')
    actor_source=Path(inspect.getfile(DataParallelPPOActor))
    if file_hash(actor_source)!=args.actor_sha256:
        raise ValueError('pinned native actor source changed')
    dist.init_process_group('gloo',init_method=(args.output/'gloo-init').resolve().as_uri(),rank=0,world_size=1)
    try:
        processor,model=load_runtime(config['model_path'],'cpu','float32')
        native_config=OmegaConf.create({
            'use_remove_padding':False,'use_fused_kernels':False,
            'ulysses_sequence_parallel_size':1,'use_dynamic_bsz':False,
            'use_prefix_grouper':False,'entropy_from_logits_with_chunking':False,
            'use_torch_compile':False,'fsdp_config':{'dtype':'float32'},
            'calculate_sum_pi_squared':False,'entropy_checkpointing':False})
        actor=DataParallelPPOActor(native_config,model)
        waves=[load_audio(ex.audio) for ex in examples]
        rows=[examples[0],examples[0],examples[1],examples[1]]
        audios=[waves[0],waves[0],waves[1],waves[1]]
        # Deliberately controlled response fixtures, never rollout evidence.
        values=[examples[0].target,None,examples[1].target,None]
        responses=[answer_ids(processor.tokenizer,value) for value in values]
        batch=make_batch(processor,rows,audios,responses,
                         [[-2.]*len(ids) for ids in responses],[1.,0.,1.,0.],2)
        inputs={**batch.batch,**batch.non_tensor_batch}
        atomic_json(args.output/'progress.json',{'phase':'native_forward','heartbeat_unix':time.time()})
        from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_logprobs
        dispatch=(native_actor_cpu_logprobs() if args.cpu_logprob_adapter else nullcontext())
        with torch.inference_mode(),dispatch:
            observed=actor._forward_micro_batch(inputs,temperature=1.,calculate_entropy=False)['log_probs']
            errors=[]
            for i,(example,wave,value) in enumerate(zip(rows,audios,values)):
                from dataclasses import replace
                # Replacing target affects teacher-forcing labels only, never the prompt.
                one,boundary=teacher_forcing(processor,replace(example,target=value),wave)
                one.pop('labels')
                expected=response_log_probs(model(**one,use_cache=False).logits,one['input_ids'],boundary)[0]
                actual=observed[i,:len(responses[i])]
                if not torch.isfinite(actual).all():raise ValueError('nonfinite native log-prob')
                errors.append(float((actual-expected).abs().max()))
                torch.testing.assert_close(actual,expected,atol=3e-4,rtol=1e-5)
                atomic_json(args.output/'progress.json',{'phase':'unpacked_comparison','completed':i+1,
                    'max_error':max(errors),'heartbeat_unix':time.time()})
        atomic_json(args.output/'result.json',{
            'status':'passed','actor_source_sha256':file_hash(actor_source),
            'config_sha256':file_hash(args.config),'native_config':OmegaConf.to_container(native_config),
            'example_ids':[ex.example_id for ex in examples],
            'audio_sha256':{ex.audio:ex.metadata['audio_sha256'] for ex in examples},
            'responses':4,'response_lengths':list(map(len,responses)),
            'cpu_logprob_adapter':args.cpu_logprob_adapter,
            'max_logprob_error':max(errors),'per_response_error':errors,
            'atol':3e-4,'rtol':1e-5,'process_group':{'backend':'gloo','world_size':1},
            'scope':('native verl actor microbatch forward with explicit CPU log-prob adapter'
                     if args.cpu_logprob_adapter else 'unmodified native verl actor microbatch forward')+
                    ' versus unpadded HF on real audio; '
                    'controlled responses; no compute_log_prob wrapper, optimizer, Ray, FSDP or multi-rank validation'})
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--actor-sha256',required=True)
    parser.add_argument('--cpu-logprob-adapter',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
        raise
    atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


if __name__=='__main__':
    main()
