#!/usr/bin/env python3
"""Actual verl input/audio path, stopping before the first text layer.

All original six TRAIN responses remain fixed false+EOS. Both microbatch sizes
are diagnostic prefixes only: no logits, generation, optimizer or replay of
the complete failed actor. Full captured tensors allow independent comparison.
"""
import argparse
import importlib.metadata
import inspect
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_actor_batch import prepare,evidence
from multimodal_jev.omni_actor_forward import compute_log_probs
from multimodal_jev.omni_prefix_capture import PrefixCaptured,capture_prefix
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_logprobs,native_actor_cpu_device
from probe_omni_pretrained_verl import diagnostic_batch,read
from probe_omni_pretrained_forward import loading_report
from sample_sfc_omni_learning import load_inputs
from infer_qwen3_omni_probe import verify_assets

FAILED_IDENTITY='35cfdf92ddc144f1245f9416590d4c162a072abfe138f0c645ddd793ed96ecf1'


def run(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.device import get_device_name
    from transformers import Qwen3OmniMoeProcessor,Qwen3OmniMoeThinkerForConditionalGeneration
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from multimodal_jev.omni_compact_attention import compact_text_attention
    torch.set_num_threads(16);torch.manual_seed(17);start=time.monotonic()
    progress=lambda phase,**kw:atomic_json(args.output/'progress.json',dict(phase=phase,heartbeat_unix=time.time(),**kw))
    if file_hash(args.failed/'run/identity.json')!=FAILED_IDENTITY:
        raise ValueError('original failed actor input identity required')
    failed=read(args.failed/'run/identity.json')
    if get_device_name()!='cpu' or file_hash(inspect.getfile(DataParallelPPOActor))!='207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce':
        raise ValueError('original actual CPU verl actor required')
    if file_hash(inspect.getfile(Qwen3OmniMoeThinkerForConditionalGeneration))!='a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340':
        raise ValueError('original HF model source required')
    for obj in (diagnostic_batch,prepare,compute_log_probs,native_actor_cpu_device,native_actor_cpu_logprobs,compact_text_attention):
        path=Path(inspect.getfile(inspect.unwrap(obj)));relative=str(path.relative_to(Path(__file__).resolve().parents[2]))
        if file_hash(path)!=failed['source_files'][relative]:raise ValueError('failed actor helper differs: '+relative)
    base,rows,_,checks=load_inputs(failed['config'])
    if {k:importlib.metadata.version(k) for k in base['versions']}!=base['versions']:
        raise ValueError('fixed runtime required')
    progress('verify_model');assets=verify_assets(base)
    if assets['files']!=failed['model_files']:raise ValueError('original model required')
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    prepared=prepare(processor,rows)
    for key,pin in evidence(prepared).items():
        if any(checks[key][name]!=value for name,value in pin.items()):raise ValueError('original prompt changed')
    keys=[tuple(k) for k in failed['keys']];response=failed['response_ids']
    if response!=processor.tokenizer.encode('false',add_special_tokens=False)+[processor.tokenizer.eos_token_id]:
        raise ValueError('original fixed response required')
    batch=diagnostic_batch(processor,prepared,keys,response)
    if file_hash(args.trace/'result.json')!='23ee087e133334428c0e0fcc85b728a4f7459350954b32ef7a22b1b6a284a161':
        raise ValueError('original accepted complete reference traces required')
    trace_reports=read(args.trace/'result.json')['reports']
    refs=[];pins=[]
    for row in failed['order']:
        record=read(args.trace/f'row-{row:02d}.json')
        if record!=trace_reports[row]:raise ValueError('reference row differs from original result')
        path=args.trace/f'row-{row:02d}-trace/layer-00.input.pt'
        if file_hash(path)!=record['reference']['layer-00.input']['sha256']:raise ValueError('reference activation changed')
        refs.append(torch.load(path,map_location='cpu',weights_only=True))
        pins.append(dict(row=row,path=str(path),sha256=file_hash(path)))
    atomic_json(args.output/'identity.json',dict(failed_identity_sha256=FAILED_IDENTITY,order=failed['order'],keys=keys,
        response_ids=response,model_files=assets['files'],reference_inputs=pins,batch_shape=list(batch.batch['input_ids'].shape),
        targets_read=False,weight_updates=0,synthetic_old_log_probs=True,
        source_files={str(p.relative_to(Path(__file__).resolve().parents[2])):file_hash(p)
                      for folder in ('src/multimodal_jev','tools/jev') for p in (Path(__file__).resolve().parents[2]/folder).glob('*.py')}))
    progress('load_thinker')
    model,loading=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(base['audio_model'],
        local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa',output_loading_info=True)
    loading=loading_report(loading);atomic_json(args.output/'loading-info.json',loading)
    if (any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
            or any(not k.startswith(('talker.','code2wav.')) for k in loading.get('unexpected_keys',[]))):
        raise ValueError('complete Thinker load required')
    model.eval().requires_grad_(False)
    config=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,ulysses_sequence_parallel_size=1,
        use_dynamic_bsz=False,use_prefix_grouper=False,entropy_from_logits_with_chunking=False,use_torch_compile=False,
        fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False))
    dist.init_process_group('gloo',init_method=(args.output/'gloo-init').resolve().as_uri(),rank=0,world_size=1)
    records=[]
    try:
        actor=DataParallelPPOActor(config,model)
        with native_actor_cpu_logprobs(),native_actor_cpu_device(actor),torch.inference_mode():
            for size in (1,2):
                for offset in range(0,6,size):
                    progress('capture_prefix',micro_batch_size=size,offset=offset)
                    part=batch[offset:offset+size];storage={};begin=time.monotonic()
                    with compact_text_attention(hf) as attention_records:
                        try:
                            with capture_prefix(model.model.layers[0],part.batch['attention_mask'],storage):
                                compute_log_probs(actor,part,micro_batch_size=size,pad_token_id=processor.tokenizer.pad_token_id)
                        except PrefixCaptured:pass
                        else:raise ValueError('first-layer diagnostic stop was not reached')
                    if not storage or attention_records:raise ValueError('capture must precede all text attention')
                    path=args.output/f'micro-{size}-offset-{offset}.pt';torch.save(storage,path)
                    comparisons=[]
                    for j in range(size):
                        valid=storage['hidden_states'][j,storage['token_mask'][j].bool()];ref=refs[offset+j]
                        if valid.shape!=ref.shape or valid.dtype!=ref.dtype:raise ValueError('original valid activation shape required')
                        delta=(valid.float()-ref.float()).abs()
                        comparisons.append(dict(row=failed['order'][offset+j],equal=bool(torch.equal(valid,ref)),
                            max_abs_error=float(delta.max()),differing_elements=int((valid!=ref).sum())))
                    record=dict(micro_batch_size=size,offset=offset,tensor_file=path.name,tensor_sha256=file_hash(path),
                        comparisons=comparisons,seconds=time.monotonic()-begin,text_attention_calls=0)
                    atomic_json(args.output/f'micro-{size}-offset-{offset}.json',record);records.append(record)
    finally:dist.destroy_process_group()
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('unexpected gradients')
    progress('verify_weights_after');verify_assets(base)
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',records=records,elapsed_seconds=time.monotonic()-start,
        identity_sha256=file_hash(args.output/'identity.json'),weight_updates=0,targets_read=False,
        scope='Nine actual verl calls stopped before first text decoder layer: six batch1 and three batch2 audio/input prefixes. Saved tensors; no actor probabilities, full text forward, generation or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('failed','trace','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
