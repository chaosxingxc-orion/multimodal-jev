#!/usr/bin/env python3
"""Frozen pretrained Omni -> actual verl wrapper, after accepted TRAIN6 forward.

Controlled response tokens and explicit CPU adapters. Not rollout probabilities
or a training batch: old_log_probs are labeled synthetic placeholders.
"""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import resource
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_actor_batch import prepare,evidence
from multimodal_jev.omni_actor_forward import compute_log_probs
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_logprobs,native_actor_cpu_device
from sample_sfc_omni_learning import load_inputs
from infer_qwen3_omni_probe import verify_assets
from probe_omni_pretrained_forward import loading_report
from verify_omni_pretrained_forward import check_records,check_container


def read(path):return json.loads(Path(path).read_text())


def prior_evidence(root,acceptance,acceptance_hash,inspection,inspection_hash):
    if file_hash(acceptance)!=acceptance_hash or file_hash(inspection)!=inspection_hash:
        raise ValueError('registered prior acceptance or inspection changed')
    accepted=read(acceptance);launch=read(root/'launch.json');intent=read(root/'intent.json')
    check_container(read(inspection),launch,intent)
    result=read(root/'run/result.json')
    if (accepted['schema']!='omni-pretrained-cpu-forward-acceptance-v1'
            or accepted['container_id']!=launch['container_id'] or accepted['image']!=launch['image']
            or accepted['launch_sha256']!=file_hash(root/'launch.json')
            or accepted['intent_sha256']!=file_hash(root/'intent.json')
            or accepted['result_sha256']!=file_hash(root/'run/result.json')
            or accepted['forwards']!=6 or accepted['weight_updates']!=0
            or result['status']!='passed' or result['weight_updates']!=0
            or result['identity_sha256']!=file_hash(root/'run/identity.json')
            or read(root/'run/terminal.json')['exit_code']!=0
            or file_hash(root/'config.json')!=intent['config_sha256']):
        raise ValueError('complete accepted original TRAIN6 forward required')
    return accepted,result,read(root/'run/identity.json')


def diagnostic_batch(processor,prepared,keys,response):
    import numpy as np
    import torch
    from verl import DataProto
    items=[prepared[k] for k in keys];n=len(items);pmax=max(len(x['input_ids']) for x in items)
    width=max(x['input_features'].shape[-1] for x in items);rlen=len(response)
    if pmax+rlen>4096:raise ValueError('padded context overflow')
    pad=processor.tokenizer.pad_token_id
    if pad is None:raise ValueError('actual pad token required')
    ids=torch.full((n,pmax+rlen),pad,dtype=torch.long);mask=torch.zeros_like(ids)
    audio=np.empty(n,dtype=object)
    for i,item in enumerate(items):
        length=len(item['input_ids']);ids[i,pmax-length:pmax]=item['input_ids']
        ids[i,pmax:]=torch.tensor(response);mask[i,pmax-length:]=1
        audio[i]={k:torch.nn.functional.pad(item[k],(0,width-item[k].shape[-1]))
                  for k in ('input_features','feature_attention_mask')}
    positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,1)
    return DataProto.from_dict(tensors=dict(input_ids=ids,attention_mask=mask,position_ids=positions,
        responses=ids[:,pmax:].clone(),response_mask=mask[:,pmax:].clone(),
        old_log_probs=torch.full((n,rlen),-2.)),non_tensors={'multi_modal_inputs':audio},
        meta_info=dict(audio_contract='sfc-jev-qwen3-omni-padded-v1',requires_remove_padding=False,
            requires_sequence_parallel_size=1,synthetic_old_log_probs=True))


def run(args):
    start=time.monotonic();out=args.output
    accepted,prior,identity=prior_evidence(args.prior,args.acceptance,args.acceptance_hash,
                                          args.inspection,args.inspection_hash)
    cfg=read(args.prior/'config.json');base,rows,preflight,checks=load_inputs(cfg)
    if identity['config']!=cfg:raise ValueError('prior input identity changed')
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.device import get_device_name
    from transformers import Qwen3OmniMoeProcessor,Qwen3OmniMoeThinkerForConditionalGeneration
    torch.set_num_threads(16);torch.manual_seed(17)
    if get_device_name()!='cpu':raise ValueError('CPU-only diagnostic')
    if file_hash(inspect.getfile(DataParallelPPOActor))!='207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce':
        raise ValueError('fixed actual verl actor required')
    if {k:importlib.metadata.version(k) for k in base['versions']}!=base['versions']:
        raise ValueError('fixed model runtime required')
    progress=lambda phase,**kw:atomic_json(out/'progress.json',dict(phase=phase,heartbeat_unix=time.time(),**kw))
    progress('verify_weights');assets=verify_assets(base)
    if assets['files']!=identity['model_files']:raise ValueError('prior model differs')
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    response=processor.tokenizer.encode('false',add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    check_records(prior['responses'],rows,checks,response)
    progress('prepare_audio');prepared=prepare(processor,rows)
    for k,pin in evidence(prepared).items():
        if any(checks[k][name]!=v for name,v in pin.items()):raise ValueError('processor evidence differs')
    # Reorder every original row: pair each double-audio row with a single row,
    # retaining all six original first decisions (no quality-based sampling).
    doubles=[i for i,r in enumerate(rows) if len(r['audio_sha256'])==2]
    singles=[i for i,r in enumerate(rows) if len(r['audio_sha256'])==1]
    order=[x for pair in zip(doubles,singles) for x in pair]+singles[len(doubles):]
    if sorted(order)!=list(range(6)):raise ValueError('complete six-row ordering required')
    keys=[(rows[i]['id'],rows[i]['input']['decisions'][0]['id']) for i in order]
    batch=diagnostic_batch(processor,prepared,keys,response)
    prior_logp=torch.tensor([prior['responses'][i]['response_log_probs'] for i in order])
    atomic_json(out/'identity.json',dict(prior_acceptance_sha256=args.acceptance_hash,
        inspection_sha256=args.inspection_hash,config=cfg,order=order,keys=keys,response_ids=response,
        model_files=assets['files'],synthetic_old_log_probs=True,targets_read=False,
        source_files={str(p.relative_to(Path(__file__).parents[2])):file_hash(p)
            for folder in ('src/multimodal_jev','tools/jev') for p in (Path(__file__).parents[2]/folder).glob('*.py')}))
    if args.preflight_only:
        atomic_json(out/'result.json',dict(status='passed',preflight_only=True,model_loaded=False,
            tensor_shape=list(batch.batch['input_ids'].shape),order=order,weight_updates=0));return
    progress('load_thinker')
    model,loading=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(base['audio_model'],
        local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cpu',
        attn_implementation='sdpa',output_loading_info=True)
    loading=loading_report(loading);atomic_json(out/'loading-info.json',loading)
    if (any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
            or any(not k.startswith(('talker.','code2wav.')) for k in loading.get('unexpected_keys',[]))):
        raise ValueError('incomplete Thinker load')
    model.eval().requires_grad_(False)
    native=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
        ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
        entropy_from_logits_with_chunking=False,use_torch_compile=False,
        fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False))
    dist.init_process_group('gloo',init_method=(out/'gloo-init').resolve().as_uri(),rank=0,world_size=1)
    values=[];old=batch.batch['old_log_probs'].clone();meta=dict(batch.meta_info)
    try:
        actor=DataParallelPPOActor(native,model)
        with native_actor_cpu_logprobs(),native_actor_cpu_device(actor):
            for size in (1,2):
                progress('actual_verl_compute_log_prob',micro_batch_size=size)
                begin=time.monotonic()
                logp=compute_log_probs(actor,batch,micro_batch_size=size,pad_token_id=processor.tokenizer.pad_token_id).float()
                values.append(logp)
                atomic_json(out/f'microbatch-{size}.json',dict(log_probs=logp.tolist(),seconds=time.monotonic()-begin,
                    max_error_from_unpadded=float((logp-prior_logp).abs().max())))
    finally:dist.destroy_process_group()
    assert torch.equal(old,batch.batch['old_log_probs']) and batch.meta_info==meta
    errors=dict(microbatch=float((values[0]-values[1]).abs().max()),
                unpadded=[float((v-prior_logp).abs().max()) for v in values])
    # Register before execution; preserve all raw differences if the gate fails.
    passed=errors['microbatch']<=.1 and max(errors['unpadded'])<=.1
    atomic_json(out/'comparison.json',dict(errors=errors,atol=.1,passed=passed,
        note='BF16 CPU diagnostic threshold, not NPU/vLLM parity or quality acceptance'))
    if not passed:raise ValueError('predeclared BF16 actor comparison threshold exceeded')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('unexpected trainable actor')
    progress('verify_weights_after');verify_assets(base)
    atomic_json(out/'result.json',dict(status='passed',preflight_only=False,model_loaded=True,weight_updates=0,
        errors=errors,atol=.1,elapsed_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Actual frozen pretrained Thinker CPU verl wrapper and mixed audio microbatches; controlled '
              'response tokens and explicit CPU adapters, no sampled probabilities, vLLM/NPU parity or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('prior','acceptance','inspection','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('acceptance-hash','inspection-hash'):p.add_argument('--'+key,required=True)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
