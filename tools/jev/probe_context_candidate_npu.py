#!/usr/bin/env python3
"""Single-card full60 context-head forward/backward diagnostic, no update.

This is a new HF eager branch path. Old Final60 parity acceptance does not apply.
Raw snapshots and repeat error precede any numerical/backend promotion decision.
"""
import argparse
import json
import os
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    import torch_npu
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_decision_head import ContextDecisionHead
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(output)
    if (len(config['physical_devices'])!=1 or
            sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci'+str(config['physical_devices'][0])]):
        raise ValueError('one isolated authorized physical NPU required')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('changed dependency '+p)
    if checkpoint_identity(config['model'])[0]!=config['base_id']:raise ValueError('base differs')
    torch.set_num_threads(2);torch.npu.set_device(0);device=torch.device('npu',0)
    torch.manual_seed(317);torch.npu.manual_seed_all(317)
    torch.use_deterministic_algorithms(True)
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    processor=Qwen3ASRProcessor.from_pretrained(config['processor'],local_files_only=True)
    _,base=load_runtime(config['model'],device,'bfloat16')
    token_id=config['decision_token_id'];embedding=base.get_input_embeddings().weight
    if token_id==base.config.timestamp_token_id:raise ValueError('reserved token collision')
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in ('A','B')]
    if any(len(x)!=1 for x in ids):raise ValueError('single initialization symbols required')
    with torch.no_grad():embedding[token_id].copy_(embedding[[x[0] for x in ids]].float().mean(0).to(embedding.dtype))
    adapted=initialize_adapter(base,dict(lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16')).eval()
    head=ContextDecisionHead(1024,8,1024,.1).to(device=device,dtype=torch.bfloat16).eval()
    head.load_state_dict(torch.load(config['reference_branch'],map_location='cpu',weights_only=True),strict=True)
    model=ContextDecisionModel(adapted.get_base_model().model,head,token_id).eval()
    task=json.loads(Path(config['tasks']).read_text())['intent']
    candidates=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('full sixty candidate definitions required')
    records=[]
    for i,asset in enumerate(config['audio']):
        if file_hash(asset['audio'])!=asset['sha256']:raise ValueError('audio changed')
        request=compile_candidates(asset['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        wave,receipt=load_audio(asset['audio'],return_receipt=True)
        batch,positions=encode_context_request(processor,request,wave,decision_token_id=token_id,
            audio_token_id=base.config.audio_token_id,max_tokens=4096)
        identities={k:tensor_identity(v) for k,v in batch.items()}
        inputs={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()}
        positions=positions.to(device)
        torch.npu.reset_peak_memory_stats();torch.npu.synchronize()
        atomic_json(output/'progress.json',dict(phase='forward_backward',asset=i))
        start=time.monotonic();logits=model.score_logits(inputs,positions);torch.npu.synchronize()
        forward=time.monotonic()-start
        if logits.shape!=(1,60) or not torch.isfinite(logits).all():raise ValueError('finite full60 logits required')
        expected=logits.detach().clone();start=time.monotonic()
        # Structural derivative only: no label, reward, update or training claim.
        coefficients=torch.linspace(-1,1,60,device=device)[None]/60
        (logits*coefficients).sum().backward();torch.npu.synchronize()
        backward=time.monotonic()-start
        gradients={}
        for name,p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('invalid gradient '+name)
                gradients[name]=float(p.grad.float().norm())
        if not any(v>0 for k,v in gradients.items() if '.lora_B.' in k):raise ValueError('no backbone gradient')
        if not all(v>0 for k,v in gradients.items() if k.startswith('decision_head.')):raise ValueError('no branch gradient')
        peak=dict(allocated=torch.npu.max_memory_allocated(),reserved=torch.npu.max_memory_reserved())
        model.zero_grad(set_to_none=True);del logits
        with torch.no_grad():repeat=model.score_logits(inputs,positions)
        torch.npu.synchronize()
        snap=output/f'asset-{i}.pt';torch.save(dict(logits=expected.cpu(),repeat=repeat.cpu(),positions=positions.cpu()),snap)
        records.append(dict(audio_id=asset['id'],audio_sha256=asset['sha256'],request_sha256=content_hash(request.model_input()),
            input_identities=identities,positions=positions.cpu().tolist(),forward_seconds=forward,backward_seconds=backward,
            peak_memory=peak,gradients=gradients,snapshot_sha256=file_hash(snap),
            repeat_logit_max_error=float((expected-repeat).abs().max()),
            repeat_probability_max_error=float((expected.sigmoid()-repeat.sigmoid()).abs().max())))
        atomic_json(output/'records.json',records)
    return dict(records=records,training_dispatch_allowed=False,
        scope='Actual NPU HF eager full60 single requests, diagnostic backward and repeated output; '
              'no optimizer, batch/distributed proof, calibrated/task result or vLLM parity acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
