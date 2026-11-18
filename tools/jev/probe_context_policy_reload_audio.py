#!/usr/bin/env python3
"""Load a complete execution policy into fresh Qwen/PEFT and replay real audio."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_decision_contract import ContextDecisionRequest, ApplicabilityQuestion, encode_context_request
    from multimodal_jev.context_decision_head import ContextDecisionHead
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_policy_snapshot import export_policy, load_policy, lora_contract
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('dependency changed: '+path)
    torch.set_num_threads(2);torch.manual_seed(317)
    reference=json.loads(Path(config['reference_result']).read_text())
    base_id=checkpoint_identity(config['model'])[0]
    if base_id!=config['base_id']:raise ValueError('base identity differs')
    processor=Qwen3ASRProcessor.from_pretrained(config['processor'],local_files_only=True)
    _,base=load_runtime(config['model'],'cpu','bfloat16')
    token_id=config['decision_token_id'];embedding=base.get_input_embeddings().weight
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in ('A','B')]
    if any(len(x)!=1 for x in ids):raise ValueError('single init symbols required')
    with torch.no_grad():embedding[token_id].copy_(embedding[[x[0] for x in ids]].float().mean(0).to(embedding.dtype))
    if tensor_identity(embedding[token_id])!=reference['token_initialization']['after']:
        raise ValueError('marker initialization differs')
    adapted=initialize_adapter(base,dict(lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16')).eval()
    head=ContextDecisionHead(1024,8,1024,.1).to(torch.bfloat16).eval()
    head.load_state_dict(torch.load(config['reference_branch'],weights_only=True),strict=True)
    model=ContextDecisionModel(adapted.get_base_model().model,head,token_id).eval()
    branch=json.loads(Path(config['branch_config']).read_text())['branch']
    contract=dict(base_id=base_id,runtime_profile='cpu-eager-bf16-context-diagnostic-v1',
        request_version='jev-audio-context-applicability-v1',branch=branch,
        token=dict(id=token_id,text='<|jev_decision|>',row_trainable=False,registration_status='test_only'),
        output=dict(field='prob',activation='sigmoid',normalize_across_candidates=False),
        lora=lora_contract(model.backbone))
    atomic_json(output/'progress.json',dict(phase='export_joint_policy'))
    policy_id=export_policy(output/'policy',model,config['processor'],contract)
    # Destroy the live execution state, then recover only from the saved policy.
    original={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.add_(.125)
        embedding[token_id].zero_()
    recovered=load_policy(output/'policy',model,expected_policy_id=policy_id,
                          expected_contract=contract,expected_base_id=base_id)
    if any(not torch.equal(p,original[n]) for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('joint tensors not exactly restored')
    if tensor_identity(embedding[token_id])!=reference['token_initialization']['after']:
        raise ValueError('marker not restored')
    del original
    reloaded=Qwen3ASRProcessor.from_pretrained(recovered['processor'],local_files_only=True)
    records=[]
    for index,r in enumerate(reference['records']):
        data=r['request'];data['questions']=tuple(ApplicabilityQuestion(**q) for q in data['questions'])
        data['requirements']=tuple(data['requirements']);data['history']=tuple(data['history'])
        request=ContextDecisionRequest(**data)
        if file_hash(request.audio)!=r['audio_sha256']:raise ValueError('audio changed')
        wave,receipt=load_audio(request.audio,return_receipt=True)
        inputs,positions=encode_context_request(reloaded,request,wave,decision_token_id=token_id,
            audio_token_id=base.config.audio_token_id,max_tokens=4096)
        if (positions.tolist()!=r['positions'] or tensor_identity(inputs['input_ids'])!=r['input_ids'] or
                receipt!=r['audio_receipt']):raise ValueError('restored processor input differs')
        inputs={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
        atomic_json(output/'progress.json',dict(phase='reloaded_audio_forward',asset=index))
        start=time.monotonic()
        with torch.no_grad():prob=model(inputs,positions)
        expected=torch.tensor(r['probabilities'])
        if not torch.equal(prob,expected):raise ValueError('joint restored actual audio probability differs')
        records.append(dict(audio_id=r['audio_id'],probabilities=prob.tolist(),exact_prior_probability=True,
                            input_ids=r['input_ids'],seconds=time.monotonic()-start))
        atomic_json(output/'records.json',records)
    return dict(policy_id=policy_id,contract=contract,recovered=recovered,records=records,
        optimizer_restored=False,training_dispatch_allowed=False,
        scope='Real Qwen CPU execution snapshot export/perturb/restore including marker, LoRA, head and processor; '
              'compared with earlier two audio outputs. No optimizer update/recovery, NPU, vLLM or task gain.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
