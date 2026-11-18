#!/usr/bin/env python3
"""Real audio hidden states, frozen model and three same-position readouts."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.slurp_training import content_hash


def run(config, output):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    from audit_harness_routing_support import policy
    from probe_harness_routing_context import request_for
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('this diagnostic is CPU-only')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('changed input: '+path)
    torch.set_num_threads(2)
    processor,model=load_runtime(config['model'],'cpu','bfloat16')
    model.requires_grad_(False)
    if model.lm_head.bias is not None:raise ValueError('diagnostic assumes bias-free frozen head')
    task=json.loads(Path(config['tasks']).read_text())['intent']
    spec=policy(sorted(task['candidate_pairs']),'partition6',317)
    records=[]
    for index,asset in enumerate(config['audio']):
        if file_hash(asset['audio'])!=asset['sha256']:raise ValueError('audio changed')
        wave,receipt=load_audio(asset['audio'],return_receipt=True)
        request=request_for(asset['audio'],task,spec,structured=True,representation='compact_json_v1')
        batch,positions=encode_span_request(processor,request,wave)
        mapping=span_labels(processor.tokenizer,request.questions)
        token_ids=mapping[0]['token_ids']
        if any(list(m['values'])!=[False,True] or m['token_ids']!=token_ids for m in mapping[:-1]):
            raise ValueError('matching Boolean support before Final required')
        if int(batch['input_ids'].shape[-1])>4096:raise ValueError('no input truncation allowed')
        inputs={k:v.to(dtype=torch.bfloat16) if v.is_floating_point() else v for k,v in batch.items()}
        start=time.monotonic()
        atomic_json(output/'progress.json',dict(phase='audio_forward',asset=index,completed=len(records)))
        with torch.inference_mode():
            hidden=model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
            pos=positions[:,:-1]
            selected=hidden.gather(1,pos.unsqueeze(-1).expand(-1,-1,hidden.shape[-1]))
            full_logits=model.lm_head(selected)[...,token_ids]
            pair_weight=model.lm_head.weight[token_ids].detach().clone()
            pair_logits=torch.nn.functional.linear(selected,pair_weight)
            scalar_weight=(pair_weight[1].float()-pair_weight[0].float()).to(torch.bfloat16)[None,:]
            scalar_logits=torch.nn.functional.linear(selected,scalar_weight).squeeze(-1)
            probs=dict(native=full_logits.float().softmax(-1)[...,1],
                       two_row=pair_logits.float().softmax(-1)[...,1],
                       scalar=scalar_logits.float().sigmoid())
            comparisons={name:dict(max_probability_error=float((value-probs['native']).abs().max()),
                threshold_disagreements=int(((value>=.5)!=(probs['native']>=.5)).sum()))
                for name,value in probs.items() if name!='native'}
        snapshot=output/f'asset-{index}-readout.pt'
        torch.save(dict(selected=selected.clone(),pair_weight=pair_weight,scalar_weight=scalar_weight,
                        native_logits=full_logits.clone()),snapshot)
        saved=torch.load(snapshot,weights_only=True)
        restored=torch.nn.functional.linear(saved['selected'],saved['scalar_weight']).squeeze(-1).float().sigmoid()
        if not torch.equal(restored,probs['scalar']):raise ValueError('scalar save/load output differs')
        record=dict(audio_id=asset['id'],audio_sha256=asset['sha256'],audio_processing=receipt,
            request=request.model_input(),request_sha256=content_hash(request.model_input()),
            input_tokens=int(batch['input_ids'].shape[-1]),positions=positions.tolist(),
            candidate_mapping=mapping,input_ids=tensor_identity(batch['input_ids']),
            input_features=tensor_identity(batch['input_features']),hidden=tensor_identity(selected),
            seconds=time.monotonic()-start,readout_snapshot_sha256=file_hash(snapshot),
            probabilities={k:v.tolist() for k,v in probs.items()},comparisons=comparisons,
            exact_scalar_restore=True)
        records.append(record)
        atomic_json(output/'records.json',records)
        del hidden,selected,inputs,batch
    return dict(schema='numeric-head-real-audio-cpu-v1',records=records,
        config_sha256=content_hash(config),training_dispatch_allowed=False,
        scope='Frozen BF16 CPU eager backbone, actual median/longest audio, six Boolean positions under partition6. No labels, new token, trained numeric head, NPU/vLLM or task-gain acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
