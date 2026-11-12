#!/usr/bin/env python3
"""Read-only real-audio shared LoRA gradient diagnosis on CPU BF16.

No optimizer, training, FSDP or NPU equivalence claim. Negative expected
correctness has the on-policy PPO derivative, not its surrogate value.
"""
import argparse
import json
import math
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def geometry(final, process):
    import torch
    if final.keys() != process.keys() or not final:
        raise ValueError('same nonempty shared parameter coordinates required')
    f2 = p2 = dot = 0.
    for name, f in final.items():
        p = process[name]
        if f.shape != p.shape or not torch.isfinite(f).all() or not torch.isfinite(p).all():
            raise ValueError('aligned finite gradients required')
        f = f.double(); p = p.double()
        f2 += f.square().sum().item(); p2 += p.square().sum().item()
        dot += (f*p).sum().item()
    return dict(final_norm=math.sqrt(f2), process_norm=math.sqrt(p2), dot=dot,
                cosine=dot/math.sqrt(f2*p2) if f2*p2 else None,
                process_to_final_norm=math.sqrt(p2/f2) if f2 else None,
                combined_gradient_dot_final=f2+dot,
                scope='Loss gradients; negative dot means conflict. No Adam update prediction.')


def control():
    import torch
    w = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    f = w.square().sum(); p = -2*f
    gf = torch.autograd.grad(f, w, retain_graph=True)[0]
    gp = torch.autograd.grad(p, w)[0]
    measured = geometry(dict(w=gf), dict(w=gp))
    if abs(measured['cosine']+1)>1e-12 or abs(measured['process_to_final_norm']-2)>1e-12:
        raise AssertionError('known conflicting shared gradient control failed')
    if measured['combined_gradient_dot_final'] >= 0:
        raise AssertionError('opposing auxiliary dominance control failed')
    if geometry(dict(w=gf), dict(w=torch.zeros_like(gp)))['cosine'] is not None:
        raise AssertionError('zero norm must not invent a cosine')
    return measured


def run(config, out):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.slurp_intent_pilot import load_pilot
    from multimodal_jev.intent_actor_inputs import schedule
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only diagnostic required')
    progress=lambda phase,**fields:atomic_json(out/'progress.json',dict(phase=phase,updated_unix=time.time(),**fields))
    progress('authenticate_assets')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('diagnostic pin changed: '+path)
    torch.set_num_threads(4); torch.manual_seed(317)
    check = control()
    policy_id = checkpoint_identity(config['model_path'])[0]+':'+checkpoint_identity(config['initial_adapter'])[0]
    if policy_id != config['policy_id']:
        raise ValueError('fixed accepted policy required')
    items, admission = load_pilot(config['pilot'], config['manifest_sha256'], 'P')
    progress('load_model')
    processor, base = load_runtime(config['model_path'], 'cpu', 'bfloat16')
    adapted = initialize_adapter(base, config).eval()
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    named = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if len(named)!=112 or any(p.dtype!=torch.bfloat16 for p in named.values()):
        raise ValueError('112 BF16 LoRA parameter tensors required')
    before = {n:p.detach().clone() for n,p in named.items()}
    records = {r['id']:r for r in json.loads(Path(config['accepted_records']).read_text())['records']}
    reports=[]; started=time.monotonic()
    for batch_index in config['schedule_steps']:
        chosen=schedule(items,317,batch_index,8)
        total_weight=sum(i['target']['request_weight'] for i in chosen)
        sums=[{n:torch.zeros_like(p,dtype=torch.float32) for n,p in named.items()} for _ in range(2)]
        rows=[]
        for item in chosen:
            progress('encode',batch=batch_index+1,completed=len(rows),recording_id=item['id'])
            target=item['target']; request=item['request']
            if target['span_weights']!=[.1,.1,1.] or target['supervised']!=[True]*3:
                raise ValueError('fixed full process supervision required')
            audio,receipt=load_audio(request.audio,return_receipt=True)
            inputs,positions=encode_span_request(processor,request,audio)
            candidates=[m['token_ids'] for m in span_labels(processor.tokenizer,request.questions)]
            rid,identity=bind_span_request(request,receipt,inputs,positions,candidates,policy_id)
            ref=records[item['id']+'::Q']
            if rid!=ref['request_id'] or identity!=ref['request_identity']:
                raise ValueError('accepted request identity differs')
            inputs={k:v.to(dtype=torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
            progress('forward',batch=batch_index+1,completed=len(rows),recording_id=item['id'],tokens=inputs['input_ids'].shape[-1])
            _,logp=model(inputs,positions,candidates)
            delta=max((a.detach().float()-torch.tensor([b])).abs().max().item()
                      for a,b in zip(logp,ref['candidate_logprobs'],strict=True))
            probabilities=[p[0,y].exp() for p,y in zip(logp,target['targets'],strict=True)]
            losses=[-probabilities[-1], -sum(w*p for w,p in zip(target['span_weights'][:-1],probabilities[:-1],strict=True))]
            gradients=[]
            for j,loss in enumerate(losses):
                progress('backward_final' if j==0 else 'backward_process',batch=batch_index+1,completed=len(rows),recording_id=item['id'])
                values=torch.autograd.grad(loss,tuple(named.values()),retain_graph=(j==0),allow_unused=True)
                gradient={n:(torch.zeros_like(p,dtype=torch.float32) if g is None else g.detach().float())
                          for (n,p),g in zip(named.items(),values,strict=True)}
                gradients.append(gradient)
                for n,g in gradient.items():sums[j][n].add_(g,alpha=target['request_weight']/total_weight)
            rows.append(dict(id=item['id'],request_id=rid,cpu_vs_accepted_max_logprob_error=delta,
                             gold_probabilities=[p.item() for p in probabilities],geometry=geometry(*gradients)))
            progress('record_complete',batch=batch_index+1,completed=len(rows))
            # Release references before the next forward allocates its graph.
            del _,logp,probabilities,losses,loss,values,gradient,gradients
        reports.append(dict(schedule_batch=batch_index+1,recordings=8,geometry=geometry(*sums),records=rows))
        atomic_json(out/f'batch-{batch_index+1}.json',reports[-1])
    if any(not torch.equal(p,before[n]) or p.grad is not None for n,p in named.items()):
        raise AssertionError('read-only diagnostic modified parameters or accumulated gradients')
    return dict(passed=True,control=check,admission=admission,policy_id=policy_id,reports=reports,
                trainable_parameters=sum(p.numel() for p in named.values()),seconds=time.monotonic()-started,
                parameter_execution_and_gradient_dtype='bfloat16',cross_request_accumulation_dtype='float32',
                scope='Real audio, CPU BF16 shared LoRA gradients at fixed step0. Batches1/6 chosen after output-logit diagnosis; exploratory. CPU probabilities reported against accepted NPU policy, no NPU gradient equivalence or task improvement claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
