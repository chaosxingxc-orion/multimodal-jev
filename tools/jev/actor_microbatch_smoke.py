#!/usr/bin/env python3
"""Compare full versus uneven microbatch actual Qwen/verl gradients and updates.

Uses deliberately controlled TRAIN responses, never a vLLM rollout receipt.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from peft import LoraConfig, get_peft_model
from multimodal_jev.asr import load_runtime
from multimodal_jev.audio_policy import load_audio, hf_prompt, answer_ids, response_log_probs
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.rollout import Rollout
from multimodal_jev.rl_update import update_actor
from multimodal_jev.task_schema import JevExample


def run(args):
    torch.set_num_threads(4); torch.manual_seed(17)
    with args.manifest.open() as handle:
        ex = next(JevExample.from_record(row) for line in handle
                  if (row := json.loads(line))['metadata']['variant'] == 'noul.True')
    if ex.split != 'train': raise ValueError('TRAIN only')
    processor, base = load_runtime(args.model, 'cpu', 'float32')
    modules = [n for n, _ in base.named_modules() if n.startswith('model.language_model.')
               and n.rsplit('.', 1)[-1] in {'q_proj', 'v_proj'}]
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
                          target_modules=modules, task_type='CAUSAL_LM', bias='none')).eval()
    wave = load_audio(ex.audio); records = []
    for i, value in enumerate((True, False, 'invalid boolean with a longer response')):
        ids = answer_ids(processor.tokenizer, value)
        inputs = dict(hf_prompt(processor, ex, wave)); boundary = inputs['input_ids'].shape[-1]
        inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor([ids])), dim=-1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        with torch.no_grad():
            logp = response_log_probs(model(**inputs, use_cache=False).logits,
                                      inputs['input_ids'], boundary)[0].tolist()
        records.append(Rollout(ex.example_id, i, 'controlled-fixture', tuple(ids), tuple(logp),
                               processor.tokenizer.decode(ids, skip_special_tokens=True), 'stop'))
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]; results = []; snapshots = []
    for size in (3, 2):
        with torch.no_grad():
            for p, saved in zip(params, initial): p.copy_(saved)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0., foreach=False, fused=False)
        atomic_json(args.output/'progress.json', {'phase': 'update', 'micro_batch_size': size,
                                                 'heartbeat_unix': time.time()})
        metrics = update_actor(model, processor, optimizer, [ex], [wave], records,
                               'controlled-fixture', 3, max_logprob_error=3e-4, micro_batch_size=size)
        assert metrics['updated']
        results.append(metrics)
        snapshots.append({'parameters': [p.detach().clone() for p in params],
                          'gradients': [p.grad.clone() for p in params],
                          'exp_avg': [optimizer.state[p]['exp_avg'].clone() for p in params]})
    errors = {}; failures = []; diagnostics = {}
    for key in snapshots[0]:
        pairs = list(zip(snapshots[0][key], snapshots[1][key]))
        errors[key] = max((a-b).abs().max().item() for a,b in pairs)
        diagnostics[key] = {'max_absolute_error': errors[key],
                            'reference_max': max(a.abs().max().item() for a,b in pairs),
                            'error_l2': sum(float((a.double()-b.double()).square().sum()) for a,b in pairs)**0.5,
                            'reference_l2': sum(float(a.double().square().sum()) for a,b in pairs)**0.5}
        for a,b in pairs:
            try: torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
            except AssertionError as exc: failures.append({'field': key, 'error': str(exc)})
    atomic_json(args.output/'comparison.json',{'metrics':results,'diagnostics':diagnostics,
                'failures':failures,'tolerances':{'atol':2e-6,'rtol':2e-4},
                'response_lengths':[len(r.token_ids) for r in records]})
    # A bad final microbatch must reject the whole update before optimizer.step.
    from dataclasses import replace
    bad = [*records[:2], replace(records[2], log_probs=tuple(p-1 for p in records[2].log_probs))]
    with torch.no_grad():
        for p,saved in zip(params,initial): p.copy_(saved)
    optimizer = torch.optim.AdamW(params, lr=1e-4, foreach=False, fused=False)
    try:
        update_actor(model,processor,optimizer,[ex],[wave],bad,'controlled-fixture',3,
                     max_logprob_error=3e-4,micro_batch_size=2)
    except ValueError as exc:
        if 'logprob mismatch' not in str(exc): raise
    else: raise AssertionError('late microbatch mismatch accepted')
    assert not optimizer.state
    assert all(torch.equal(p,saved) for p,saved in zip(params,initial))
    atomic_json(args.output/'late-parity-check.json',{'no_update':True,'optimizer_states':0})
    if failures: raise AssertionError('microbatch equivalence gate failed; see comparison.json')
    assert abs(results[0]['loss']-results[1]['loss']) < 2e-5
    return {'status':'passed','metrics':results,'max_absolute_errors':errors,
            'response_lengths':[len(r.token_ids) for r in records],
            'late_parity_failure_no_update':True,
            'source_sha256':{name:file_hash(Path(__file__).parents[2]/'src/multimodal_jev'/name)
                             for name in ('rl_update.py','verl_audio.py','audio_policy.py')},
            'scope':'real Qwen/verl controlled response gradient/update equivalence; no vLLM rollout or efficacy'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(args))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
    atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
