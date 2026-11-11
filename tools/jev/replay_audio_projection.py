#!/usr/bin/env python3
"""Diagnostic intervention: replay fixed acoustic projections, never train.

HF-native projection replay is a plumbing control. vLLM projection replay asks
whether language-side probability differences survive identical audio embeds.
Neither arm is an admissible actor or a replacement for saved old logprobs.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import hf_prompt, load_audio, response_log_probs
    from multimodal_jev.rl_run import load_rollout_run
    nodes = sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
    if nodes != ['davinci4']:
        raise ValueError('isolated physical NPU 4 required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('changed input: ' + path)
    identity, examples, records = load_rollout_run(config['rollout'])
    capture = Path(config['capture'])
    capture_id = json.loads((capture/'identity.json').read_text())
    capture_result = json.loads((capture/'result.json').read_text())
    if json.loads((capture/'terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('capture did not complete')
    if capture_id['policy_id'] != identity['policy_id']:
        raise ValueError('capture policy mismatch')
    wanted = set(capture_id['config']['example_ids'])
    selected = {ex.example_id: ex for ex in examples if ex.example_id in wanted}
    if set(selected) != wanted or len({ex.audio for ex in selected.values()}) != 1:
        raise ValueError('one identical recording required for projection replay')
    for side in ('hf', 'vllm'):
        if file_hash(capture/side/'projected.pt') != capture_result['stages']['projected'][side+'_sha256']:
            raise ValueError('captured projection changed')
    prompts = {r['example_id']: r['prompt_token_ids'] for r in json.loads((capture/'prompt-identity.json').read_text())}
    torch.set_num_threads(4); torch.manual_seed(17)
    processor, model = load_runtime(identity['config']['checkpoint'], 'npu:0', 'bfloat16')
    projections = {side: torch.load(capture/side/'projected.pt', weights_only=True).to('npu:0')
                   for side in ('hf', 'vllm')}
    for tensor in projections.values():
        if tensor.dtype != model.dtype or tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError('invalid saved projection')
    rows = []
    for record in records:
        if record.example_id not in selected:
            continue
        ex = selected[record.example_id]
        batch = dict(hf_prompt(processor, ex, load_audio(ex.audio)))
        if batch['input_ids'][0].tolist() != prompts[ex.example_id]:
            raise ValueError('expanded prompt mismatch')
        boundary = batch['input_ids'].shape[-1]
        batch['input_ids'] = torch.cat((batch['input_ids'], torch.tensor([record.token_ids])), dim=-1)
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['position_ids'] = torch.arange(batch['input_ids'].shape[-1]).unsqueeze(0)
        batch = {k: v.to(device='npu:0', dtype=model.dtype if v.is_floating_point() else v.dtype)
                 for k, v in batch.items()}
        values = {}
        with torch.inference_mode():
            for arm in ('native', 'hf', 'vllm'):
                calls = []
                handle = None
                if arm != 'native':
                    def replacement(_module, _inputs, actual):
                        saved = projections[arm]
                        if actual.shape != saved.shape or actual.dtype != saved.dtype:
                            raise ValueError('projection shape/dtype mismatch')
                        calls.append(float((actual-saved).abs().max()))
                        return saved
                    handle = model.model.multi_modal_projector.register_forward_hook(replacement)
                try:
                    logits = model(**batch, use_cache=False).logits
                    logp = response_log_probs(logits, batch['input_ids'], boundary)[0]
                    if not torch.isfinite(logp).all():
                        raise ValueError('nonfinite response probability')
                    values[arm] = {'logprobs': logp.tolist(), 'replacement_max_abs_errors': calls}
                    if arm != 'native' and len(calls) != 1:
                        raise ValueError('expected one projector invocation')
                finally:
                    if handle is not None:
                        handle.remove()
        row = {'example_id': record.example_id, 'replicate': record.replicate,
               'token_ids': record.token_ids, 'vllm_logprobs': record.log_probs, 'arms': values}
        rows.append(row)
        atomic_json(output/f'response-{len(rows)-1:03d}.json', row)
        atomic_json(output/'progress.json', {'done':len(rows), 'heartbeat_unix':time.time()})
    if len(rows) != len(wanted)*identity['config']['sampling']['n']:
        raise ValueError('response coverage mismatch')
    maxima = {arm: max(abs(a-b) for row in rows
                      for a,b in zip(row['arms'][arm]['logprobs'], row['vllm_logprobs']))
              for arm in ('native','hf','vllm')}
    control_error = max(abs(a-b) for row in rows for a,b in
                        zip(row['arms']['native']['logprobs'], row['arms']['hf']['logprobs']))
    return {'status':'diagnostic_complete', 'responses':len(rows), 'policy_id':identity['policy_id'],
            'max_logprob_error_vs_sampler':maxima, 'native_vs_hf_replay_max_error':control_error,
            'threshold_reference_only':0.02, 'optimizer_updates':0,
            'scope':'Single-recording projection intervention; not an actor admission or task improvement.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(args.output/'result.json', run(json.loads(args.config.read_text()), args.output))
        atomic_json(args.output/'terminal.json', {'exit_code':0,'finished_unix':time.time()})
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', {'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
        raise
