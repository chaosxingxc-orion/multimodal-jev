#!/usr/bin/env python3
"""Read-only actual audio/LoRA forwards through verl collation and GRPO loss."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from peft import PeftModel
    from verl.utils.model import extract_multi_modal_inputs
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio, response_log_probs
    from multimodal_jev.rl_run import load_rollout_run
    from multimodal_jev.policy_identity import validate_adapter_export
    from multimodal_jev.paired_rollout import compose_rewards
    from multimodal_jev.verl_audio import make_batch, compute_advantages, policy_loss
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('physical NPU 4 isolation required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('changed input: '+path)
    identity, examples, records = load_rollout_run(config['rollout'], allow_diagnostic=True)
    validate_adapter_export(config['base'], identity['config']['checkpoint'], config['adapter'])
    torch.set_num_threads(4); torch.manual_seed(17)
    processor, base = load_runtime(config['base'], 'npu:0', 'bfloat16')
    model = PeftModel.from_pretrained(base, config['adapter'], is_trainable=False).eval()
    by_id = {ex.example_id: ex for ex in examples}
    waves = {ex.example_id: load_audio(ex.audio) for ex in examples}
    ordered = [by_id[r.example_id] for r in records]
    rewards, _ = compose_rewards(ordered, records, 'grpo', 0.)
    batch = make_batch(processor, ordered, [waves[r.example_id] for r in records],
                       [r.token_ids for r in records], [r.log_probs for r in records],
                       rewards, identity['config']['sampling']['n'])
    batch = compute_advantages(batch)
    total_tokens = int(batch.batch['response_mask'].sum())
    mask = batch.batch['response_mask'].bool()
    old = batch.batch['old_log_probs']
    summaries = {}
    for size in config['micro_batch_sizes']:
        if type(size) is not int or not 1 <= size <= len(records):
            raise ValueError('invalid microbatch size')
        all_logp = []; weighted_loss = 0.
        with torch.inference_mode():
            for begin in range(0, len(records), size):
                part = batch[begin:begin+size]
                part.batch = part.batch.to('npu:0')
                tensors = part.batch
                mm = extract_multi_modal_inputs(part.non_tensor_batch['multi_modal_inputs'])
                mm = {k:v.to(device='npu:0', dtype=model.dtype if v.is_floating_point() else v.dtype)
                      for k,v in mm.items()}
                result = model(input_ids=tensors['input_ids'], attention_mask=tensors['attention_mask'],
                               position_ids=tensors['position_ids'], **mm, use_cache=False)
                logp = response_log_probs(result.logits, tensors['input_ids'], tensors['prompts'].shape[-1])
                if not torch.isfinite(logp[tensors['response_mask'].bool()]).all():
                    raise ValueError('nonfinite response probabilities')
                loss, _ = policy_loss(part, logp)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite verl loss')
                weighted_loss += float(loss)*int(tensors['response_mask'].sum())/total_tokens
                all_logp.append(logp.cpu())
                atomic_json(output/'progress.json', {'micro_batch_size':size,'done':begin+len(part),
                                                     'heartbeat_unix':time.time()})
        joined = torch.cat(all_logp)
        direct_loss, _ = policy_loss(batch, joined)
        aggregation_error = abs(weighted_loss-float(direct_loss))
        error = float((joined-old).abs()[mask].max())
        atomic_json(output/f'micro-{size}.json', {'logprobs':joined.tolist(),'mask':mask.tolist(),
            'old_logprobs':old.tolist(),'example_ids':[r.example_id for r in records],
            'replicates':[r.replicate for r in records]})
        summaries[str(size)] = {'max_logprob_error':error,'threshold':0.02,'probability_passed':error<=0.02,
            'weighted_micro_loss':weighted_loss,'whole_batch_loss':float(direct_loss),
            'aggregation_error':aggregation_error,'aggregation_passed':aggregation_error<=1e-6}
    return {'status':'diagnostic_complete','cases':summaries,'responses':len(records),'tokens':total_tokens,
            'nonzero_advantage_tokens':int((batch.batch['advantages'][mask]!=0).sum()),
            'rewards':rewards,'rollout_sha256':file_hash(Path(config['rollout'])/'rollouts.jsonl'),
            'optimizer_updates':0,
            'scope':'Actual BF16 base+LoRA audio forwards, verl DataProto/GRPO/vanilla loss; no backward, optimizer, native worker or distributed admission.'}


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
