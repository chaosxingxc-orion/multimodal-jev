#!/usr/bin/env python3
"""Actual audio, marker-self backbone and shallow head diagnostic; no updates.

Uses an explicitly test-only reservation tokenizer. A diagnostic derivative is
not a supervised learning target or a claim of calibrated task probabilities.
"""
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
    from multimodal_jev.context_decision_contract import (
        ApplicabilityQuestion, ContextDecisionRequest, encode_context_request)
    from multimodal_jev.context_decision_head import ContextDecisionHead
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.slurp_training import content_hash

    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU only')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('pinned dependency changed: '+path)
    torch.set_num_threads(2)
    torch.manual_seed(317)
    processor = Qwen3ASRProcessor.from_pretrained(config['processor'], local_files_only=True)
    _, base = load_runtime(config['model'], 'cpu', 'bfloat16')
    token_id = config['decision_token_id']
    if token_id == base.config.timestamp_token_id:
        raise ValueError('timestamp collision')
    embeddings = base.get_input_embeddings().weight
    if token_id >= len(embeddings):
        raise ValueError('probe must not resize vocabulary')
    # Test-only row initialization, unchanged original model on disk. No claim
    # that a new marker has the semantics of these old Boolean label tokens.
    pair_ids = [processor.tokenizer.encode(v, add_special_tokens=False) for v in ('A','B')]
    if any(len(ids) != 1 for ids in pair_ids):
        raise ValueError('single initialization symbols required')
    before = tensor_identity(embeddings[token_id].detach())
    with torch.no_grad():
        embeddings[token_id].copy_(embeddings[[ids[0] for ids in pair_ids]].float().mean(0).to(embeddings.dtype))
    after = tensor_identity(embeddings[token_id].detach())
    adapted = initialize_adapter(base, dict(lora_rank=4, lora_alpha=8, lora_execution_dtype='bfloat16')).eval()
    head = ContextDecisionHead(1024, 8, 1024, residual_scale=.1).to(torch.bfloat16).eval()
    model = ContextDecisionModel(adapted.get_base_model().model, head, token_id).eval()
    calls = []
    captured = {}
    def capture(module, args, result):
        calls.append(1)
        captured['hidden'] = result.last_hidden_state.detach().clone()
    hook = model.backbone.register_forward_hook(capture)
    branch_path = output/'branch.pt'
    torch.save(head.state_dict(), branch_path)
    records = []
    for index, asset in enumerate(config['audio']):
        if file_hash(asset['audio']) != asset['sha256']:
            raise ValueError('audio changed')
        request = ContextDecisionRequest(asset['audio'], (
            ApplicabilityQuestion('request', 'An actionable request asks a system to perform an action.',
                                  'Does the audio contain an actionable request under this definition?'),
            ApplicabilityQuestion('alarm', 'The alarm tool creates or changes a timed alarm; calendar events are excluded.',
                                  'Does the audio express a request within the alarm tool definition?'),
            ApplicabilityQuestion('final', 'The caller allows the alarm tool only for actionable alarm requests.',
                                  'Is the alarm tool applicable under the supplied caller rule?')),
            ('Judge each supplied definition from the audio. Do not infer labels from question order.',),
            history=())
        wave, receipt = load_audio(asset['audio'], return_receipt=True)
        inputs, positions = encode_context_request(processor, request, wave,
            decision_token_id=token_id, audio_token_id=base.config.audio_token_id, max_tokens=4096)
        inputs = {k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
        before_calls = len(calls)
        start = time.monotonic()
        atomic_json(output/'progress.json', dict(phase='audio_forward', asset=index))
        with torch.set_grad_enabled(index == 0):
            logits = model.score_logits(inputs, positions)
        if len(calls)-before_calls != 1 or not torch.isfinite(logits).all():
            raise ValueError('one finite backbone forward required')
        probs = logits.sigmoid()
        snapshot = output/f'asset-{index}.pt'
        torch.save(dict(hidden=captured['hidden'], positions=positions,
                        valid_tokens=inputs['attention_mask'].bool(), logits=logits.detach(),
                        probabilities=probs.detach()), snapshot)
        # Only first audio backward: derivative of an explicitly arbitrary
        # diagnostic linear functional, no labels/reward/optimizer step.
        gradients = None
        if index == 0:
            atomic_json(output/'progress.json', dict(phase='diagnostic_backward', asset=index))
            weights = torch.tensor([[.2,-.3,.5]])
            (logits*weights).sum().backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError('missing or invalid gradient: '+name)
                    gradients[name] = float(parameter.grad.float().norm())
            branch_norms = [v for k,v in gradients.items() if k.startswith('decision_head.')]
            lora_b = [v for k,v in gradients.items() if '.lora_B.' in k]
            if not branch_norms or not all(v > 0 for v in branch_norms) or not lora_b or not any(v>0 for v in lora_b):
                raise ValueError('head and backbone LoRA gradient path required')
            model.zero_grad(set_to_none=True)
        restored = ContextDecisionHead(1024,8,1024,residual_scale=.1).to(torch.bfloat16).eval()
        restored.load_state_dict(torch.load(branch_path, weights_only=True), strict=True)
        with torch.no_grad():
            reload_probs = restored(captured['hidden'], positions, inputs['attention_mask'].bool())
        if not torch.equal(reload_probs, probs.detach()):
            raise ValueError('branch reload probability differs')
        records.append(dict(audio_id=asset['id'], audio_sha256=asset['sha256'],
            request=request.model_input(),request_sha256=content_hash(request.model_input()),
            audio_receipt=receipt,tokens=inputs['input_ids'].shape[-1],positions=positions.tolist(),
            input_ids=tensor_identity(inputs['input_ids']),one_backbone_forward=True,
            probabilities=probs.detach().tolist(),snapshot_sha256=file_hash(snapshot),
            gradients=gradients,exact_branch_reload=True,seconds=time.monotonic()-start))
        atomic_json(output/'records.json',records)
        del logits,probs,inputs
    hook.remove()
    return dict(records=records,branch_sha256=file_hash(branch_path),
        token_initialization=dict(token_id=token_id,before=before,after=after,method='frozen_mean_A_B_test_only'),
        training_dispatch_allowed=False,
        scope='Actual audio CPU forward with one-layer context head; arbitrary diagnostic derivative '
              'through fresh LoRA, no optimizer update, no gold labels, no calibrated/task gain/NPU/vLLM claims. '
              'Placeholder tokenizer and frozen marker row are test-only, not production registration.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
