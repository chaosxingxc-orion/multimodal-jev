#!/usr/bin/env python3
"""Reconstruct original TRAIN6 logits from saved final decoder states.

This control evaluates only the original final norm and full-sequence output
projection. It neither reruns the decoder nor accepts a candidate actor repair.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


TRACE_RESULT = '23ee087e133334428c0e0fcc85b728a4f7459350954b32ef7a22b1b6a284a161'
PRECISION_IDENTITY = 'e39bdebee8d2d434df8a51efc9d57b4f697462f0a67e8577d72f7f2e97cae9dd'
HF_SOURCE = 'a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'


def project_response(hidden, norm, head, response_length):
    """Preserve the original GEMM row count before slicing response positions."""
    import torch

    if (hidden.ndim != 3 or hidden.dtype != torch.bfloat16
            or hidden.shape[1] <= response_length or response_length < 1):
        raise ValueError('complete BF16 sequence including response required')
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        normalized = norm(hidden)
        full_logits = head(normalized)
        response = full_logits[:, -response_length-1:-1].detach().clone()
    if not torch.isfinite(response).all():
        raise ValueError('nonfinite response logits')
    return response


def run(args):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf

    torch.set_num_threads(16)
    begin = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    if file_hash(args.trace/'result.json') != TRACE_RESULT:
        raise ValueError('original accepted full layer trace required')
    if file_hash(args.precision/'identity.json') != PRECISION_IDENTITY:
        raise ValueError('original precision identity required')
    if file_hash(inspect.getfile(hf)) != HF_SOURCE:
        raise ValueError('original runtime model implementation required')
    trace = read(args.trace/'result.json')
    identity = read(args.trace/'identity.json')
    precision = read(args.precision/'identity.json')
    if (identity['precision_identity_sha256'] != PRECISION_IDENTITY
            or identity['keys'] != precision['keys']
            or identity['response_ids'] != precision['response_ids']
            or identity['response_ids'] != [3849, 151645]
            or len(trace['reports']) != 6):
        raise ValueError('complete original TRAIN6 identity required')
    pins = {r['file']: r['sha256'] for r in identity['model_files']}
    verified = set()

    def verify(name):
        if file_hash(args.model/name) != pins[name]:
            raise ValueError('model file changed: '+name)
        verified.add(name)

    for name in ('config.json', 'model.safetensors.index.json'):
        verify(name)
    config = read(args.model/'config.json')['thinker_config']['text_config']
    index = read(args.model/'model.safetensors.index.json')['weight_map']
    names = ['thinker.model.norm.weight', 'thinker.lm_head.weight']
    weights = {}
    for shard in sorted({index[n] for n in names}):
        verify(shard)
        with safe_open(args.model/shard, framework='pt', device='cpu') as f:
            weights.update({n: f.get_tensor(n) for n in names if index[n] == shard})
    with torch.device('meta'):
        norm = hf.Qwen3OmniMoeTextRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        head = torch.nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
    norm.load_state_dict({'weight': weights[names[0]]}, strict=True, assign=True)
    head.load_state_dict({'weight': weights[names[1]]}, strict=True, assign=True)
    for module in (norm, head):
        module.eval().requires_grad_(False)
        if any(p.dtype != torch.bfloat16 or p.device.type != 'cpu' for p in module.parameters()):
            raise ValueError('original BF16 CPU weights required')
    atomic_json(args.output/'identity.json', dict(
        trace_result_sha256=TRACE_RESULT, trace_identity_sha256=file_hash(args.trace/'identity.json'),
        precision_identity_sha256=PRECISION_IDENTITY, hf_source_sha256=HF_SOURCE,
        runner_sha256=file_hash(__file__), keys=identity['keys'], response_ids=identity['response_ids'],
        projection='original BF16 final norm and full-sequence lm_head before slicing',
        model_files={n: pins[n] for n in sorted(verified)}, targets_read=False, weight_updates=0))
    labels = torch.tensor(identity['response_ids']).unsqueeze(-1)
    records = []
    for row in range(6):
        atomic_json(args.output/'progress.json', dict(phase='output_head', row=row, heartbeat_unix=time.time()))
        report = trace['reports'][row]
        if report['row'] != row or report['key'] != identity['keys'][row]:
            raise ValueError('reference row identity differs')
        state_path = args.trace/f'row-{row:02d}-trace/layer-47.output.pt'
        state_pin = report['reference']['layer-47.output']['sha256']
        if file_hash(state_path) != state_pin:
            raise ValueError('original final decoder state changed')
        record_path = args.precision/f'row-{row:02d}-audio_padding.json'
        expected_pin = identity['dependency']['logits_audit']['record_sha256'][record_path.name]
        if file_hash(record_path) != expected_pin:
            raise ValueError('original logits record changed')
        original = read(record_path)
        logits_path = args.precision/f'row-{row:02d}-audio_padding-logits.pt'
        if file_hash(logits_path) != original['logits_sha256']:
            raise ValueError('original response logits changed')
        hidden = torch.load(state_path, map_location='cpu', weights_only=True)
        reference = torch.load(logits_path, map_location='cpu', weights_only=True)
        logits = project_response(hidden.unsqueeze(0), norm, head, len(labels))[0]
        if logits.shape != reference.shape or logits.dtype != reference.dtype:
            raise ValueError('reference shape or dtype differs')
        logp = logits.float().log_softmax(-1).gather(-1, labels).squeeze(-1)
        ref_logp = reference.float().log_softmax(-1).gather(-1, labels).squeeze(-1)
        path = args.output/f'row-{row:02d}-logits.pt'
        torch.save(logits, path)
        result = dict(row=row, key=identity['keys'][row], input_shape=[1, *hidden.shape],
            state_sha256=state_pin, original_record_sha256=expected_pin,
            original_logits_sha256=original['logits_sha256'], logits_sha256=file_hash(path),
            logits_equal=bool(torch.equal(logits, reference)),
            logits_max_error=float((logits.float()-reference.float()).abs().max()),
            response_log_probs=logp.tolist(), response_logprob_max_error=float((logp-ref_logp).abs().max()))
        atomic_json(args.output/f'row-{row:02d}.json', result)
        records.append(result)
    for name in sorted(verified):
        verify(name)
    atomic_json(args.output/'result.json', dict(status='diagnostic_complete', records=records,
        exact_reference_reconstruction=all(r['logits_equal'] for r in records),
        identity_sha256=file_hash(args.output/'identity.json'), elapsed_seconds=time.monotonic()-begin,
        weight_updates=0, scope='Original final norm/head control only; no candidate, full actor, NPU or optimizer acceptance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'trace', 'precision', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))
