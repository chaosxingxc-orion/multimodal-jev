#!/usr/bin/env python3
"""Measure saved FP32-expert decoder states with the verified original head.

No decoder replay, optimizer, generation or full-actor acceptance. The original
0.1 response-logprob gate is retained, and full-vocabulary differences are saved.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from diagnose_omni_output_head import project_response, HF_SOURCE


def compare_logits(actual, reference, response_ids):
    import torch

    if (actual.shape != reference.shape or actual.ndim != 2
            or actual.shape[0] != len(response_ids)
            or actual.dtype != reference.dtype or actual.dtype != torch.bfloat16
            or not torch.isfinite(actual).all() or not torch.isfinite(reference).all()):
        raise ValueError('matching finite full-vocabulary BF16 response logits required')
    label = torch.tensor(response_ids).unsqueeze(-1)
    a = actual.float().log_softmax(-1)
    b = reference.float().log_softmax(-1)
    ap = a.gather(-1, label).squeeze(-1)
    bp = b.gather(-1, label).squeeze(-1)
    error = float((ap-bp).abs().max())
    return dict(logits_equal=bool(torch.equal(actual, reference)),
                logits_max_error=float((actual.float()-reference.float()).abs().max()),
                full_vocab_logprob_max_error=float((a-b).abs().max()),
                response_log_probs=ap.tolist(), reference_log_probs=bp.tolist(),
                response_logprob_max_error=error, within_original_atol=error <= .1)


def run(args):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf

    torch.set_num_threads(16)
    begin = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    for root, pin in [(args.decoder, args.decoder_audit_hash), (args.control, args.control_audit_hash)]:
        if file_hash(root/'diagnostic-audit.json') != pin:
            raise ValueError('coordinator-pinned independent terminal audit required')
        audit = read(root/'diagnostic-audit.json')
        if file_hash(root/'run/result.json') != audit['result_sha256']:
            raise ValueError('audited terminal result changed')
        if file_hash(root/'run/identity.json') != audit['identity_sha256']:
            raise ValueError('audited runtime identity changed')
        if read(root/'run/terminal.json')['exit_code'] != 0:
            raise ValueError('successful terminal required')
    control = read(args.control/'diagnostic-audit.json')
    if not control['exact_reference_reconstruction'] or len(control['records']) != 6:
        raise ValueError('all six original head reconstructions must be exact')
    control_plan = read(args.control/'intent.json')
    if file_hash(inspect.getfile(project_response)) != control_plan['source_files']['tools/jev/diagnose_omni_output_head.py']:
        raise ValueError('verified original projection implementation required')
    if file_hash(inspect.getfile(hf)) != HF_SOURCE:
        raise ValueError('original HF implementation required')
    decoder = read(args.decoder/'run/result.json')
    di = read(args.decoder/'run/identity.json')
    ci = read(args.control/'run/identity.json')
    if (decoder['valid_difference_found'] or decoder['first_differing_layer'] is not None
            or [r['layer'] for r in decoder['layers']] != list(range(48))
            or di['fp32_projections'] is not True or di['start_layer'] != 0
            or di['order'] != [2, 0, 5, 1, 3, 4]
            or ci['response_ids'] != [3849, 151645]):
        raise ValueError('complete batch-equal candidate decoder required')
    final_path = args.decoder/'run/layer-47.json'
    if file_hash(final_path) != decoder['layers'][-1]['report_sha256']:
        raise ValueError('final layer report changed')
    final = read(final_path)
    expected = [(s, o) for s in (1, 2) for o in range(0, 6, s)]
    if [(r['size'], r['offset']) for r in final['saved']] != expected:
        raise ValueError('all nine original calls required')
    pins = ci['model_files']
    for name, pin in pins.items():
        if file_hash(args.model/name) != pin:
            raise ValueError('original head model files changed')
    config = read(args.model/'config.json')['thinker_config']['text_config']
    index = read(args.model/'model.safetensors.index.json')['weight_map']
    names = ['thinker.model.norm.weight', 'thinker.lm_head.weight']
    weights = {}
    for shard in sorted({index[n] for n in names}):
        if shard not in pins:
            raise ValueError('unverified model shard')
        with safe_open(args.model/shard, framework='pt', device='cpu') as f:
            weights.update({n: f.get_tensor(n) for n in names if index[n] == shard})
    with torch.device('meta'):
        norm = hf.Qwen3OmniMoeTextRMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        head = torch.nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
    norm.load_state_dict({'weight': weights[names[0]]}, strict=True, assign=True)
    head.load_state_dict({'weight': weights[names[1]]}, strict=True, assign=True)
    for m in (norm, head):
        m.eval().requires_grad_(False)
        if any(p.dtype != torch.bfloat16 or p.device.type != 'cpu' for p in m.parameters()):
            raise ValueError('original BF16 CPU weights required')
    references = {}
    for row, rec in enumerate(control['records']):
        p = args.control/f'run/row-{row:02d}-logits.pt'
        if rec['row'] != row or file_hash(p) != rec['logits_sha256'] or not rec['logits_equal']:
            raise ValueError('verified original logits changed')
        references[row] = torch.load(p, map_location='cpu', weights_only=True)
    atomic_json(args.output/'identity.json', dict(
        decoder_audit_sha256=args.decoder_audit_hash, control_audit_sha256=args.control_audit_hash,
        decoder_result_sha256=file_hash(args.decoder/'run/result.json'),
        control_result_sha256=file_hash(args.control/'run/result.json'),
        runner_sha256=file_hash(__file__), projection_source_sha256=file_hash(inspect.getfile(project_response)),
        order=di['order'], response_ids=ci['response_ids'], keys=ci['keys'], atol=.1,
        model_files=pins, weight_updates=0, targets_read=False,
        scope='Saved candidate full padded sequence -> original final norm/head; no full actor acceptance.'))
    captured = {}
    records = []
    for rec in final['saved']:
        size, offset = rec['size'], rec['offset']
        atomic_json(args.output/'progress.json', dict(phase='candidate_head', size=size, offset=offset, heartbeat_unix=time.time()))
        p = args.decoder/'run'/rec['file']
        if file_hash(p) != rec['sha256']:
            raise ValueError('candidate state changed')
        state = torch.load(p, map_location='cpu', weights_only=True)
        mask = state['token_mask']
        hidden = state['stages']['output']
        if (hidden.shape[:2] != mask.shape or hidden.shape[0] != size
                or not torch.all((mask == 0) | (mask == 1))
                or not torch.all(mask[:, -3:] == 1)
                or not torch.all(mask[:, 1:] >= mask[:, :-1])):
            raise ValueError('original left-padded sequence with valid response required')
        logits = project_response(hidden, norm, head, len(ci['response_ids']))
        path = args.output/f'micro-{size}-offset-{offset}-logits.pt'
        torch.save(logits, path)
        comparisons = []
        for j, row in enumerate(di['order'][offset:offset+size]):
            captured[size, row] = logits[j].clone()
            comparisons.append(dict(row=row, **compare_logits(logits[j], references[row], ci['response_ids'])))
        record = dict(size=size, offset=offset, input_shape=list(hidden.shape),
            input_file=rec['file'], input_sha256=rec['sha256'],
            logits_file=path.name, logits_sha256=file_hash(path), comparisons=comparisons)
        atomic_json(args.output/f'micro-{size}-offset-{offset}.json', record)
        records.append(record)
        del state, hidden, logits
    between = [dict(row=row, **compare_logits(captured[2, row], captured[1, row], ci['response_ids'])) for row in di['order']]
    reference_pass = all(c['within_original_atol'] for r in records for c in r['comparisons'])
    batch_pass = all(c['within_original_atol'] for c in between)
    for name, pin in pins.items():
        if file_hash(args.model/name) != pin:
            raise ValueError('head model files changed during diagnostic')
    atomic_json(args.output/'result.json', dict(status='diagnostic_complete', records=records,
        between=between, reference_within_original_atol=reference_pass,
        batch_within_original_atol=batch_pass, numerical_gate_passed=reference_pass and batch_pass,
        identity_sha256=file_hash(args.output/'identity.json'), elapsed_seconds=time.monotonic()-begin,
        atol=.1, weight_updates=0, scope='Candidate saved-state output-head diagnostic, not actual verl/NPU/vLLM/optimizer acceptance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'decoder', 'control', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('decoder-audit-hash', 'control-audit-hash'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))
