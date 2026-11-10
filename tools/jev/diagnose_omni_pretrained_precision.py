#!/usr/bin/env python3
"""Isolate CPU precision and padding effects on the same complete TRAIN6 roster.

Diagnostic only: frozen weights, synthetic response, no targets or optimizer.
Each cell changes one factor from its predecessor; save response logits so
normalization can be audited without another expensive model forward.
"""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_actor_batch import prepare, evidence
from probe_omni_pretrained_verl import prior_evidence, read
from probe_omni_pretrained_forward import loading_report
from sample_sfc_omni_learning import load_inputs
from infer_qwen3_omni_probe import verify_assets
from verify_omni_pretrained_forward import check_records


CELLS = (
    ('reference', False, False, False, False),
    ('autocast', True, False, False, False),
    ('fp32_features', True, True, False, False),
    ('audio_padding', True, True, True, False),
    ('text_padding', True, True, True, True),
)


def cell_inputs(item, response, *, prompt_width, audio_width, fp32_features,
                audio_padding, text_padding):
    import torch
    prompt = item['input_ids']
    pwidth = prompt_width if text_padding else len(prompt)
    awidth = audio_width if audio_padding else item['input_features'].shape[-1]
    if pwidth < len(prompt) or awidth < item['input_features'].shape[-1]:
        raise ValueError('diagnostic cannot truncate inputs')
    # Left padding is masked. Its token value is supplied by the caller.
    ids = torch.cat((torch.zeros(pwidth-len(prompt), dtype=prompt.dtype), prompt,
                     torch.tensor(response))).unsqueeze(0)
    mask = torch.ones_like(ids); mask[:, :pwidth-len(prompt)] = 0
    pos = mask.cumsum(-1)-1; pos.masked_fill_(mask == 0, 1)
    audio = torch.nn.functional.pad(item['input_features'],
                                    (0, awidth-item['input_features'].shape[-1]))
    feature_mask = torch.nn.functional.pad(item['feature_attention_mask'],
                                           (0, awidth-item['input_features'].shape[-1]))
    return dict(input_ids=ids, attention_mask=mask, position_ids=pos,
                input_features=audio.to(torch.float32 if fp32_features else torch.bfloat16),
                feature_attention_mask=feature_mask, use_cache=False, return_dict=True)


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
    torch.set_num_threads(16); torch.manual_seed(17)
    start = time.monotonic(); out = args.output
    def progress(phase, **kw):
        atomic_json(out/'progress.json', dict(phase=phase, heartbeat_unix=time.time(), **kw))
    _, prior, prior_identity = prior_evidence(args.prior, args.acceptance,
        args.acceptance_hash, args.inspection, args.inspection_hash)
    cfg = read(args.prior/'config.json'); base, rows, _, checks = load_inputs(cfg)
    if {k: importlib.metadata.version(k) for k in base['versions']} != base['versions']:
        raise ValueError('fixed runtime required')
    progress('verify_weights'); assets = verify_assets(base)
    if assets['files'] != prior_identity['model_files']:
        raise ValueError('prior checkpoint identity changed')
    processor = Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'], local_files_only=True)
    response = processor.tokenizer.encode('false', add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    check_records(prior['responses'], rows, checks, response)
    progress('prepare_audio'); prepared = prepare(processor, rows)
    for key, values in evidence(prepared).items():
        if any(checks[key][name] != value for name, value in values.items()):
            raise ValueError('registered prompt differs')
    keys = [(r['id'], r['input']['decisions'][0]['id']) for r in rows]
    items = [prepared[k] for k in keys]
    pwidth = max(len(i['input_ids']) for i in items)
    awidth = max(i['input_features'].shape[-1] for i in items)
    if pwidth+len(response)>4096 or processor.tokenizer.pad_token_id is None:
        raise ValueError('bounded context and pad ID required')
    atomic_json(out/'identity.json', dict(config=cfg, keys=keys, cells=CELLS,
        response_ids=response, prompt_width=pwidth, audio_width=awidth,
        prior_acceptance_sha256=args.acceptance_hash, model_files=assets['files'],
        targets_read=False, weight_updates=0, preflight_only=args.preflight_only,
        runner_sha256=file_hash(__file__)))
    if args.preflight_only:
        for item in items:
            for _, _, fp32, apad, tpad in CELLS:
                cell_inputs(item, response, prompt_width=pwidth, audio_width=awidth,
                            fp32_features=fp32, audio_padding=apad, text_padding=tpad)
        atomic_json(out/'result.json', dict(status='passed', preflight_only=True,
            model_loaded=False, rows=len(items), cells=len(CELLS))); return
    progress('load_thinker')
    model, loading = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        base['audio_model'], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map='cpu', attn_implementation='sdpa', output_loading_info=True)
    loading = loading_report(loading); atomic_json(out/'loading-info.json', loading)
    if (any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs'))
            or any(not k.startswith(('talker.', 'code2wav.')) for k in loading.get('unexpected_keys', []))):
        raise ValueError('incomplete Thinker load')
    model.eval().requires_grad_(False)
    if any(p.is_meta or p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('materialized CPU model required')
    records = []
    for i, item in enumerate(items):
        previous = None
        for name, autocast, fp32, apad, tpad in CELLS:
            progress('forward', row=i, cell=name)
            inputs = cell_inputs(item, response, prompt_width=pwidth, audio_width=awidth,
                                 fp32_features=fp32, audio_padding=apad, text_padding=tpad)
            inputs['input_ids'].masked_fill_(inputs['attention_mask']==0, processor.tokenizer.pad_token_id)
            begin = time.monotonic()
            with torch.inference_mode(), (torch.autocast('cpu', dtype=torch.bfloat16) if autocast else nullcontext()):
                output = model(**inputs)
                logits = output.logits[0, -len(response)-1:-1].detach().clone()
                del output
            label = torch.tensor(response).unsqueeze(-1)
            low = logits.log_softmax(-1).gather(-1, label).squeeze(-1).float()
            high = logits.float().log_softmax(-1).gather(-1, label).squeeze(-1)
            if not torch.isfinite(high).all() or (high>0).any():
                raise ValueError('invalid diagnostic probability')
            artifact = out/f'row-{i:02d}-{name}-logits.pt'
            torch.save(logits, artifact)
            record = dict(row=i, key=keys[i], cell=name, seconds=time.monotonic()-begin,
                fp32_log_probs=high.tolist(), native_dtype_log_probs=low.tolist(),
                logits_dtype=str(logits.dtype), logits_sha256=file_hash(artifact),
                normalization_max_error=float((high-low).abs().max()),
                predecessor_max_error=None if previous is None else float((high-previous).abs().max()),
                prior_unpadded_max_error=float((high-torch.tensor(prior['responses'][i]['response_log_probs'])).abs().max()))
            atomic_json(out/f'row-{i:02d}-{name}.json', record)
            records.append(record); previous=high
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('unexpected trainable model')
    progress('verify_weights_after'); verify_assets(base)
    atomic_json(out/'result.json', dict(status='diagnostic_complete', records=records,
        weight_updates=0, elapsed_seconds=time.monotonic()-start,
        scope='Complete TRAIN6 five-cell CPU diagnostic; no automatic parity acceptance, generation, training or replay.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('prior','acceptance','inspection','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('acceptance-hash','inspection-hash'):p.add_argument('--'+key,required=True)
    p.add_argument('--preflight-only',action='store_true'); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try: run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc))); raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
