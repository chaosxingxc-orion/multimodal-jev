#!/usr/bin/env python3
"""Frozen full MMAR native generation, with a separate one-token capacity mode."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA, VERSIONS
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mmar_omni import INPUT_SHA, prepare_request
from multimodal_jev.mmar_inference import (ENGINE, PROCESSOR_RESULT_SHA, population,
    capacity_indices, ids_hash, validate_input, accept_response, admit_capacity)

MODEL = '/mnt/cephfs/xingchao/models/qwen3-omni-30b-a3b-instruct'


def run(args):
    rows, prior, processor_identity = population(args.inputs, args.processor_run)
    admission = None
    if not args.capacity_only:
        admission = admit_capacity(args.capacity_root, args.capacity_intent_hash,
            args.capacity_inspection, args.capacity_inspection_hash, rows, prior)
    if str(args.model) != MODEL or file_hash(args.asset_audit) != ASSET_AUDIT_SHA:
        raise ValueError('fixed original local model required')
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS or versions != processor_identity['versions']:
        raise ValueError('accepted fixed runtime versions required')
    audit = json.loads(args.asset_audit.read_text())
    if audit['model'] != MODEL:
        raise ValueError('original model asset audit required')
    def weights():
        for entry in audit['files']:
            if file_hash(args.model/entry['file']) != entry['sha256']:
                raise ValueError('model asset changed')
    def progress(phase, **kwargs):
        atomic_json(args.output/'progress.json', dict(phase=phase, heartbeat_unix=time.time(), **kwargs))
    start = time.monotonic(); progress('verify_weights'); weights()
    for name,pin in processor_identity['processor_assets'].items():
        if file_hash(args.model/name) != pin:
            raise ValueError('processor assets differ from accepted run')
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    selected = capacity_indices(prior) if args.capacity_only else list(range(len(rows)))
    budget = 1 if args.capacity_only else 256
    sampling = dict(temperature=0., max_tokens=budget, n=1, seed=17)
    repo = Path(__file__).resolve().parents[2]
    source = {str(p.relative_to(repo)): file_hash(p) for p in sorted((repo/'src/multimodal_jev').glob('*.py'))}
    source['tools/jev/infer_mmar_omni.py'] = file_hash(__file__)
    write_once(args.output/'identity.json', dict(engine=ENGINE, sampling=sampling, versions=versions,
        model=MODEL, model_files=audit['files'], asset_audit_sha256=ASSET_AUDIT_SHA,
        processor_result_sha256=PROCESSOR_RESULT_SHA, inputs_sha256=INPUT_SHA,
        source_files=source, capacity=admission, selected_ids=[rows[i]['id'] for i in selected],
        parameters_frozen=True, targets_read=False))
    for name in ('attempts', 'records', 'failures'):
        (args.output/name).mkdir()
    engine = None; records = []
    try:
        progress('engine_load'); engine = LLM(model=MODEL, **ENGINE)
        params = SamplingParams(**sampling)
        for done,i in enumerate(selected):
            row = rows[i]; progress('input', done=done, total=len(selected), id=row['id'])
            request, receipt = prepare_request(processor, row, args.audio_root)
            wave = request['multi_modal_data']['audio'][0][0]
            batch = processor(text=[request['prompt']], audio=[wave], sampling_rate=16000,
                              return_tensors='pt', padding=True, truncation=False)
            ids = batch['input_ids'][0].tolist()
            validate_input(request, receipt, ids, batch['feature_attention_mask'].sum(-1).tolist(), prior[i])
            if not torch.isfinite(batch['input_features']).all():
                raise ValueError('nonfinite audio features')
            del batch
            attempt = dict(id=row['id'], input_tokens=len(ids), input_ids_sha256=ids_hash(ids), audio=receipt,
                request_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
                prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest())
            name = f'{i:04d}.json'
            write_once(args.output/'attempts'/name, attempt)
            progress('generation', done=done, total=len(selected), id=row['id']); begin = time.monotonic()
            try:
                responses = engine.generate([request], params, use_tqdm=False)
                # Save outputs even when subsequent protocol validation rejects them.
                raw = [dict(finished=r.finished, prompt_token_ids=list(r.prompt_token_ids),
                    outputs=[dict(text=a.text, token_ids=list(a.token_ids), finish_reason=a.finish_reason,
                                  stop_reason=a.stop_reason) for a in r.outputs]) for r in responses]
                write_once(args.output/'records'/name, dict(attempt=attempt, responses=raw, seconds=time.monotonic()-begin))
                response = accept_response(responses, ids, budget)
            except Exception as exc:
                write_once(args.output/'failures'/name, dict(id=row['id'], error=repr(exc), seconds=time.monotonic()-begin))
                raise
            records.append(dict(**attempt, **response, error=None, seconds=time.monotonic()-begin))
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown(timeout=30)
    progress('verify_weights_after', done=len(records), total=len(selected)); weights()
    with (args.output/'predictions.jsonl').open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    write_once(args.output/'result.json', dict(status='passed', mode='capacity' if args.capacity_only else 'full',
        records=len(records), identity_sha256=file_hash(args.output/'identity.json'),
        predictions_sha256=file_hash(args.output/'predictions.jsonl'),
        elapsed_seconds=time.monotonic()-start, weight_updates=0, targets_read=False,
        length_limited=sum(r['length_limited'] for r in records), model_files_unchanged=True,
        scope='Input-maxima one-token capacity only' if args.capacity_only else
              'All 1000 original MMAR inputs, frozen audio-to-answer. External container acceptance and task scoring remain separate.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('inputs', 'processor-run', 'audio-root', 'model', 'asset-audit', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--capacity-only', action='store_true')
    for key in ('capacity-root', 'capacity-inspection'):
        p.add_argument('--'+key, type=Path)
    for key in ('capacity-intent-hash', 'capacity-inspection-hash'):
        p.add_argument('--'+key)
    args = p.parse_args()
    if not args.capacity_only and any(getattr(args,k) is None for k in (
            'capacity_root','capacity_inspection','capacity_intent_hash','capacity_inspection_hash')):
        p.error('full run requires actual inspected successful capacity evidence')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time())); raise
    write_once(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
