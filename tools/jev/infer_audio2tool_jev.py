#!/usr/bin/env python3
"""Full 30733x457 independent audio-to-Jev generation, no automatic replay."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA, ENGINE, VERSIONS, prior_records
from multimodal_jev.audio2tool_jev_capacity import MODEL, require_capacity
from multimodal_jev.audio2tool_jev_engine import CONTEXT_SHA, MANIFEST_SHA
from multimodal_jev.audio2tool_jev_inference import export, generate_all, population
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    registry = args.registry.read_bytes()
    rows, bank, budgets, profiles = population(args.release, args.context_run, registry, args.inputs, args.budgets)
    total = len(rows) * len(bank['decisions'])
    if args.preflight_only:
        write_once(args.output / 'result.json', dict(status='population_preflight_passed', records=len(rows),
            decisions=total, profiles=len(profiles), manifest_sha256=MANIFEST_SHA, context_sha256=CONTEXT_SHA,
            model_loaded=False, targets_read=False,
            scope='Complete paired input/budget/profile roster only; no waveform, feature, engine, capacity or score acceptance.'))
        return
    capacity = require_capacity(args.capacity_root, args.capacity_intent_hash, args.capacity_inspection,
                                args.capacity_inspection_hash, args.capacity_container_id,
                                args.context_run, args.release, registry)
    versions = {k:importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS or str(args.model) != MODEL or file_hash(args.asset_audit) != ASSET_AUDIT_SHA:
        raise ValueError('fixed engine stack and original local Omni model required')
    audit = json.loads(args.asset_audit.read_text())
    if audit['model'] != MODEL:
        raise ValueError('original full model asset manifest required')
    def weights():
        for entry in audit['files']:
            if file_hash(args.model / entry['file']) != entry['sha256']:
                raise ValueError('model asset changed')
    start = time.monotonic()
    progress = lambda phase, **kw:atomic_json(args.output / 'progress.json',dict(phase=phase, heartbeat_unix=time.time(), **kw))
    progress('verify_weights'); weights()
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    import torch
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    _, processor_identity = prior_records(args.native_processor_run)
    if file_hash(inspect.getfile(type(processor))) != processor_identity['hf_processor_source_sha256']:
        raise ValueError('accepted actual HF processor source required')
    repo = Path(__file__).resolve().parents[2]
    sources = {str(p.relative_to(repo)):file_hash(p) for p in sorted((repo / 'src/multimodal_jev').glob('*.py'))}
    sources['tools/jev/infer_audio2tool_jev.py'] = file_hash(__file__)
    write_once(args.output / 'identity.json', dict(manifest_sha256=MANIFEST_SHA, context_sha256=CONTEXT_SHA,
        inputs_sha256=file_hash(args.inputs), budgets_sha256=file_hash(args.budgets),
        model=str(args.model), model_files=audit['files'], asset_audit_sha256=ASSET_AUDIT_SHA,
        versions=versions, engine=ENGINE, sampling=dict(temperature=0., max_tokens=1024, n=1, seed=17),
        capacity=capacity, sources=sources, records=len(rows), decisions=total,
        parameters_frozen=True, targets_read=False, shard_decisions=4096,
        duplicate_policy='independent_generation_for_every_release_record_and_decision'))
    engine = None
    try:
        progress('engine_load'); engine = LLM(model=str(args.model), **ENGINE)
        params = SamplingParams(temperature=0., max_tokens=1024, n=1, seed=17)
        summary = generate_all(processor, engine, params, rows, bank, budgets, profiles,
                               args.root, registry, args.output / 'ledger', progress,
                               lambda x:bool(torch.isfinite(x).all()))
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown(timeout=30)
    progress('verify_weights_after', done=total, total=total); weights()
    projection = export(args.output / 'ledger', rows, bank, args.output / 'predictions.jsonl')
    write_once(args.output / 'result.json', dict(status='passed', ledger=summary, projection=projection,
        identity_sha256=file_hash(args.output / 'identity.json'), model_files_unchanged=True,
        weight_updates=0, targets_read=False, elapsed_seconds=time.monotonic()-start,
        scope='All independent audio Jev generations; raw outputs and truncations retained. Separate actual-container/audio/token acceptance and scoring still required.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('release', 'context-run', 'registry', 'inputs', 'budgets', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('root', 'model', 'asset-audit', 'native-processor-run', 'capacity-root', 'capacity-inspection'):
        parser.add_argument('--' + key, type=Path)
    for key in ('capacity-intent-hash', 'capacity-inspection-hash', 'capacity-container-id'):
        parser.add_argument('--' + key)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not args.preflight_only and any(getattr(args,k.replace('-','_')) is None for k in (
        'root','model','asset-audit','native-processor-run','capacity-root','capacity-inspection',
        'capacity-intent-hash','capacity-inspection-hash','capacity-container-id')):
        parser.error('actual generation requires all original model and coordinator-pinned capacity arguments')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_once(args.output / 'terminal.json', dict(exit_code=1, error=repr(exc))); raise
    write_once(args.output / 'terminal.json', dict(exit_code=0))
