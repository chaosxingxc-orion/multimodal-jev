#!/usr/bin/env python3
"""Twelve Jev NPU prefill boundaries; explicit CPU mode never loads weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA, ENGINE, VERSIONS, prior_records
from multimodal_jev.audio2tool_jev_engine import (
    CONTEXT_SHA, MANIFEST_SHA, accept_response, boundaries, prepare_request, validate_boundary)
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    from transformers import Qwen3OmniMoeProcessor
    start = time.monotonic()
    progress = lambda phase, **kw:atomic_json(args.output / 'progress.json', dict(phase=phase, heartbeat_unix=time.time(), **kw))
    versions = {key:importlib.metadata.version(key) for key in VERSIONS}
    if versions != VERSIONS or file_hash(args.asset_audit) != ASSET_AUDIT_SHA:
        raise ValueError('fixed inference stack and original model asset audit required')
    registry = args.registry.read_bytes()
    rows, bank, chosen = boundaries(args.context_run, args.release, registry)
    _, native_processor = prior_records(args.native_processor_run)
    audit = json.loads(args.asset_audit.read_text())
    if (audit['model'] != str(args.model) or audit['status'] != 'verified_inference_assets'
            or audit['repository'] != 'Qwen/Qwen3-Omni-30B-A3B-Instruct'
            or audit['revision'] != '26291f793822fb6be9555850f06dfe95f2d7e695'
            or not audit['all_publisher_files_present']):
        raise ValueError('original complete local Omni model required')
    def weights():
        for entry in audit['files']:
            if file_hash(args.model / entry['file']) != entry['sha256']:
                raise ValueError('model asset changed')
    for name, pin in native_processor['processor_assets'].items():
        if file_hash(args.model / name) != pin:
            raise ValueError('processor asset changed')
    if not args.preflight_only:
        progress('verify_weights'); weights()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    source = Path(__file__).resolve().parents[2]
    files = ['tools/jev/probe_audio2tool_jev_engine.py', 'src/multimodal_jev/audio2tool_jev_engine.py',
             'src/multimodal_jev/audio2tool_jev_runtime.py', 'src/multimodal_jev/audio2tool_jev.py',
             'src/multimodal_jev/audio2tool_audio.py', 'src/multimodal_jev/audio2tool_engine.py']
    write_once(args.output / 'identity.json', dict(versions=versions, engine=ENGINE,
        model=str(args.model), asset_audit_sha256=ASSET_AUDIT_SHA, context_result_sha256=CONTEXT_SHA,
        manifest_sha256=MANIFEST_SHA, sources={name:file_hash(source / name) for name in files},
        preflight_only=args.preflight_only, sampling=dict(temperature=0., max_tokens=1, n=1, seed=17),
        targets_read=False, weight_updates=0, full_roster_records=len(rows), boundary_cases=12))
    engine = None; records = []
    try:
        if not args.preflight_only:
            from vllm import LLM, SamplingParams
            progress('engine_load'); engine = LLM(model=str(args.model), **ENGINE)
            sampling = SamplingParams(temperature=0., max_tokens=1, n=1, seed=17)
        for row, case in chosen:
            progress('input', case=case['case'])
            request, receipts = prepare_request(processor, row, case['decision_id'], args.root, bank, registry)
            validate_boundary(request, receipts, case)
            record = dict(case=case['case'], id=row['id'], decision_id=case['decision_id'],
                          input_tokens=case['input_tokens'], audio_count=len(receipts), input_matches_processor=True)
            if engine is not None:
                begin = time.monotonic(); progress('engine_prefill', case=case['case'])
                record.update(accept_response(engine.generate([request], sampling, use_tqdm=False), case))
                record['engine_seconds'] = time.monotonic()-begin
            write_once(args.output / (case['case'].replace(':', '-') + '.json'), record); records.append(record)
            del request
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown(timeout=30)
    if not args.preflight_only:
        progress('verify_weights_after'); weights()
    write_once(args.output / 'result.json', dict(status='passed', records=records,
        identity_sha256=file_hash(args.output / 'identity.json'), elapsed_seconds=time.monotonic()-start,
        model_loaded=not args.preflight_only, targets_read=False, weight_updates=0,
        scope=('CPU engine-input preflight: twelve authenticated Jev waveform requests and prompts match prior real HF/vLLM processor cases. No weights or inference.'
               if args.preflight_only else 'Twelve real NPU Jev prefills and one generated token each. No complete JSON decode, full population inference, throughput claim, quality score or training.')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('root', 'model', 'release', 'registry', 'asset-audit', 'context-run', 'native-processor-run', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_once(args.output / 'terminal.json', dict(exit_code=1, error=repr(exc))); raise
    write_once(args.output / 'terminal.json', dict(exit_code=0))
