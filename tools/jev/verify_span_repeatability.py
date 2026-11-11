#!/usr/bin/env python3
"""Recompute same-engine stability, retaining failures as diagnostic evidence."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_repeatability import compare_repeats
from summarize_span_tp4_diagnostic import summarize


def verify(root):
    summary=summarize(root)
    config=json.loads((root/'config.json').read_text())
    if config.get('trace_repeatability_probe') is not True:
        raise ValueError('repeat diagnostic config required')
    workers_config=json.loads((root/'run/worker.json').read_text())
    expected_env=config.get('expected_collective_environment')
    if expected_env is not None and any(w.get('collective_environment')!=expected_env for w in workers_config):
        raise ValueError('worker collective environment differs')
    repeat=json.loads((root/'run/repeatability.json').read_text())
    result=json.loads((root/'run/result.json').read_text())
    row=result['records'][0]
    if repeat['request_id']!=row['request_id'] or repeat['request_identity']!=row['request_identity']:
        raise ValueError('repeat/suite input mismatch')
    if repeat['mm_processor_cache_gb']!=0 or repeat['unique_audio_cache_keys'] is not True:
        raise ValueError('uncached repeat controls missing')
    comparison=compare_repeats(repeat['records'])
    if comparison!=repeat['comparison']:
        raise ValueError('repeat comparison mismatch')
    traces=json.loads((root/'run/repeat-trace-files.json').read_text())
    workers=sorted(traces['workers'],key=lambda r:r['rank'])
    if ([r['rank'] for r in workers]!=list(range(4)) or
        len(set(r['path'] for r in workers))!=4 or
        set(traces['files'])!={r['path'] for r in workers}):
        raise ValueError('four distinct trace files required')
    for path,digest in traces['files'].items():
        if file_hash(path)!=digest:
            raise ValueError('trace file changed')
        trace=torch.load(path,map_location='cpu',weights_only=True)
        if any(k not in trace for k in ('audio.conv2d1.input','audio_projected','layer.0.input_norm','final_norm')):
            raise ValueError('full real audio trace missing; cached execution not accepted')
    first=torch.tensor(repeat['records'][0]['candidate_logits'])
    sixth=torch.tensor(row['candidate_logits'])
    return dict(root=str(root),request_id=row['request_id'],comparison=comparison,
        collective_environment=[w.get('collective_environment') for w in workers_config],
        sixth_plain_request_exact=bool(torch.equal(first,sixth)),
        sixth_plain_maximum_logit_difference=float((first-sixth).abs().max()),
        full_suite=summary,training_admitted=False,
        files={str(p):file_hash(p) for p in (root/'run/repeatability.json',root/'run/repeat-trace-files.json')},
        scope='One fixed request, five controlled repeats plus suite sixth call; one engine only, no cross-engine or HF equivalence claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=verify(args.root);atomic_json(args.output,result);print(json.dumps(result,indent=2))
