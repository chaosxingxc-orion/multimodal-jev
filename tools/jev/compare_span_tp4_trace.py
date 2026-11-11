#!/usr/bin/env python3
"""Authenticate diagnostic traces and compare the first fixed request offline."""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_trace import compare_tp4_traces
from summarize_span_tp4_diagnostic import summarize


def compare(root, prior):
    acceptance = summarize(root)
    original = summarize(prior)
    hf_launch=json.loads((root/'hf-launch.json').read_text())
    if (hf_launch['physical_devices']!=[4] or
            hf_launch['config_sha256']!=file_hash(root/'hf-config.json') or
            hf_launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json') or
            json.loads((root/'hf/terminal.json').read_text())['exit_code']!=0):
        raise ValueError('HF trace launch identity or terminal differs')
    hf_config=json.loads((root/'hf-config.json').read_text())
    for path,digest in hf_config['pins'].items():
        if file_hash(path)!=digest:
            raise ValueError('HF trace pinned input differs')
    if acceptance['adapter_id'] != original['adapter_id']:
        raise ValueError('trace and original policy differ')
    current = json.loads((root/'run/result.json').read_text())
    previous = json.loads((prior/'run/result.json').read_text())
    repeat_deltas=[]
    for a,b in zip(current['records'],previous['records'],strict=True):
        if a['request_id']!=b['request_id'] or a['request_identity']!=b['request_identity']:
            raise ValueError('traced rollout input differs from original')
        delta=float((torch.tensor(a['candidate_logprobs'])-torch.tensor(b['candidate_logprobs'])).abs().max())
        repeat_deltas.append(dict(id=a['id'],maximum_logprob_difference=delta))
    config = json.loads((prior/'config.json').read_text())
    old_hf = json.loads(Path(config['hf_results']).read_text())
    hf = json.loads((root/'hf/result.json').read_text())
    if hf['records']!=old_hf['records']:
        raise ValueError('traced HF differs from original outputs')
    hfr = json.loads((root/'hf/trace-request.json').read_text())
    vr = json.loads((root/'run/trace-request.json').read_text())
    if hfr['request_id']!=vr['request_id'] or hfr['request_id']!=hf['records'][0]['request_id']:
        raise ValueError('trace input identity differs')
    files={hfr['path']:hfr['sha256'],**vr['files']}
    for path,digest in files.items():
        if file_hash(path)!=digest:
            raise ValueError('trace file hash differs')
    workers=sorted(vr['workers'],key=lambda r:r['rank'])
    if [r['rank'] for r in workers]!=list(range(4)) or any(r['world_size']!=4 for r in workers):
        raise ValueError('four unique worker traces required')
    if len({r['path'] for r in workers})!=4:
        raise ValueError('worker trace paths collide')
    if set(vr['files'])!={r['path'] for r in workers}:
        raise ValueError('trace receipt file coverage differs')
    actor=torch.load(hfr['path'],map_location='cpu',weights_only=True)
    ranks=[torch.load(r['path'],map_location='cpu',weights_only=True) for r in workers]
    comparison=compare_tp4_traces(actor,ranks)
    points=comparison['reconstructed']
    first_boundary=next((name for name in [
        'audio_projected','layer.0.pre_norm','layer.0.input_norm',
        'layer.0.q_proj','layer.0.k_proj','layer.0.v_proj',
        'layer.0.o_proj.input','layer.0.o_proj.output','layer.0.post_attention_norm',
        'layer.0.down_proj.input','layer.0.down_proj.output','layer.0.mlp']
        if points[name]['maximum']!=0),None)
    return dict(root=str(root),request_id=hfr['request_id'],id=hfr['id'],
                unchanged_full48_outputs=all(r['maximum_logprob_difference']==0 for r in repeat_deltas),
                repeat_changed_records=sum(r['maximum_logprob_difference']!=0 for r in repeat_deltas),
                repeat_maximum_logprob_difference=max(r['maximum_logprob_difference'] for r in repeat_deltas),
                repeat_deltas=repeat_deltas,hf_full48_outputs_unchanged=True,training_admitted=False,
                first_nonzero_selected_boundary=first_boundary,
                comparison=comparison,trace_files=files,probability_summary=acceptance,
                scope='One fixed request layer diagnosis; full48 stability measured, not assumed; run variation versus trace effects unresolved; no training admission')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare(args.root,args.prior)
    atomic_json(args.output,report)
    print(json.dumps(dict(first_nonzero_selected_boundary=report['first_nonzero_selected_boundary'],
                         points={k:v for k,v in report['comparison']['reconstructed'].items()
                                 if k.startswith('layer.0.') or k in ('audio_projected','final_norm')}),indent=2))
