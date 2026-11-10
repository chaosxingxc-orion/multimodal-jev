#!/usr/bin/env python3
"""Real waveform-loader check against registered full-roster context budgets."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_audio import load_audio
from multimodal_jev.baseline import atomic_json, file_hash

INPUT_SHA='f7e4ebe4e7cf2cf411f3f26a8714bbdd0a493c0c7aeb98dd4199112d197e1a35'
BUDGET_SHA='87e750243a3dcc56ca006e9f661dcaa2ac130cc27fd7312d6e986892aa1ef223'


def run(args):
    start=time.monotonic()
    if file_hash(args.inputs)!=INPUT_SHA or file_hash(args.budgets)!=BUDGET_SHA:
        raise ValueError('registered complete inputs and context budgets required')
    rows=[json.loads(l) for l in args.inputs.open()]
    budgets=[json.loads(l) for l in args.budgets.open()]
    if len(rows)!=30733 or [r['id'] for r in rows]!=[r['id'] for r in budgets]:
        raise ValueError('complete ordered roster required')
    selected={}
    for row,budget in zip(rows,budgets,strict=True):
        mode=row['input']['mode']
        if mode not in selected or budget['input_tokens']>selected[mode][1]['input_tokens']:
            selected[mode]=(row,budget)
    results=[]
    for mode,(row,budget) in sorted(selected.items()):
        waves,receipts=load_audio(row['input'],args.root)
        if [r['expected_frames'] for r in receipts]!=budget['frames']:
            raise ValueError('waveform loader differs from accepted HF frame budget')
        results.append(dict(id=row['id'],mode=mode,audios=receipts))
        del waves
    repo=Path(__file__).resolve().parents[2]
    return dict(status='passed',inputs_sha256=INPUT_SHA,budgets_sha256=BUDGET_SHA,
        cases=results,elapsed_seconds=time.monotonic()-start,
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','soundfile')},
        source_files={p:file_hash(repo/p) for p in ('tools/jev/check_audio2tool_audio_loader.py',
            'src/multimodal_jev/audio2tool_audio.py','src/multimodal_jev/audio2tool_input.py',
            'src/multimodal_jev/audio2tool_dialogue_input.py','src/multimodal_jev/sfc_audio.py',
            'src/multimodal_jev/omni_waveform_boundary.py')},
        scope='Three deterministic mode maxima with real original waveform bytes, complete decoding and frame comparison to prior accepted HF budgets. Host CPU only; no model, HF processor or vLLM engine invocation; not full-population audio decoding or model performance.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','inputs','budgets','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous result')
    atomic_json(args.output,run(args))
