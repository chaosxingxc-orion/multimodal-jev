"""Compare installed graphs at equal real anchor coverage, with stage timings."""
import hashlib
import json
import time
from pathlib import Path

import torch

from multimodal_jev.card_contrast import Directions, components
from multimodal_jev.card_training_acoustics import collate
from multimodal_jev.qwen3_asr_provider import Qwen3ASR
from tools.jev.card_real_bucket_panel import read, render


def compare(spec, lane, c, wrapped, base, processor, model, device, profile):
    graph = profile['components'].get('training_graph')
    if graph != lane['graph']:
        raise ValueError('installed graph differs from requested graph')
    panel = read(spec['panel'], spec['panel_sha256'])[lane['band']]
    width = lane['width']
    anchors = panel['anchors']
    objective = dict(c['instruction_objective'])
    for k, v in spec['contrast_samples'].items():
        objective[k + '_per_anchor'] = v
    d = Directions.from_objective(objective)
    if len(anchors) % width or any(len(a['pairs']) != d.width for a in anchors):
        raise ValueError('invalid cohort or contrast width')
    col = Qwen3ASR.collation(base, processor, model)
    inputs = []
    for start in range(0, len(anchors), width):
        chunk = anchors[start:start+width]
        rows, _ = render(c,dict(anchors=chunk),width,col['audio_token_id'],col['decision_token_id'])
        inp,pos = collate(rows,'probe',**col)
        if inp['input_ids'].shape != (width*d.width,max(a['expected'] for a in chunk)):
            raise ValueError('padding disagrees with real sample lengths')
        inputs.append(({k:v.to(device) for k,v in inp.items()},pos.to(device)))
    opt = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=c['lr_schedule']['peak_lr'])
    records = []
    for i in range(spec['warmup']+spec['repeats']):
        phases = [0.,0.,0.]
        digest = hashlib.sha256()
        for inp,pos in inputs:
            opt.zero_grad(set_to_none=True)
            torch.npu.synchronize()
            start = time.monotonic()
            z = wrapped(inp,pos,return_logits=True).reshape(width,d.width)
            loss = components(z,d)[:,0].mean()
            torch.npu.synchronize()
            forward = time.monotonic()
            loss.backward()
            torch.npu.synchronize()
            backward = time.monotonic()
            norm = wrapped.clip_grad_norm_(1.)
            opt.step()
            torch.npu.synchronize()
            end = time.monotonic()
            for j,t in enumerate((forward-start,backward-forward,end-backward)):
                phases[j] += t
            if not torch.isfinite(loss).item() or not torch.isfinite(norm).item():
                raise ValueError('nonfinite update')
            for v in (z,loss,norm):
                digest.update(v.detach().cpu().float().contiguous().numpy().tobytes())
        records.append(dict(seconds=phases, witness=digest.hexdigest()))
    print('REAL_AB '+json.dumps(dict(graph=graph, width=width, tokens=panel['upper'], anchors=len(anchors),
        pairs=d.width, policy=c['source_policy_id'], band=lane['band'], panel_sha256=spec['panel_sha256'],
        lengths=[x['input_ids'].shape[1] for x,_ in inputs], runs=records)),flush=True)
