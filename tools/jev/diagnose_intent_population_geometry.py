#!/usr/bin/env python3
"""Describe fixed accepted intent candidate distributions without gold labels."""
import argparse
import string
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_population_geometry import summarize
from queue_span_native_actor_npu import read,completed
from queue_span_dp4_resume import check_pins


def run(config):
    check_pins(config['pins']);reports=[];reference_inputs=None
    for entry in config['points']:
        root=Path(entry['gate']);launch=read(root/'launch.json');cfg=read(root/'config.json')
        if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):
            raise ValueError('actual completed accepted gate required')
        for name in ('config.json','source-manifest.json','run/READY.json','run/vllm/result.json','run/terminal.json'):
            if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('fixed gate evidence required')
        check_pins(cfg['pins'])
        if read(root/'run/terminal.json')['exit_code']!=0:raise ValueError('gate phase failed')
        ready=read(root/'run/READY.json');data=read(root/'run/vllm/result.json')
        if (not ready['passed'] or ready['requests']!=178 or ready['adapter_id']!=cfg['adapter_id']
                or ready['result_sha256']!=file_hash(root/'run/vllm/result.json')):raise ValueError('full accepted population required')
        rows=data['records'];inputs={r['id']:r['request_identity']['request'] for r in rows}
        if len(rows)!=178 or len(inputs)!=178:raise ValueError('complete distinct input population required')
        if reference_inputs is None:reference_inputs=inputs
        elif inputs!=reference_inputs:raise ValueError('fixed inputs required along curve')
        groups={}
        for arm in ('D','Q'):
            selected=[r for r in rows if r['id'].endswith('::'+arm)]
            if len(selected)!=89:raise ValueError('full template population required')
            qs=selected[0]['request_identity']['request']['questions'];spans={}
            for j,q in enumerate(qs):
                if any(r['request_identity']['request']['questions'][j]!=q for r in selected):raise ValueError('same candidate definition required')
                values=[c['id'] for c in q['schema']['candidates']]
                symbols=list((string.ascii_uppercase+string.ascii_lowercase+string.digits)[:len(values)])
                spans[q['id']]=summarize([r['candidate_logprobs'][j] for r in selected],values,symbols)
            groups[arm]=spans
        reports.append(dict(update=entry['update'],adapter_id=cfg['adapter_id'],templates=groups))
    return dict(reports=reports,files=config['pins'],scope='Descriptive accepted learning-input geometry only, no labels used here. '
        'Does not isolate audio causality or distinguish token prior from semantic class preference without intervention.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable diagnosis exists')
    atomic_json(a.output,run(read(a.config)))
