#!/usr/bin/env python3
"""Score all learning binding views after actual first-update acceptance."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_binding_gate import authenticate_artifacts
from multimodal_jev.intent_binding_inputs import load
from multimodal_jev.intent_binding_resume import expected_selection
from multimodal_jev.intent_binding_metrics import score,compare
from queue_span_native_actor_npu import read,completed
from queue_intent_binding_first_step import verify_final


def run(training,suite):
    training,suite=Path(training),Path(suite);cfg=read(training/'config.json')
    verify_final(suite,training,cfg['image']);verified=read(suite/'verified.json')
    after_gate=read(suite/'config.json');base=after_gate['base_id']
    configs=[cfg,dict(cfg,acceptance_root=str(suite),initial_adapter=str(training/'run/adapter'),pins=dict(cfg['pins']))]
    gate_paths=('config.json','launch.json','source-manifest.json','run/READY.json','run/terminal.json',
        'run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json')
    reports=[];evidence={}
    for current in configs:
        gate=Path(current['acceptance_root']);gc=read(gate/'config.json');launch=read(gate/'launch.json')
        if launch['image']!=cfg['image'] or not completed(launch['container_id'],cfg['image']):
            raise ValueError('actual completed before/after policies required')
        paths=[gate/n for n in gate_paths]+[Path(gc['inputs'])]
        for p in paths:current['pins'][str(p)]=file_hash(p);evidence[str(p)]=file_hash(p)
        populations,indices,audit=authenticate_artifacts(current,base+':'+gc['adapter_id'])
        reports.append(dict(adapter_id=gc['adapter_id'],audit=audit,arms={a:score(rows,
            [indices[1][r['gate_id']] for r in rows],a) for a,rows in populations.items()}))
    if reports[1]['adapter_id']!=verified['adapter_id']:raise ValueError('accepted adapter differs')
    _,selection=expected_selection(cfg);recorded=read(training/'run/selection.json')
    if any(recorded.get(k)!=v for k,v in selection.items()):raise ValueError('actual update source schedule differs')
    for p in (training/'config.json',training/'run/selection.json',suite/'verified.json',suite/'ACCEPTED.json',
              Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_binding_metrics.py'):
        evidence[str(p)]=file_hash(p)
    return dict(role='learning',updates=1,training_root=str(training),suite=str(suite),
        comparison={a:compare(reports[0]['arms'][a],reports[1]['arms'][a],selection['source_ids']) for a in ('D','Q')},
        before=reports[0],after=reports[1],files=evidence,
        scope='All89 learning sources and all fixed binding views. D trained; Q is template-only on the same weights. No independent generalization or FSC transfer claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--suite',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable binding comparison exists')
    atomic_json(a.output,run(a.training,a.suite))
