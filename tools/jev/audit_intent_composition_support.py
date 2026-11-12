#!/usr/bin/env python3
"""Audit admitted SLURP learning tuples and FSC public ontology only."""
import argparse
from collections import Counter
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.composition_support import audit
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_native_actor_npu import read


def run(pilot,manifest,fsc):
    pilot=Path(pilot);fsc=Path(fsc)
    items,admission=load_pilot(pilot,manifest,'P')
    m=read(pilot/'MANIFEST.json');task_path=Path(m['release'])/'tasks.json'
    pairs=read(task_path)['intent']['candidate_pairs'];counts=Counter(i['target']['target_values'][-1] for i in items)
    slurp=audit(pairs,counts);cfg=read(fsc)
    commands={r['id']:tuple(r['values'][k] for k in ('action','object','location')) for r in cfg['commands']}
    fsc_report=audit(commands)
    sources=[pilot/'MANIFEST.json',task_path,fsc,Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/composition_support.py']
    return dict(slurp=slurp,fsc_ontology_only=fsc_report,learning_admission=admission,
        files={str(p.resolve()):file_hash(p) for p in sources},
        limits=['No FSC utterance labels opened; no target test feedback.',
            'Existing 89 learning sources were already used for RL. A proposed heldout subset cannot become an unseen test of existing checkpoints.',
            'A future compositional run requires new frozen splits, fresh optimization from a disclosed initialization, source isolation and audio review.',
            'Component uniqueness and oracle scores measure annotation structure, not independent audio facts or causal reasoning.'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('pilot','manifest','fsc','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable audit exists')
    atomic_json(out,run(a.pilot,a.manifest,a.fsc))
