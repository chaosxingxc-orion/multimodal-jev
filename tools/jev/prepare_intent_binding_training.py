#!/usr/bin/env python3
"""Publish a separate source-only binding training candidate, without dispatch."""
import argparse
from pathlib import Path
from collections import Counter
import math
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_binding_training import expand
from multimodal_jev.span_candidate_intervention import VARIANTS
from multimodal_jev.slurp_training import content_hash


def prepare(pilot, manifest_sha256, output):
    output=Path(output)
    if output.exists():raise FileExistsError('immutable candidate release exists')
    arms={};audit={}
    for arm in ('D','Q','P'):
        items,admission=load_pilot(pilot,manifest_sha256,arm)
        requests=[];targets=[]
        original=Counter();expanded=Counter()
        for item in items:
            original[item['target']['target_values'][-1]]+=item['target']['request_weight']
            for row,label in expand(item):
                requests.append(row);targets.append(label)
                expanded[label['target_values'][-1]]+=label['request_weight']
        if any(not math.isclose(original[k],expanded[k],rel_tol=1e-12) for k in original):
            raise ValueError('class weight mass differs')
        if len({r['id'] for r in requests})!=len(requests):raise ValueError('unique view IDs required')
        arms[arm]=(requests,targets)
        audit[arm]=dict(admission=admission,source_recordings=len(items),views=len(requests),
            request_weight_sum=sum(t['request_weight'] for t in targets),class_weight_mass=dict(expanded),
            source_groups=len({t['source_group'] for t in targets}))
    if content_hash(arms['Q'][0])!=content_hash(arms['P'][0]):raise ValueError('Q/P input mismatch')
    output.mkdir(parents=True)
    for arm,(requests,targets) in arms.items():
        atomic_json(output/f'{arm}-requests.json',requests);atomic_json(output/f'{arm}-targets.json',targets)
    files={p.name:file_hash(p) for p in output.glob('*.json')}
    repo=Path(__file__).resolve().parents[2]
    sources=[Path(__file__),repo/'src/multimodal_jev/intent_binding_training.py',
             repo/'src/multimodal_jev/span_candidate_intervention.py',repo/'src/multimodal_jev/slurp_intent_structure.py']
    manifest=dict(schema='slurp-intent-binding-training-candidate-v1',role='learning',
        source_pilot=str(pilot),source_manifest_sha256=manifest_sha256,variants=list(VARIANTS),files=files,
        implementation_sha256={str(p):file_hash(p) for p in sources},audit=audit,
        status='candidate_only_not_actor_admitted',
        budget='Four views share each source weight equally; 356 views per arm are 89 unique sources, not new independent audio.',
        requirements=['New versioned training protocol; do not splice into frozen D/Q/P.',
            'Exact transformed input HF/vLLM gate for every behavior policy before actor admission.',
            'A source group and all its views must remain within one split; target datasets not used.',
            'Match source uses and disclose extra view/input compute; do not count views as independent recordings.'],
        scope='Reference-text-reviewed learning labels only; no independent evaluation, listening certification or transfer gain.')
    atomic_json(output/'MANIFEST.json',manifest)
    return dict(root=str(output),manifest_sha256=file_hash(output/'MANIFEST.json'),**manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if a.receipt.exists():raise FileExistsError('preserve receipt')
    atomic_json(a.receipt,prepare(a.pilot,a.manifest_sha256,a.output))
