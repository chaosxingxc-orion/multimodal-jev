#!/usr/bin/env python3
"""Actual tensors of an accepted D2 replay; caller observes Docker exits."""
import argparse
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_correctness_replay import validate_factors,validate_rank
from multimodal_jev.intent_residency_comparison import compare_states
from multimodal_jev.policy_identity import checkpoint_identity
from queue_span_native_actor_npu import read
from queue_intent_finite_npu import verify_final


def verify(reference,reference_suite,replay,replay_suite):
    reference,reference_suite,replay,replay_suite=map(Path,(reference,reference_suite,replay,replay_suite))
    factors=validate_factors(read(reference/'config.json'),read(replay/'config.json'))
    evidence={};ranks=[]
    for root,suite in ((reference,reference_suite),(replay,replay_suite)):
        accepted=verify_final(suite);verified=read(suite/'verified.json')
        if (verified['root']!=str(root) or verified['suite']!=str(suite) or verified['updates']!=2
                or verified['arm']!='D' or accepted['adapter_id']!=checkpoint_identity(root/'run/adapter')[0]):
            raise ValueError('accepted exact D2 policy required')
        evidence.update(verified['files'])
        for p in (root/'config.json',suite/'config.json',suite/'verified.json',suite/'ACCEPTED.json',suite/'controller-terminal.json'):
            evidence[str(p)]=file_hash(p)
    if checkpoint_identity(reference/'run/adapter')[0]!=checkpoint_identity(replay/'run/adapter')[0]:
        raise ValueError('BF16 adapter export differs')
    for name in ('selection.json','supervision.json'):
        if read(reference/'run'/name)!=read(replay/'run'/name):raise ValueError('source or supervision replay differs')
        for root in (reference,replay):evidence[str(root/'run'/name)]=file_hash(root/'run'/name)
    for rank in range(4):
        a,b=[root/'run'/f'rank-{rank}' for root in (reference,replay)]
        validate_rank(read(a/'result.json'),read(b/'result.json'),rank)
        # Equality here tests the behavior and pre-update actor distributions,
        # beyond merely checking same-dtype or post-update accuracy.
        for name in ('before.json','after.json'):
            if read(a/name)!=read(b/name):raise ValueError('actual candidate probabilities differ: '+name)
        states=[torch.load(p/'optimizer-local.pt',map_location='cpu',weights_only=True) for p in (a,b)]
        ranks.append(compare_states(*states))
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_correctness_replay.py',
              Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_residency_comparison.py'):
        evidence[str(p)]=file_hash(p)
    return dict(passed=True,reference=str(reference),replay=str(replay),factors=factors,ranks=ranks,files=evidence,
        scope='Exact accepted D2 master/Adam/export and candidate-probability artifact replay. Caller must observe actual Docker exits. Not new efficacy or continuous trainer evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','reference-suite','replay','replay-suite','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable replay verification exists')
    atomic_json(out,verify(a.reference,a.reference_suite,a.replay,a.replay_suite))
