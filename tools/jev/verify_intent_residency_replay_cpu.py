#!/usr/bin/env python3
"""Compare authenticated real replay post-states on CPU after backend acceptance.

Caller must separately observe exact replay/reference container completions.
This tool validates artifacts and tensors; it cannot observe the Docker daemon.
"""
import argparse
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_residency_comparison import compare_states
from multimodal_jev.policy_identity import checkpoint_identity
from queue_span_native_actor_npu import read
from queue_intent_finite_npu import verify_final
from queue_span_dp4_resume import check_pins
from probe_intent_residency_replay_npu import validate


def verify(root,suite):
    root,suite=map(Path,(root,suite));cfg=read(root/'config.json');validate(cfg)
    check_pins(cfg['pins']);reference=Path(cfg['replay_reference']);reference_suite=Path(cfg['replay_reference_suite'])
    evidence={};ranks=[]
    for training,gate in ((reference,reference_suite),(root,suite)):
        accepted=verify_final(gate);verified=read(gate/'verified.json')
        if verified['root']!=str(training) or verified['updates']!=2 or verified['arm']!='D':
            raise ValueError('matching complete D second-update acceptance required')
        if accepted['adapter_id']!=checkpoint_identity(training/'run/adapter')[0]:raise ValueError('adapter identity differs')
        evidence.update(verified['files'])
    if checkpoint_identity(root/'run/adapter')[0]!=checkpoint_identity(reference/'run/adapter')[0]:
        raise ValueError('replay BF16 export differs')
    for name in ('selection.json','supervision.json'):
        if read(root/'run'/name)!=read(reference/'run'/name):raise ValueError('actual selected inputs or supervision differ')
    for rank in range(4):
        out=root/'run'/f'rank-{rank}';receipt=read(out/'residency.json')
        if (receipt['rank']!=rank or any(receipt.get(k) is not True for k in
                ('passed','exact_parameters_and_adam','original_tensor_devices_preserved','parameter_optimizer_references_preserved'))):
            raise ValueError('complete rank residency evidence required')
        paths=[p/'run'/f'rank-{rank}'/'optimizer-local.pt' for p in (reference,root)]
        report=compare_states(*[torch.load(p,map_location='cpu',weights_only=True) for p in paths])
        if report['rank']!=rank:raise ValueError('rank shard substitution')
        ranks.append(dict(report,residency=receipt))
        evidence[str(out/'residency.json')]=file_hash(out/'residency.json')
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_residency_comparison.py'):
        evidence[str(p)]=file_hash(p)
    return dict(passed=True,ranks=ranks,files=evidence,reference=str(reference),replay=str(root),
                scope='Exact post-update real replay master/Adam/export artifacts and accepted full backend gates. Actual Docker exits must be observed by caller; no actor/vLLM co-residency or new efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--suite',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable replay verification exists')
    atomic_json(a.output,verify(a.root,a.suite))
