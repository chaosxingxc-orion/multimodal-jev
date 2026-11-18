#!/usr/bin/env python3
"""Exact two-step control: same global batch/microbatch, changed bookkeeping.

Both checkpoints must first pass the independent full state/export audit.
This is not a tolerance-based microbatch comparison or a throughput claim.
"""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(reference_audit, candidate_audit):
    import torch
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.policy_identity import compare_adapter_artifacts
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only independent state comparison')
    torch.set_num_threads(2)
    audits=[json.loads(Path(p).read_text()) for p in (reference_audit,candidate_audit)]
    for a in audits:
        if not all(a[k] for k in ('passed','exact_master_export','exact_peft_reload','durable_commit_verified')):
            raise ValueError('accepted state/PEFT/durable audits required')
        if a['updates']!=2:raise ValueError('two-update control only')
        for path,digest in a['files'].items():
            if file_hash(path)!=digest:raise ValueError('audited evidence changed')
    roots=[Path(a['training_root']) for a in audits]
    configs=[json.loads((r/'config.json').read_text()) for r in roots]
    from multimodal_jev.slurp_instruction_resume import FIELDS
    for field in FIELDS:
        if configs[0][field]!=configs[1][field]:raise ValueError('control configuration differs: '+field)
    for field,default in [('microbatch_per_rank',1),('task_mode','instruction_views'),('loss_reduction','legacy_weight_mean'),('start_update',0)]:
        if configs[0].get(field,default)!=configs[1].get(field,default):raise ValueError('control configuration differs: '+field)
    initial = compare_adapter_artifacts(*(r/'run/adapter-step0000' for r in roots))
    for step in range(2):
        batches=[json.loads((r/'run'/f'batch-{step:04d}.json').read_text()) for r in roots]
        if batches[0]['records']!=batches[1]['records']:raise ValueError('exact same supervised audio slots required')
    reports=[]
    for rank in range(4):
        states=[torch.load(r/'run'/f'rank-{rank}'/'checkpoint-0002.pt',map_location='cpu',weights_only=True) for r in roots]
        reports.append(compare_state(*states,rank=rank,updates=2))
    endpoint = compare_adapter_artifacts(*(r/'run/adapter-step0002' for r in roots))
    if [a['adapter_id'] for a in audits] != [endpoint['reference_id'], endpoint['candidate_id']]:
        raise ValueError('audited endpoint artifact identity differs')
    return dict(passed=True,exact_fp32_masters_and_adam=True,exact_bf16_adapter=True,
        adapter_id=audits[1]['adapter_id'],reference_adapter_id=audits[0]['adapter_id'],
        initial_artifact_comparison=initial,endpoint_artifact_comparison=endpoint,ranks=reports,
        audits={str(p):file_hash(p) for p in (reference_audit,candidate_audit)},
        scope='Exact same-input/same-micro two-step state control across physical devices and bookkeeping cadence; no new full-dataset efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-audit',type=Path,required=True)
    p.add_argument('--candidate-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable comparison output exists')
    atomic_json(a.output,run(a.reference_audit,a.candidate_audit))
