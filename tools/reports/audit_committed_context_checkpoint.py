"""Validate an immutable live-run snapshot and restore its CPU master/Adam.

No model construction, NPU restart or fresh-process inference acceptance.
"""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.local_config import load_config
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_training_checkpoint import restore
from tools.jev.audit_context_rl_training import verify_checkpoint
from tools.reports.collect_committed_context_checkpoint import inventory
from tools.reports.context_master_export_audit import inspect_committed
from tools.reports.prepare_context_dual_throughput_pair import sha,write


def audit(c):
    snapshot=Path(c['snapshot']);proof=json.loads((snapshot/'COMPLETE.json').read_text())
    attempt=json.loads((snapshot/'attempt.json').read_text())['config']
    launch=json.loads(Path(attempt['launch']).read_text())
    if sha(attempt['launch'])!=attempt['launch_sha256']:
        raise ValueError('original launch changed')
    root=snapshot/'collection';step=attempt['step']
    if (proof['schema']!='context-committed-checkpoint-snapshot-v1' or proof['step']!=step or
            proof['container_id']!=launch['container_id'] or proof['image']!=launch['image'] or
            proof['collection']!=str(root) or sha(snapshot/'checkpoint.tar')!=proof['archive_sha256']):
        raise ValueError('snapshot provenance differs')
    actual=inventory(root,step,launch)
    if actual['files']!=proof['files'] or actual['policy_id']!=proof['policy_id']:
        raise ValueError('snapshot inventory differs')
    config=json.loads((root/'config.json').read_text());run=root/'run'
    commit_path=run/f'checkpoint-{step:04d}-committed.json';commit=json.loads(commit_path.read_text())
    expected={k:config[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
        'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm',
        'behavior_bounds','binary_policy_arithmetic')}
    expected.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        objective=dict(group_size=16,reward='complete_candidate_set_f1',kl=.01,
                       auxiliary='cosine_0.3_to_0.1_first_80pct'),grad_clip=1.,world_size=4)
    if commit['contract']!=expected:
        raise ValueError('saved optimizer/objective differs from original experiment')
    bundle=read_execution_bundle(run/f'policy-{step:04d}',expected_policy_id=proof['policy_id'],
        expected_contract=config['source_contract'],expected_base_id=config['base_id'])
    verify_checkpoint(run,step,expected,bundle)
    master=inspect_committed(dict(run_root=str(run),step=step,commit_sha256=sha(commit_path)))
    restored=[]
    for rank in range(4):
        state=torch.load(run/f'rank-{rank}/checkpoint-{step:04d}.pt',map_location='cpu',weights_only=True)
        named=[(n,torch.nn.Parameter(torch.zeros_like(v))) for n,v in state['parameters'].items()]
        opt=torch.optim.AdamW([p for _,p in named],lr=1e-4,betas=(.9,.999),eps=1e-8,
            weight_decay=0.,foreach=False,fused=False)
        restored.append(restore(named,opt,state,rank=rank,world_size=4,completed_updates=step,contract=expected))
    return dict(schema='context-committed-cpu-recovery-audit-v1',container_id=launch['container_id'],
        step=step,algorithm=config['algorithm'],policy_id=proof['policy_id'],master_export=master,
        restored=restored,files=proof['files'],snapshot_manifest_sha256=sha(snapshot/'COMPLETE.json'),
        implementation_sha256=sha(__file__),native_adapter_tensor_audit=False,
        fresh_process_npu_resume_verified=False,stop_training_allowed=False,
        scope='Real saved LoRA/head tensors and Adam restored on CPU; no NPU execution, native adapter reload or throughput claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);torch.set_num_threads(1)
    result=audit(c);result['config_provenance']=provenance;write(c['output'],result)
    print(json.dumps(dict(step=result['step'],master_parameters=result['master_export']['parameters'],
        exact_cpu_restored_ranks=len(result['restored']),fresh_process_npu_resume_verified=False)))
