"""Audit a collected two-update recovery against original continuous output.

The original run has no Adam checkpoint at the comparison endpoint. Validate
the recovered run's Adam independently; never claim equality to absent states.
"""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.local_config import load_config
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from tools.jev.audit_context_checkpoint_lifecycle import audit as audit_lifecycle
from tools.jev.audit_context_rl_training import verify_checkpoint
from tools.jev.context_replay_audit import require_exact
from tools.jev.context_rl_group_audit import audit as audit_groups
from tools.reports.audit_context_following_snapshot import audit as audit_reference
from tools.reports.context_arithmetic_resume_operations import validate, command
from tools.reports.context_dual_probe_operations import check_runtime
from tools.reports.context_master_export_audit import inspect_committed
from tools.reports.prepare_context_dual_throughput_pair import sha, write


def compare_step(original, resumed, index):
    """Compare semantic batch/groups; policy serialization IDs may differ.

Caller authenticates each policy and verifies sampled logs against its own
actual rollout receipts before invoking this tensor comparison.
"""
    def read(root,name):return json.loads((Path(root)/name).read_text())
    left=read(original,f'batch-{index:04d}.json');right=read(resumed,f'batch-{index:04d}.json')
    require_exact({k:v for k,v in left.items() if k!='policy_id'},
                  {k:v for k,v in right.items() if k!='policy_id'},'executed_batch')
    groups=[torch.load(Path(root)/f'groups-{index:04d}.pt',map_location='cpu',weights_only=True)
            for root in (original,resumed)]
    require_exact([{k:v for k,v in g.items() if k!='policy_id'} for g in groups[0]],
                  [{k:v for k,v in g.items() if k!='policy_id'} for g in groups[1]],'saved_groups')
    for g in groups[1]:
        if g['policy_id']!=right['policy_id']:raise ValueError('resumed group policy differs from batch')
    return groups[1]


def terminal(root, start, stop):
    def read(name):return json.loads((root/name).read_text())
    report=read('supervision/terminal.json')
    if (report['passed'] is not True or report['actor_started'] is not True
            or report['forced_cleanup'] or report['actor_exit_code']!=0 or report['service_exit_code']!=0
            or report['child_exit_codes']!=[0,0] or read('service/terminal.json')['exit_code']!=0):
        raise ValueError('normal supervised actor and service completion required')
    expected=dict(passed=True,start_update=start,stop_update=stop,schedule_updates=634,
                  learning_population=40513,engineering_prefix=True)
    if read('run/result.json')!=expected:raise ValueError('exact finite resumed segment required')
    for rank in range(4):
        restored=read(f'run/rank-{rank}/restored.json')
        if restored!=dict(exact_masters=True,exact_adam=True,completed_updates=start):
            raise ValueError('all-rank exact initial master/Adam restore required')
        result=read(f'run/rank-{rank}/result.json')
        if (result['passed'] is not True or result['start_update']!=start or result['stop_update']!=stop
                or len(result['executions'])!=113 or set(result['executions'].values())!={'bfloat16'}):
            raise ValueError('complete BF16 rank execution required')


def audit(c):
    operations,_=load_config(c['operations_config'],c['operations_local'])
    manifest,config=validate(operations);argv=command(operations,manifest,config)
    root=Path(c['collection']);wire=json.loads(Path(c['transport']).read_text())
    launch=json.loads(Path(c['launch']).read_text());observed=json.loads(Path(c['inspect']).read_text())
    check_runtime(dict(argv=argv,image=manifest['image']),launch,observed)
    if (wire['status']!='transport_complete_independent_audit_required'
            or wire['state']!=observed['State'] or wire['container_id']!=launch['container_id']
            or wire['image']!=launch['image'] or wire['remote_root']!=manifest['remote_root']
            or launch['remote_root']!=manifest['remote_root']
            or any(wire[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))):
        raise ValueError('exact unchanged terminal recovery collection required')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in root.rglob('*')) or actual!=wire['files']:
        raise ValueError('complete bound collection required')
    if sha(root.with_suffix('.tar'))!=wire['archive_sha256']:raise ValueError('transport archive changed')
    package=Path(operations['package'])
    for name,key in [('config.json','config_sha256'),('service-config.json','service_config_sha256'),
                     ('PREPARED.json','manifest_sha256')]:
        if actual[name]!=sha(package/name) or actual[name]!=launch[key]:raise ValueError('launch/package binding differs')
    start=config['start_update'];stop=config['stop_update'];terminal(root,start,stop)
    reference_config,_=load_config(c['reference_config'],c['reference_local'])
    reference=audit_reference(reference_config)
    snapshot=Path(reference_config['snapshot']);original=snapshot/'collection/run'
    binding=json.loads((package/'RESUME.json').read_text())
    proof=json.loads((snapshot/'COMPLETE.json').read_text())
    if (reference['source_container']!=binding['original_container']
            or reference['start_update']!=start or reference['stop_update']!=stop
            or launch['container_id']==reference['source_container']
            or proof['policy_id']!=binding['behavior_policy_id']
            or any(proof['files'].get(n)!=h for n,h in binding['checkpoint_files'].items())):
        raise ValueError('continuous reference differs from restored checkpoint')
    run=root/'run'
    lifecycle=audit_lifecycle(run,files={n[4:]:h for n,h in actual.items() if n.startswith('run/')},
        start_update=start,stop_update=stop,checkpoint_interval=config['checkpoint_interval'])
    if json.loads((root/'service/ready.json').read_text())['policy_id']!=proof['policy_id']:
        raise ValueError('initial restored engine differs')
    current=proof['policy_id'];summaries=[]
    for index in range(start,stop):
        groups=compare_step(original,run,index)
        batch=json.loads((run/f'batch-{index:04d}.json').read_text())
        receipt=json.loads((run/f'rollout-{index:04d}.json').read_text())
        original_receipt=json.loads((original/f'rollout-{index:04d}.json').read_text())
        if (batch['policy_id']!=current or receipt['behavior_policy_id']!=current or receipt['update']!=index
                or receipt['fixed_reference_policy_id']!=config['source_policy_id']):
            raise ValueError('resumed behavior chain differs')
        # Authenticate actual input tensors/positions as well as saved targets.
        for kind in ('behavior_record','reference_record'):
            require_exact([(q['request_id'],q[kind]['request_identity']['inputs'],q[kind]['request_identity']['positions'])
                           for q in original_receipt['queries']],
                          [(q['request_id'],q[kind]['request_identity']['inputs'],q[kind]['request_identity']['positions'])
                           for q in receipt['queries']],kind+'_inputs')
        targets={r['request_id']:dict(labels=r['target']['labels'],real_request=r['weight']==1.,weight=r['weight'])
                 for r in batch['records']}
        summary=audit_groups(groups,receipt['queries'],targets,algorithm=config['algorithm'],update=index,
            policy_id=current,reference_policy_id=config['source_policy_id'])
        current=lifecycle['policy_ids'][index-start]
        bundle=read_execution_bundle(run/f'policy-{index+1:04d}',expected_policy_id=current,
            expected_contract=config['source_contract'],expected_base_id=config['base_id'])
        old=read_execution_bundle(original/f'policy-{index+1:04d}',expected_policy_id=reference['policy_ids'][index-start],
            expected_contract=config['source_contract'],expected_base_id=config['base_id'])
        require_exact(old['tensors'],bundle['tensors'],'published_policy')
        manifests=[json.loads((p/f'policy-{index+1:04d}/MANIFEST.json').read_text()) for p in (original,run)]
        require_exact({k:v for k,v in manifests[0]['files'].items() if k!='execution.pt'},
                      {k:v for k,v in manifests[1]['files'].items() if k!='execution.pt'},'policy_processor_assets')
        verify_native_adapter(run/f'adapter-{index+1:04d}',bundle)
        for rank in range(4):
            record=json.loads((run/f'rank-{rank}/update-{index+1:04d}.json').read_text())
            parity=record['behavior_parity_history']
            if (record['rollout']!=receipt or record['real_queries']!=summary['real_queries'] or len(parity)!=4
                    or any(len(g)!=4 or any(p['passed'] is not True or p['threshold_flips']!=0 for p in g) for g in parity)):
                raise ValueError('all-rank behavior parity and rollout required')
        summaries.append(summary)
    commit_path=run/f'checkpoint-{stop:04d}-committed.json';commit=json.loads(commit_path.read_text())
    old_commit=json.loads((original/f'checkpoint-{start:04d}-committed.json').read_text())
    require_exact(old_commit['contract'],commit['contract'],'restored_objective')
    verify_checkpoint(run,stop,commit['contract'],bundle)
    master=inspect_committed(dict(run_root=str(run),step=stop,commit_sha256=sha(commit_path)))
    return dict(schema='context-arithmetic-finite-resume-audit-v1',passed=True,
        container_id=launch['container_id'],original_container=reference['source_container'],
        start_update=start,stop_update=stop,exact_samples=True,exact_bf16_policies=True,
        initial_restore_receipts_exact=True,final_master_export=master,final_adam_validated=True,
        final_adam_equal_to_continuous=None,continuous_final_adam_available=False,
        updates=summaries,transport_sha256=sha(c['transport']),inspect_sha256=sha(c['inspect']),
        package_sha256=operations['manifest_sha256'],reference_audit=reference,
        implementation_sha256=sha(__file__),full_training_admitted=False,throughput_gain_verified=False,
        scope='Two collected recovery updates match original samples and BF16 policies; final Adam independently valid, '
              'not equal-to-original proof because original intermediate Adam was not saved.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);torch.set_num_threads(1)
    result=audit(c);result['config_provenance']=provenance;write(c['output'],result)
    print(json.dumps(dict(passed=result['passed'],container_id=result['container_id'])))
