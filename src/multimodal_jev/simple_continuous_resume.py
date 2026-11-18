"""Authenticate independently audited resident checkpoints before model loading.

All four shards and the BF16 export are checked even when a caller loads one
rank. This validates a recovery source, never creates training authorization.
"""
import json
from pathlib import Path
from .baseline import file_hash
from .policy_identity import checkpoint_identity
from .slurp_training import content_hash
from .slurp_instruction_resume import FIELDS


def authenticate(config, *, formal=False):
    import torch
    from safetensors.torch import load_file
    from .simple_sampled_checkpoint import reconstruct
    plan=config['continuous_resume'];files={}
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('continuous recovery dependency changed: '+path)
        files[path]=digest
    def read(path,pinned=False):
        path=Path(path);digest=file_hash(path)
        if pinned and config['pins'].get(str(path))!=digest:
            raise ValueError('pinned continuous recovery dependency required: '+str(path))
        files[str(path)]=digest
        return json.loads(path.read_text())
    audit=read(plan['audit'],pinned=True)
    if (audit.get('passed') is not True or audit.get('exact_master_export') is not True
            or audit.get('exact_peft_reload') is not True or audit.get('ranks')!=4
            or audit.get('exact_pre_update_recovery') is not True):
        raise ValueError('independently accepted continuous checkpoint required')
    if formal and (audit.get('stage') != 'simple_stage2_rl'
                   or audit.get('audit_mode') != 'stopped_prefix'
                   or audit.get('complete_training') is not False):
        raise ValueError('explicit independently audited formal stopped prefix required')
    audit_launch=read(plan['audit_launch'],pinned=True)
    audit_observed=read(plan['audit_observed'],pinned=True)
    if (audit_observed['container_id']!=audit_launch['container_id']
            or audit_observed['image']!=audit_launch['image']
            or audit_observed['state']['Status']!='exited' or audit_observed['state']['ExitCode']!=0):
        raise ValueError('successful exact independent audit process required')
    from .resident_service_admission import IMAGE
    if audit_launch['image']!=IMAGE:raise ValueError('pinned audit image required')
    root=Path(plan['training_root']);updates=plan['completed_updates']
    if (type(updates) is not int or not 1<=updates<=634 or audit['updates']!=updates
            or Path(plan['audit'])!=Path(audit_launch['remote_root'])/'run/state-verified.json'):
        raise ValueError('exact audited checkpoint/cursor required')
    if formal and (updates >= 634 or audit.get('new_updates') != updates
                   or len(audit.get('checks', [])) != updates
                   or any(c.get('passed') is not True for c in audit['checks'])):
        raise ValueError('every formal prefix update must be independently accepted')
    if formal:
        from .simple_resident_selection import SELECTION_UPDATES
        due=[step for step in SELECTION_UPDATES if step<=updates]
        if ([entry.get('updates') for entry in audit.get('selection',[])]!=due
                or len(audit.get('initial_state_checks',[]))!=4
                or any(c.get('passed') is not True for c in audit['initial_state_checks'])):
            raise ValueError('complete initial-state and due selection audit required')
        policies=audit.get('policies',{})
        if (set(policies)!={str(i) for i in range(updates+1)}
                or policies.get('0')!=plan['reference_policy_id']
                or any(c.get('updates')!=i or c.get('policy_id')!=policies[str(i)]
                       for i,c in enumerate(audit['checks'],1))):
            raise ValueError('complete ordered prefix policy chain required')
    for path,digest in audit['files'].items():
        if file_hash(path)!=digest:raise ValueError('audited recovery file changed: '+path)
        files[path]=digest
    train=read(root/'config.json')
    launch=read(root/'launch.json');observed=read(plan['training_observed'],pinned=True)
    terminal=read(root/'terminal.json') if not formal else None
    if (str(root/'config.json') not in audit['files'] or str(root/'launch.json') not in audit['files']
            or (not formal and str(root/'terminal.json') not in audit['files'])
            or str(Path(plan['training_observed'])) not in audit['files']
            or launch['container_id']!=observed['container_id'] or launch['image']!=IMAGE
            or observed['image']!=IMAGE or observed['state']['Status']!='exited'
            or type(observed['state'].get('ExitCode')) is not int
            or (not formal and (observed['state']['ExitCode']!=0 or terminal['exit_code']!=0))
            or launch['config_sha256']!=file_hash(root/'config.json')):
        raise ValueError('audited successful source run required')
    for key in FIELDS+('microbatch_per_rank','task_mode','loss_reduction'):
        if train.get(key)!=config.get(key) or key not in train:
            raise ValueError('continuous recovery factor changed: '+key)
    if formal:
        if 'sampled_probe' in train or 'sampled_probe' in config:
            raise ValueError('formal recovery cannot consume engineering fixture')
        previous=train['stage2_initialization'];current=config['stage2_initialization']
        for key in ('mode','algorithm','start_update','stop_update','total_updates','reset_optimizer',
                    'gate','selection_inputs','selection_updates','experiment','lifecycle'):
            if key not in previous or previous[key]!=current.get(key):
                raise ValueError('formal recovery definition changed: '+key)
        algorithm=previous['algorithm']
    else:
        algorithm=train['sampled_probe']['algorithm']
    fixed=plan['reference_policy_id']
    if algorithm!=plan['algorithm'] or config['physical_devices']!=train['physical_devices']:
        raise ValueError('fixed algorithm and DP4 physical topology required')
    commit_path=root/'run'/f'checkpoint-{updates:04d}-committed.json'
    committed=read(commit_path)
    if str(commit_path) not in audit['files'] or committed['passed'] is not True:
        raise ValueError('independently audited durable commit required')
    for path in [root/'run'/f'inputs-{updates-1:04d}.json',
                 root/'run'/f'rollout-after-{updates:04d}.json']:
        if str(path) not in audit['files']:
            raise ValueError('independently audited recovery input/probability source required')
    adapter=root/'run'/f'loop-adapter-{updates:04d}'
    identity,assets=checkpoint_identity(adapter)
    if any(audit['files'].get(str(adapter/name))!=digest for name,digest in assets.items()):
        raise ValueError('independently audited adapter assets required')
    policy=config['base_id']+':'+identity
    if formal and audit['policies'][str(updates)]!=policy:
        raise ValueError('audited last policy differs from recovered adapter')
    paths=[root/'run'/f'loop-rank-{rank}'/f'checkpoint-{updates:04d}.pt' for rank in range(4)]
    hashes={str(path):file_hash(path) for path in paths}
    if (committed['files']!=hashes or any(audit['files'].get(p)!=h for p,h in hashes.items())
            or committed['reference_policy_id']!=fixed or committed['cursor']!=dict(
                schema='jev-span-training-session-v1',completed_updates=updates,policy_id=policy,
                schedule_sha256=config['schedule_sha256'],checkpoint_id=content_hash(hashes))):
        raise ValueError('complete committed shards/reference/schedule required')
    states=[torch.load(path,map_location='cpu',weights_only=True) for path in paths]
    if any(s['schedule_sha256']!=config['schedule_sha256'] or s['next_update']!=updates for s in states):
        raise ValueError('continuous shard cursor differs')
    # Audit already authenticated the parameter layout; require all ranks agree
    # again and reconstruct export independently before restoring any rank.
    reconstruct(states,load_file(str(adapter/'adapter_model.safetensors')),algorithm=algorithm,
        adapter_id=identity,reference_policy_id=fixed,lr=config['lr'],expected_names=states[0]['names'],
        completed_updates=updates,expected_stage='simple_stage2_rl' if formal else 'continuous_rl_engineering')
    return dict(adapter=str(adapter),adapter_id=identity,policy_id=policy,completed_updates=updates,
        reference_policy_id=fixed,states=states,files=files,checkpoint_id=content_hash(hashes),
        stage='simple_stage2_rl' if formal else 'continuous_rl_engineering',
        scope='Audited recovery source authenticated before model mutation; resident reload/probabilities and formal release authorization separate.')
