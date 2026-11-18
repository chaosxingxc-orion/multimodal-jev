"""Authenticate final Stage1 and initialize a fresh task-stage optimizer.

This is a stage boundary, not continuous-RL recovery or launch authorization.
The actual parity verifier must recompute the accepted full selection gate.
"""
import json
from pathlib import Path

from .baseline import file_hash
from .resident_service_admission import IMAGE
from .slurp_instruction_resume import FIELDS


def authenticate(config, *, verify_parity):
    """Bind terminal634 sources before loading or mutating the actor.

verify_parity is tools/jev/verify_slurp_batched_parity.verify in production;
its complete file graph is checked here, not just its passed flag.
"""
    from .span_batch_training_config import batch_layout, PROFILE
    batch_layout(config)
    plan = config['stage2_initialization']
    if (plan.get('mode') != 'final_stage1_fresh_optimizer'
            or plan.get('algorithm') not in ('grpo_v1', 'rloo_v1')
            or type(plan.get('start_update')) is not int or plan['start_update'] != 0
            or type(plan.get('total_updates')) is not int or plan['total_updates'] != 634
            or plan.get('reset_optimizer') is not True):
        raise ValueError('explicit final Stage1 and fresh full RL schedule required')
    files = {}
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('Stage2 initialization dependency changed: ' + path)
        files[path] = digest

    def read(path, *, pinned=False, audited=None):
        path = Path(path)
        digest = file_hash(path)
        if pinned and config['pins'].get(str(path)) != digest:
            raise ValueError('pinned Stage1 evidence required: ' + str(path))
        if audited is not None and audited.get(str(path)) != digest:
            raise ValueError('independently audited Stage1 artifact required: ' + str(path))
        files[str(path)] = digest
        return json.loads(path.read_text())

    audit_path = Path(plan['candidate_parity_audit_config'])
    audit = read(audit_path, pinned=True)
    gate_root = Path(audit['backend_root'])
    gate_config = read(gate_root / 'config.json', pinned=True)
    gate_launch = read(audit['backend_launch'], pinned=True)
    gate_observed = read(audit['backend_observed'], pinned=True)
    if (gate_launch['image'] != IMAGE or gate_observed['image'] != IMAGE
            or gate_launch['container_id'] != gate_observed['container_id']
            or gate_observed['state']['Status'] != 'exited'
            or gate_observed['state']['ExitCode'] != 0
            or gate_launch['config_sha256'] != file_hash(gate_root / 'config.json')):
        raise ValueError('successful fixed-image Stage1 selection process required')
    gate = verify_parity(audit_path)
    if (gate.get('passed') is not True or gate.get('full_candidate_parity') is not True
            or gate.get('exact_master_export') is not True or gate.get('updates') != 634
            or gate.get('candidate_profile') != PROFILE
            or gate.get('requests') != 9226 or gate.get('population_mode') != 'full_selection'):
        raise ValueError('terminal634 full9226 Stage1 parity required')
    for path, digest in gate['files'].items():
        if file_hash(path) != digest:
            raise ValueError('verified Stage1 file changed: ' + path)
        files[path] = digest
    root = Path(gate['training_root'])
    previous = read(root / 'config.json', audited=gate['files'])
    launch = read(audit['training_launch'], audited=gate['files'])
    if launch['image'] != IMAGE or 'sampled_probe' in previous or 'stage2_initialization' in previous:
        raise ValueError('final instruction stage, not RL fixture, required')
    for key in FIELDS + ('task_mode', 'loss_reduction', 'candidate_profile', 'microbatch_per_rank', 'batch_order'):
        if key not in previous or previous[key] != config.get(key):
            raise ValueError('Stage2 common factor changed: ' + key)
    if previous.get('total_updates') != 634:
        raise ValueError('complete instruction stage required')
    state_path = Path(audit['checkpoint_evidence'])
    state = read(state_path, audited=gate['files'])
    if (file_hash(state_path) != gate['state_sha256']
            or state.get('updates') != 634 or state.get('epoch') != 2 or state.get('next_batch') != 0
            or state.get('ranks') != 4 or state.get('tensors') != 112
            or any(state.get(k) is not True for k in
                   ('passed', 'exact_master_export', 'exact_peft_reload', 'durable_commit_verified'))
            or state['training_root'] != str(root) or state['adapter_id'] != gate['adapter_id']
            or state.get('original_adapter_id') != gate['adapter_id']
            or state.get('candidate_profile') != PROFILE):
        raise ValueError('exact final Stage1 master/export evidence required')
    shards = [root / 'run' / f'rank-{rank}/checkpoint-0634.pt' for rank in range(4)]
    assets = [Path(gate['adapter']) / name for name in ('adapter_config.json', 'adapter_model.safetensors')]
    for path in shards + assets:
        digest = file_hash(path)
        if state['files'].get(str(path)) != digest or gate['files'].get(str(path)) != digest:
            raise ValueError('complete independently audited Stage1 shards and adapter required')
        files[str(path)] = digest
    return dict(adapter=gate['adapter'], adapter_id=gate['adapter_id'],
        policy_id=config['base_id'] + ':' + gate['adapter_id'],
        reference_policy_id=config['base_id'] + ':' + gate['adapter_id'],
        candidate_profile=PROFILE, backend_config=str(gate_root/'config.json'),
        source_updates=634, completed_updates=0, next_update=0,
        shard_paths=[str(p) for p in shards], files=files,
        scope='Final Stage1 initialization source; no resident reload or RL launch acceptance.')


# The same exact FP32 master restore and fresh-Adam semantics apply to both profiles.
from .simple_stage2_initialization import restore_stage_boundary, continuous_plan
