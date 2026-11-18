"""Formal recovery joins released configuration, fixed Stage1 and RL state.

No optimizer reset or policy-reference refresh occurs at an RL restart.
The source must be an explicitly audited, exited prefix of this experiment.
"""
import json
from pathlib import Path

from .baseline import file_hash


def authenticate(config, *, verify_parity):
    from .simple_stage2_release import authenticate as release
    from .simple_stage2_initialization import authenticate as initialize
    from .simple_continuous_resume import authenticate as resume
    from .simple_resident_selection import SELECTION_UPDATES
    released = release(config)
    initial = initialize(config, verify_parity=verify_parity)
    recovered = resume(config, formal=True)
    if recovered['reference_policy_id'] != initial['reference_policy_id']:
        raise ValueError('RL recovery must retain final Stage1 as fixed reference')
    plan = config['continuous_resume']; source = Path(plan['training_root'])
    audit = json.loads(Path(plan['audit']).read_text())
    history = []
    for step, accepted in zip(
            (s for s in SELECTION_UPDATES if s <= recovered['completed_updates']),
            audit['selection'], strict=True):
        path = Path(accepted['result'])
        digest = file_hash(path)
        if (recovered['files'].get(str(path)) != digest
                or accepted.get('passed') is not True
                or accepted['result_sha256'] != digest or accepted['updates'] != step):
            raise ValueError('independently audited selection history required')
        history.append(dict(updates=step, policy_id=accepted['policy_id'],
                            result=str(path), result_sha256=digest))
    recovered.update(files={**released['files'], **initial['files'], **recovered['files']},
                     selection_history=history, initial=initial, prefix_audit=audit)
    return recovered


def restore(model, optimizer, recovered, *, rank):
    """Restore all FP32 information, then independently compare before update."""
    from .span_fsdp_resume import restore_local_shard
    from .intent_resident_replay import compare_state
    state = recovered['states'][rank]; updates = recovered['completed_updates']
    receipt = restore_local_shard(model, optimizer, state, rank=rank, world_size=4, updates=updates)
    actual = dict(state, parameters={n: p.detach().cpu().clone()
        for n, p in model.named_parameters() if p.requires_grad}, optimizer=optimizer.state_dict())
    exact = compare_state(state, actual, rank=rank, updates=updates, continuous=True, formal=True)
    optimizer.zero_grad(set_to_none=True)
    return actual, dict(restore=receipt, exact_state=exact, completed_updates=updates,
                       reference_policy_id=recovered['reference_policy_id'])


def continuous_plan(plan, recovered, optimizer, *, base_id, adapter_id):
    """Continue the original total budget with the recovered Adam step."""
    updates = recovered['completed_updates']; policy = base_id + ':' + adapter_id
    if (type(updates) is not int or not 1 <= updates < 634
            or recovered.get('stage') != 'simple_stage2_rl'
            or recovered['policy_id'] != policy or recovered['adapter_id'] != adapter_id
            or recovered['reference_policy_id'] != recovered['initial']['reference_policy_id']
            or plan.get('mode') != 'final_stage1_fresh_optimizer'
            or any(type(plan.get(k)) is not int or plan[k] != v for k,v in
                   [('start_update',0),('stop_update',634),('total_updates',634)])
            or plan.get('reset_optimizer') is not True
            or not optimizer.state
            or any(s['step'].item() != updates for s in optimizer.state.values())
            or any(p.grad is not None for group in optimizer.param_groups for p in group['params'])):
        raise ValueError('exact recovered Stage2 cursor and preserved Adam required')
    return dict(policy_id=policy, reference_policy_id=recovered['reference_policy_id'],
                start_update=updates, stop_update=634)
