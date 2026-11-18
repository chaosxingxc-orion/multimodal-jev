"""One engineering update from an independently audited continuous checkpoint.

This does not grant formal Stage2 admission or change its reference/budget.
"""
def continuous_plan(config, recovered, optimizer, *, base_id, adapter_id):
    plan = config['sampled_probe']
    policy = base_id + ':' + adapter_id
    if ('stage2_initialization' in config
            or plan.get('fixture_initialization') != 'restore_continuous_then_one_update'
            or plan.get('stop_update') != 4 or type(plan.get('stop_update')) is not int
            or recovered.get('stage') != 'continuous_rl_engineering'
            or type(recovered.get('completed_updates')) is not int
            or recovered['completed_updates'] != 3
            or recovered['policy_id'] != policy or recovered['adapter_id'] != adapter_id
            or plan['algorithm'] != config['continuous_resume']['algorithm']
            or recovered['reference_policy_id'] != config['continuous_resume']['reference_policy_id']
            or not optimizer.state
            or any(s['step'].item() != 3 for s in optimizer.state.values())
            or any(p.grad is not None for group in optimizer.param_groups for p in group['params'])):
        raise ValueError('accepted step3 fixture, unchanged reference and exact continuing Adam required')
    return dict(policy_id=policy, reference_policy_id=recovered['reference_policy_id'],
                start_update=3, stop_update=4)
