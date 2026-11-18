"""Authenticated epoch boundary for the finite cached-reference GRPO check.

Keep the complete schedule immutable. Diagnostic optimizer steps 0/1 consume
the first epoch's tail and the next epoch's first batch, not formal history.
"""
from .context_topology_batch_plan import derive


def contract(config):
    plan = derive(config['batch_config'])
    integers = {'start_update': 0, 'stop_update': 2, 'checkpoint_interval': 1}
    if (any(type(config.get(k)) is not int or config[k] != v for k, v in integers.items())
            or config.get('experiment_kind') != 'cached_reference_tail_admission'
            or config.get('stage') != 'task' or config.get('resume_root') is not None
            or config.get('engineering_prefix') is not True
            or config.get('training_dispatch_allowed') is not False
            or 'reference_cache' in config
            or plan['dp_replicas'] != 2 or plan['global_batch'] != 128
            or plan['configuration']['inference_instances'] != 2
            or plan['configuration']['source_queries'] != 40513
            or plan['configuration']['epochs'] != 2):
        raise ValueError('fresh two-update DP2 cached GRPO tail diagnostic required')
    return dict(schedule_indices=[plan['updates_per_epoch'] - 1, plan['updates_per_epoch']],
                real_queries=[plan['tail_real_queries'], plan['global_batch']],
                padding_slots=[plan['tail_padding_slots'], 0], diagnostic_updates=2,
                formal_history=False, optimizer_cursor='fresh_0_1',
                full_schedule_updates=plan['total_updates'])


def select(batches, config, population):
    # Reuse the formal validator, including every sample exactly once per epoch.
    from tools.reports.context_replica_formal_candidate import validate_schedule
    spec = contract(config)
    plan = derive(config['batch_config'])
    validate_schedule(batches, population, plan)
    selected = [batches[i] for i in spec['schedule_indices']]
    if [b['real_queries'] for b in selected] != spec['real_queries']:
        raise ValueError('authenticated epoch boundary real-query counts required')
    return selected
