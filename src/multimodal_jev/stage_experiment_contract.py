"""Derive the source curve and the reference-preparation geometry from a design."""
from .context_topology_batch_plan import derive


def source_curve(design):
    s = design['stage1']
    batch = s['batch_config']
    plan = derive(batch)
    steps = s['evaluation_steps']
    if (type(steps) is not list or len(steps) < 2 or any(type(i) is not int for i in steps)
            or steps != sorted(set(steps)) or steps[0] != 0
            or steps[-1] != plan['total_updates'] or steps[-1] != s['reference_endpoint']
            or batch['source_queries'] != design['population']['learning']
            or batch['inference_instances'] or batch['training_devices'] != design['total_device_budget']):
        raise ValueError('source curve must match the declared budget and full population')
    return dict(world_size=plan['dp_replicas'], steps=steps,
                real_exposures=batch['source_queries']*batch['epochs'],
                selection_recordings=design['population']['selection'])


def reference_batch(design):
    s = design['stage2']
    batch = dict(s['batch_config'])
    if batch['source_queries'] not in (None,design['population']['learning']):
        raise ValueError('reference must cover the full source population')
    batch['source_queries'] = design['population']['learning']
    plan = derive(batch)
    if (batch['training_devices']+batch['inference_instances'] != design['total_device_budget']
            or not batch['inference_instances'] or batch['training_devices'] != plan['dp_replicas']
            or plan['macro_queries'] != s['macro_queries'] or plan['global_batch'] != s['global_batch']
            or batch['inference_batch_queries'] != s['rollout_queries']):
        raise ValueError('Stage2 topology differs from design')
    return batch  # Scoring/packaging only. The final training view binds its own N.
