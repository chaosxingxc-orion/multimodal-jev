"""Derive candidate batch sizes from topology; no training or service launch.

DP denotes independent data-parallel replicas. TP*PP devices form one replica.
The local effective batch includes gradient accumulation, not each forward.
"""
import math


def derive(config):
    keys = {'training_devices', 'tensor_parallel', 'pipeline_parallel',
            'per_dp_replica_batch', 'microbatch', 'inference_instances',
            'inference_batch_queries', 'source_queries', 'epochs', 'efficiency_steps'}
    rule = config.get('batch_rule')
    if 'batch_rule' in config and rule != 'microbatch_times_training_npus':
        raise ValueError('unknown batch_rule')
    fields = {k:v for k,v in config.items() if k != 'batch_rule'}
    if (set(fields) != keys
            or any(type(v) is not int or v < (0 if k == 'inference_instances' else 1)
                   for k, v in fields.items())):
        raise ValueError('explicit positive integer topology and batch settings required')
    c = config
    model_parallel = c['tensor_parallel'] * c['pipeline_parallel']
    if c['training_devices'] % model_parallel:
        raise ValueError('training devices must contain complete TP/PP replicas')
    if rule:
        if model_parallel != 1 or c['per_dp_replica_batch'] != c['microbatch']:
            raise ValueError('global_batch = microbatch * training NPUs requires pure DP and no accumulation')
    elif (c['per_dp_replica_batch'] not in ((64,96,128) if c['inference_instances']==0 else (64,)) or c['microbatch'] not in ((2, 4, 8, 16, 24, 32, 64) if c['inference_instances']==0 else (8, 16, 32))
            or c['per_dp_replica_batch'] % c['microbatch']
            or c['inference_batch_queries'] not in (32, 48)
            or c['efficiency_steps'] > 100):
        raise ValueError('divisible local batch required: effective64 (Stage1 also96/128), micro8/16/32 (Stage1 also2/4/24/64), inference32/48, at most100 steps required')
    dp = c['training_devices'] // model_parallel
    global_batch = c['per_dp_replica_batch'] * dp
    per_epoch = math.ceil(c['source_queries'] / global_batch)
    tail = c['source_queries'] - (per_epoch - 1) * global_batch
    return dict(schema='jev-topology-batch-plan-v1', configuration=dict(c),
        dp_replicas=dp, local_effective_batch=c['per_dp_replica_batch'],
        global_batch_formula='microbatch * training_devices' if rule else 'per_dp_replica_batch * training_devices / (TP * PP)',
        global_batch=global_batch, accumulation=c['per_dp_replica_batch'] // c['microbatch'],
        macro_queries=global_batch, inference_batch_queries=c['inference_batch_queries'],
        inference_delivery='complete_macro_to_pool_complete_rollout_to_one_instance',
        internal_batches='bounded_by_query_count_and_measured_token_audio_capacity',
        updates_per_epoch=per_epoch, total_updates=per_epoch*c['epochs'],
        tail_real_queries=tail, tail_padding_slots=global_batch-tail,
        global_loss_denominator='actual_real_query_count',
        efficiency_steps=c['efficiency_steps'],
        efficiency_query_exposures_upper_bound=c['efficiency_steps']*global_batch,
        efficiency_primary='real_queries_per_second_with_fixed_total_device_budget',
        training_dispatch_allowed=False,
        pending=['macro192 transport and runtime admission',
                 'DP3 native actor and numerical parity',
                 'service32/48 capacity and throughput',
                 'fresh schedule and runtime sealing'])
