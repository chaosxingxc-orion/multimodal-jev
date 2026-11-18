import pytest

from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.context_training_schedule import build_epoch


def config(devices=3, inference=32):
    return dict(training_devices=devices, tensor_parallel=1, pipeline_parallel=1,
        per_dp_replica_batch=64, microbatch=16, inference_instances=4-devices,
        inference_batch_queries=inference, source_queries=40513, epochs=2,
        efficiency_steps=100)


@pytest.mark.parametrize('devices,total,updates,tail', [(2,128,634,65), (3,192,424,1)])
def test_population_and_tail_arithmetic(devices,total,updates,tail):
    p = derive(config(devices))
    assert p['global_batch'] == p['macro_queries'] == total
    assert p['total_updates'] == updates
    assert p['tail_real_queries'] == tail
    assert p['tail_padding_slots'] == total-tail
    assert p['accumulation'] == 4


def test_model_parallel_devices_not_counted_as_dp():
    c = config();c.update(training_devices=6,tensor_parallel=2)
    assert derive(c)['global_batch'] == 192
    c['training_devices'] = 5
    with pytest.raises(ValueError):derive(c)


@pytest.mark.parametrize('key,value', [('microbatch',48),('inference_batch_queries',4),
    ('per_dp_replica_batch',128),('training_devices',True),('efficiency_steps',101)])
def test_reject_unapproved_or_ambiguous_settings(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):derive(c)


def test_two_topologies_preserve_identical_full_epoch_real_order():
    # Real scheduler and length grouping, synthetic records. Changed batch
    # boundaries must not change source ordering/coverage or expose padding.
    rows=[dict(id=str(i),role='learning',known_pending_source=False,
               official_split='train',intent=str(i%7),source_group=str(i//2)) for i in range(385)]
    lengths={r['id']:(10+int(r['id'])%11,20+int(r['id'])%17) for r in rows}
    orders=[]
    for devices in (2,3):
        p=derive(config(devices))
        epoch=build_epoch(rows,lengths,epoch=0,global_batch=p['global_batch'],
                          world_size=p['dp_replicas'],microbatch=16)
        batches=epoch['batches']
        orders.append([s['id'] for b in batches for s in b['slots'] if not s['padding']])
        assert all(len(indices)==64 for b in batches for indices in b['rank_indices'])
        assert sum(s['weight'] for b in batches for s in b['slots'])==len(rows)
    assert orders[0]==orders[1] and len(set(orders[0]))==len(rows)
