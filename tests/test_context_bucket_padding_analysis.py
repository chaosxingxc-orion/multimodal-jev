import pytest
import hashlib
import json

from tools.reports.analyze_context_bucket_padding import summarize, analyze
from multimodal_jev.span_length_groups import length_group_indices


def test_equal_shapes_no_padding_or_speed_claim():
    r = summarize([[(100, 200)]*128], {})
    assert r['totals']['baseline']['padded_tokens'] == 12800
    assert r['totals']['candidate']['padded_frames'] == 25600
    assert r['reduction_fraction'] == dict(padded_tokens=0, padded_frames=0, slowest_rank_cost_proxy=0)
    assert r['measured_speedup'] is None


def test_mixed_shapes_account_for_real_members_and_preserve_tokens_frames():
    lengths = [(100+i, 1 if i % 2 else 1000) for i in range(128)]
    r = summarize([lengths, lengths], {})
    assert r['useful']['tokens'] == 2*sum(t for t, f in lengths)
    assert r['useful']['frames'] == 2*sum(f for t, f in lengths)
    assert r['reduction_fraction']['slowest_rank_cost_proxy'] > 0
    for mode, values in r['totals'].items():
        assert values['padded_tokens'] >= r['useful']['tokens']
        assert values['padded_frames'] >= r['useful']['frames']
    assert sum(r['selected'].values()) == 2


def test_empty_or_incomplete_batch_rejected():
    with pytest.raises(ValueError): summarize([], {})
    with pytest.raises(ValueError): summarize([[(1, 1)]*127], {})


@pytest.mark.parametrize('fault', [None, 'engine_identity', 'rank_allocation', 'transport'])
def test_actual_trace_binding_uses_distinct_slot_and_engine_ids(tmp_path, fault):
    root = tmp_path/'collected/run'; root.mkdir(parents=True)
    files = {}
    def write(name, value):
        p = root/name; p.write_text(json.dumps(value))
        files['run/'+name] = hashlib.sha256(p.read_bytes()).hexdigest()
    write('result.json', dict(passed=True, start_update=0, stop_update=100))
    lengths = [(100+i, 200) for i in range(128)]
    for update in range(100):
        records = []; queries = []
        for i, (tokens, frames) in enumerate(lengths):
            slot = f'slot-{update}-{i}'
            identity = dict(policy_id='policy', inputs=dict(input_ids=dict(shape=[1, tokens]),
                input_features=dict(shape=[1, 128, frames])))
            engine_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if fault == 'engine_identity' and update == 0 and i == 0: engine_id = slot
            records.append(dict(request_id=slot))
            queries.append(dict(request_id=slot, behavior_record=dict(request_id=engine_id,
                request_identity=identity, policy_id='policy')))
        ranks = length_group_indices(lengths, microbatch=8, world_size=2, order='length_grouped')
        if fault == 'rank_allocation' and update == 0: ranks[0][0] = ranks[0][1]
        write(f'batch-{update:04d}.json', dict(records=records, policy_id='policy', rank_indices=ranks))
        write(f'rollout-{update:04d}.json', dict(update=update, queries=queries))
    (tmp_path/'transport.json').write_text(json.dumps(dict(files=files)))
    if fault == 'transport': (root/'rollout-0000.json').write_text('{}')
    config = dict(pins={}, operations_root=str(tmp_path), length_buckets={})
    if fault:
        with pytest.raises(ValueError): analyze(config)
    else:
        report = analyze(config)
        assert report['queries'] == 12800 and report['updates'] == 100
        assert report['measured_speedup'] is None
