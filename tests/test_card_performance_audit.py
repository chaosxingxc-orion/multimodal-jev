import hashlib
import json

import pytest

from multimodal_jev import evaluation_blocks as blocks
from tools.reports.audit_card_archive import compare_archive
from tools.reports.audit_card_latency import validate_trial


def test_twofold_is_per_checkpoint_and_small_panel_cannot_accept():
    from copy import deepcopy
    from tools.reports.audit_card_latency import apply_throughput_gate
    policy = dict(dataset='d', arms=['stage1', 'tpo'], throughput_min=2,
                  first_decision_ratio_max=1.1, screen_blocks_per_arm=3, acceptance_blocks_per_arm=20)
    row = dict(blocks=3, standard=dict(scoring_seconds=10), candidate=dict(scoring_seconds=5, probability_differences=0),
               ratios=dict(first_decision_p50=1., first_decision_p95=1.1))
    result = dict(endpoints={'d': {arm: deepcopy(row) for arm in policy['arms']}})
    apply_throughput_gate(result, policy, 'screen')
    assert result['passed'] and not result['acceptance_pass']
    tpo = result['endpoints']['d']['tpo']
    tpo['candidate']['scoring_seconds'] = 5.01
    assert not apply_throughput_gate(result, policy, 'screen')['passed']
    assert tpo['parity_pass'] and tpo['latency_pass'] and not tpo['throughput_pass']
    with pytest.raises(ValueError, match='incomplete'):
        apply_throughput_gate(result, policy, 'acceptance')


def test_archive_checks_each_probability_even_when_top1_identical(tmp_path):
    block = dict(key='block-000000', dataset='d', arm='s', ids=['u'], widths=[2])
    plan = dict(inputs={}, policies={'s': 'p'}, choice=False, blocks=[block])
    manifest = dict(sha256=blocks.content_key(plan), plan=plan)
    def products(p):
        return dict(prob=dict(policy_id='p', records=[dict(id='u', prob=[.9, p])]))
    original = products(.12)
    pin = tmp_path / 'plan.json'
    pin.write_text(json.dumps(manifest))
    blocks.artifact(tmp_path, block).write_text(json.dumps(dict(block=block['key'], products=original,
            sha256=blocks.content_key(original), plan_sha256=manifest['sha256'])))
    settings = dict(role='candidate', archive_plan=dict(path=str(pin), sha256=hashlib.sha256(pin.read_bytes()).hexdigest()))
    report = compare_archive(settings, plan, [products(.13)], 1)
    assert report['differences'] == 1 and report['panels']['d']['s']['first_difference']['column'] == 1
    with pytest.raises(ValueError, match='population'):
        compare_archive(settings, dict(plan, blocks=[]), [], 1)


def test_rejects_shuffled_latency_boundaries():
    times = dict(first_score_seconds=1., first_decision_seconds=3., scoring_seconds=2., differences=0)
    with pytest.raises(ValueError, match='host timing'):
        validate_trial(dict(standard=times, candidate=times), 2)


def test_endpoint_work_uses_wall_span_with_parallel_workers_and_nearest_rank_tail():
    from tools.reports.audit_card_endpoint_timing import metrics
    def row(start, end, first):
        return dict(claimed_unix=start, committed_unix=end, recordings=2, first_decision_seconds=first)
    result = metrics([row(100, 110, 2),row(105, 112, 5),row(110, 117, 3)])
    assert result == dict(blocks=3,recordings=6,claim_commit_span_seconds=17,
                          first_decision_p50=3,first_decision_p95=5)
