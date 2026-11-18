from copy import deepcopy

import pytest

from tools.reports.audit_context_following_snapshot import check_update


def fixture():
    records = [dict(checkpoint=False, real_queries=128, total_seconds=100.+rank,
        behavior_parity_history=[[dict(passed=True, threshold_flips=0) for _ in range(4)]
                                for _ in range(4)]) for rank in range(4)]
    activation = dict(error=None, policy_id='p', receipt=dict(policy_id='p',
        head_marker_exact=True, native_loaded_tensors=112))
    return records, activation, dict(recovery_checkpoint=False), dict(real_queries=128), 'p'


def test_all_ranks_full_step_time():
    assert check_update(*fixture()) == 103.


@pytest.mark.parametrize('fault', ['missing_rank', 'nan', 'parity', 'flip', 'activation', 'population', 'checkpoint'])
def test_incomplete_or_conflicting_evidence_rejected(fault):
    args = deepcopy(fixture()); records, activation, export, _, _ = args
    if fault == 'missing_rank': records.pop()
    if fault == 'nan': records[3]['total_seconds'] = float('nan')
    if fault == 'parity': records[3]['behavior_parity_history'][3][3]['passed'] = False
    if fault == 'flip': records[3]['behavior_parity_history'][3][3]['threshold_flips'] = 1
    if fault == 'activation': activation['receipt']['head_marker_exact'] = False
    if fault == 'population': records[3]['real_queries'] = 127
    if fault == 'checkpoint': export['recovery_checkpoint'] = True
    with pytest.raises(ValueError): check_update(*args)
