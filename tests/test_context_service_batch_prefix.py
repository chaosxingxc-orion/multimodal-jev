import copy
import pytest
from tools.reports.compare_context_service_batch_prefix import compare_rows


def rows():
    receipt = dict(policy_id='p', head_marker_exact=True, native_loaded_tensors=112)
    return [dict(update=1, records_sha256='same',
        activation=dict(policy_id='p', receipt=dict(receipt, replica_receipts=[receipt, receipt])),
        ranks=[dict(total_seconds=10., preparation_seconds=5., actor_seconds=3.)]*2,
        rollout=dict(update=0, reference_misses=128, total_seconds=4., reference_seconds=1.,
                     behavior_and_credit_seconds=2.))]


def test_matched_prefix_never_admits_full_run():
    a=rows(); b=copy.deepcopy(a)
    for rank in b[0]['ranks']: rank['total_seconds']=8.
    r=compare_rows(a,b)
    assert r['speedup']==1.25
    assert not r['winner_selection_allowed'] and not r['full_100step_result']
    assert r['candidate']['exposures']==128


@pytest.mark.parametrize('fault', ['samples','cache','gap','nan','nested','activation'])
def test_rejects_invalid_comparisons(fault):
    a=rows(); b=copy.deepcopy(a)
    if fault=='samples': b[0]['records_sha256']='different'
    if fault=='cache': b[0]['rollout']['reference_misses']=0
    if fault=='gap': b[0]['update']=2
    if fault=='nan': b[0]['ranks'][0]['total_seconds']=float('nan')
    if fault=='nested': b[0]['rollout']['total_seconds']=6.
    if fault=='activation': b[0]['activation']['receipt']['replica_receipts']=[]
    with pytest.raises(ValueError): compare_rows(a,b)
