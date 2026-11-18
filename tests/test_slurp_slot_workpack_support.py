import copy
import pytest
from tools.reports.audit_slurp_slot_workpack_support import reconcile


def fixture():
    population=[dict(id=str(i),role='learning',official_split='train',known_pending_source=False,
        source_group='shared',audio_sha256=str(i),source_target_sha256='target',intent='event') for i in range(2)]
    sources=[dict(source_id=str(i),optimization_allowed=False,semantic_annotation=None,native_intent='event',
        recording_bindings=[dict(recording_id=str(i),source_group='shared',audio_sha256=str(i),
            source_target_sha256='target',all_source_ids=[str(i)])]) for i in range(2)]
    coords=[dict(source_id='0',definition_id='slot',status='current_reference_proposal',
        optimization_allowed=False,admitted_value=None)]
    return population,sources,coords,['slot']


def test_merged_group_does_not_propagate_reference_to_unassessed_source():
    r=reconcile(*fixture())
    assert r['source_ids']==2 and r['source_groups']==1
    assert r['merged_source_groups']==1 and r['unassessed_coordinates']==1
    assert r['admitted_coordinates']==0
    assert r['by_type'][0]['recording_support_by_status']=={'current_reference_proposal':1}


@pytest.mark.parametrize('fault',['audio','target','missing','admitted','duplicate','split'])
def test_inconsistent_source_or_admission_rejected(fault):
    p,s,c,t=fixture()
    if fault=='audio':s[0]['recording_bindings'][0]['audio_sha256']='bad'
    elif fault=='target':s[0]['recording_bindings'][0]['source_target_sha256']='bad'
    elif fault=='missing':s.pop()
    elif fault=='admitted':c[0]['admitted_value']=False
    elif fault=='duplicate':c.append(copy.deepcopy(c[0]))
    else:p[0]['official_split']='test'
    with pytest.raises(ValueError):reconcile(p,s,c,t)
