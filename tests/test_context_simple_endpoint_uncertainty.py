import copy
import pytest
from tools.reports.context_simple_endpoint_uncertainty import score_pair


def fixture():
    labels=[dict(id=str(i),source_group='g'+str(i//2),role='selection',intent='a') for i in range(6)]
    a=[dict(id=r['id'],policy_id='old',prob=[.2,.8],request_identity={'inputs':{'hash':r['id']},'positions':[0,1]}) for r in labels]
    b=[dict(r,policy_id='new',prob=[.8,.2]) for r in a]
    return a,b,labels


def test_fixed_endpoint_pairs_recompute_scores_and_intervals():
    a,b,labels=fixture();result,metrics=score_pair(a,b,labels,['a','b'],['old','new'])
    assert result['recordings']==6 and result['source_groups']==3
    assert result['metrics']['set_f1']=={'delta':1.,'percentile_95':[1.,1.]}
    assert metrics[0]['top1_event_accuracy']==0 and metrics[1]['top1_event_accuracy']==1
    assert metrics[1]['candidate_binary_nll']<metrics[0]['candidate_binary_nll']


@pytest.mark.parametrize('fault',['partial','policy','input','role','duplicate'])
def test_invalid_pair_rejected(fault):
    a,b,labels=fixture();b=copy.deepcopy(b)
    if fault=='partial':b=b[:-1]
    if fault=='policy':b[0]['policy_id']='other'
    if fault=='input':b[0]['request_identity']['inputs']={}
    if fault=='role':labels[0]['role']='test'
    if fault=='duplicate':labels[1]['id']=labels[0]['id']
    with pytest.raises(ValueError):score_pair(a,b,labels,['a','b'],['old','new'])
