import copy
import pytest
from tools.reports.compare_multidevice_evaluation import compare


def predictions():
    return dict(records=[dict(id=str(i),policy_id='p',request_id=str(i),request_identity={'audio':str(i)},prob=[.2,.8],logits=[-1.,1.]) for i in range(3)])


def test_exact_equal_outputs_and_real_count_throughput():
    a=predictions();result=compare(a,copy.deepcopy(a),left_seconds=6.,right_seconds=3.)
    assert result['inference_speedup']==2 and result['right_queries_per_second']==1


@pytest.mark.parametrize('fault',['missing','order','logit','probability','policy','input','duplicate','duration'])
def test_per_record_equality_gate(fault):
    a=predictions();b=copy.deepcopy(a);seconds=3.
    if fault=='missing':b['records'].pop()
    elif fault=='order':b['records'].reverse()
    elif fault=='logit':b['records'][0]['logits'][0]+=1e-8
    elif fault=='probability':b['records'][0]['prob'][0]+=.01
    elif fault=='policy':b['records'][0]['policy_id']='other'
    elif fault=='input':b['records'][0]['request_identity']={}
    elif fault=='duplicate':a['records'][1]=a['records'][0]
    else:seconds=float('nan')
    with pytest.raises(ValueError):compare(a,b,left_seconds=6.,right_seconds=seconds)
