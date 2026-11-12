import copy
import pytest
from multimodal_jev.intent_supervision_audit import verify,float32


def fixture(arm):
    w=[1.] if arm=='D' else ([.1,.1,1.] if arm=='P' else [0.,0.,1.])
    target=dict(targets=[1]*len(w),supervised=[True]*len(w),span_weights=w,request_weight=.75)
    selected=[dict(id='audio',target=target)]
    behavior={'records':[dict(id='audio::'+('D' if arm=='D' else 'Q'),request_id='bound',
        candidate_logprobs=[[-.7,-.7]]*len(w))]}
    saved=[dict(request_id='bound',targets=[target['targets']],supervised=[target['supervised']],
        span_weights=[[float32(x) for x in w]],request_weight=.75)]
    return selected,behavior,copy.deepcopy(saved)


@pytest.mark.parametrize('arm',['D','Q','P'])
def test_registered_span_budget_matches_exact_saved_fp32(arm):
    selected,behavior,saved=fixture(arm)
    assert verify(selected,behavior,saved,arm)['passed']
    if arm=='P':assert saved[0]['span_weights'][0][0]!=.1


@pytest.mark.parametrize('defect',['zero_process','double_weight','gold','mask','bool_target','request','source_weight','extra','duplicate'])
def test_wrong_supervision_is_not_accepted(defect):
    selected,behavior,saved=copy.deepcopy(fixture('P'))
    if defect=='zero_process':saved[0]['span_weights'][0][0]=0.
    if defect=='double_weight':saved[0]['span_weights'][0][0]=float32(.01)
    if defect=='gold':saved[0]['targets'][0][0]=0
    if defect=='mask':saved[0]['supervised'][0][0]=False
    if defect=='bool_target':saved[0]['targets'][0][0]=True
    if defect=='request':saved[0]['request_id']='stale'
    if defect=='source_weight':saved[0]['request_weight']=1.
    if defect=='extra':saved[0]['other']=True
    if defect=='duplicate':behavior['records']*=2
    with pytest.raises(ValueError):verify(selected,behavior,saved,'P')
