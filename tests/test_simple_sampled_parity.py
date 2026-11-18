import hashlib,json,math
import pytest
pytest.importorskip('torch')
from multimodal_jev.simple_sampled_parity import compare_resident


def fixture():
    resident=[];backend=[];sampling=[]
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for i in range(256):
        identity=dict(policy_id='new',input=i);prior=dict(identity,policy_id='old')
        old_id=digest(prior);logp=[[-math.log(60)]*60]
        backend.append(dict(request_id=digest(identity),request_identity=identity,candidate_logprobs=logp))
        if i<128:
            uid=f'{old_id}::update0-slot{i}'
            resident.append(dict(request_id=uid,candidate_logprobs=logp))
            sampling.append(dict(request_id=old_id,actor_request_id=uid,policy_id='old',update=0,slot=i))
    return resident,backend,sampling


def test_full_candidate_parity_across_policy_version_identity():
    r,b,s=fixture();report=compare_resident(r,b,s,old_policy_id='old',new_policy_id='new')
    assert report['requests']==128 and report['maximum_logprob_error']==0


def test_explicit_second_update_does_not_reuse_zero_cursor():
    r,b,s=fixture()
    for a,c in zip(r,s):
        a['request_id']=a['request_id'].replace('::update0-', '::update1-')
        c['actor_request_id']=a['request_id'];c['update']=1
    assert compare_resident(r,b,s,old_policy_id='old',new_policy_id='new',update=1)['passed']
    with pytest.raises(ValueError):compare_resident(r,b,s,old_policy_id='old',new_policy_id='new')


@pytest.mark.parametrize('fault',['partial','duplicate','policy','input','slot','sampling','probabilities','missing_candidate'])
def test_reject_stale_partial_or_changed_outputs(fault):
    r,b,s=fixture()
    if fault=='partial':r.pop()
    elif fault=='duplicate':r[-1]=r[0]
    elif fault=='policy':b[0]['request_identity']['policy_id']='stale'
    elif fault=='input':b[0]['request_identity']['input']='different'
    elif fault=='slot':s[0]['slot']=1
    elif fault=='sampling':s[0]['policy_id']='other'
    elif fault=='probabilities':r[0]['candidate_logprobs']=[[math.log(.5)]+[math.log(.5/59)]*59]
    else:r[0]['candidate_logprobs']=[[-math.log(59)]*59]
    with pytest.raises(ValueError):compare_resident(r,b,s,old_policy_id='old',new_policy_id='new')
