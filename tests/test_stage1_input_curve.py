import hashlib,json
from pathlib import Path
import pytest
from tools.reports.finalize_stage1_input_curve import input_identity,file_hash


def artifact(root,policy,**change):
    row=dict(id='sample',audio_sha256='audio',request_sha256='request',candidate_ids=['a'],positions=[1],inputs={'audio':'hash'})
    row.update(change)
    request=dict(policy_id=policy,inputs=row['inputs'],positions=row['positions'])
    row['engine_request_id']=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    p=root/'evaluation/inference/input-receipts.json';p.parent.mkdir(parents=True);p.write_text(json.dumps([row]))
    return dict(collection=str(root)),dict(inference_audit=dict(input_receipts_sha256=file_hash(p)),result=dict(policy_id=policy))


def test_same_inputs_different_policies_equal(tmp_path):
    a=artifact(tmp_path/'a','policyA');b=artifact(tmp_path/'b','policyB')
    assert input_identity(*a)==input_identity(*b)


@pytest.mark.parametrize('change',[{'audio_sha256':'other'},{'positions':[2]},{'inputs':{'audio':'different'}},{'candidate_ids':['b']}])
def test_real_input_changes_remain_distinguishable(tmp_path,change):
    assert input_identity(*artifact(tmp_path/'a','A'))!=input_identity(*artifact(tmp_path/'b','B',**change))


def test_stale_policy_request_rejected(tmp_path):
    b,p=artifact(tmp_path/'a','A');p['result']['policy_id']='B'
    with pytest.raises(ValueError,match='checkpoint'):input_identity(b,p)
