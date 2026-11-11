from dataclasses import replace
import copy
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_request_identity import bind_span_request
from multimodal_jev.span_contract import SpanRequest,SpanQuestion
from multimodal_jev.task_schema import DecisionSchema


def fixture():
    q=SpanQuestion('final',DecisionSchema('fixture','noul','Is condition met?',allow_unknown=False))
    r=SpanRequest('/same.wav',(q,),'final',('Use audio evidence.',))
    receipt={'input_waveform_sha256':'a'*64,'output_waveform_sha256':'b'*64,'config':{},'sample_rate':16000}
    inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                input_features=torch.ones(1,2,4),input_features_mask=torch.ones(1,4,dtype=torch.long))
    return r,receipt,inputs,torch.tensor([[1]]),[[10,11]],'policy-v1'


def test_exact_request_identity_changes_with_evidence_question_candidates_or_policy():
    original=fixture();uid,payload=bind_span_request(*original)
    assert bind_span_request(*copy.deepcopy(original))[0]==uid
    variants=[]
    x=list(copy.deepcopy(original));x[2]['input_features'][0,0,0]=2;variants.append(x)
    x=list(copy.deepcopy(original));x[0]=replace(x[0],requirements=('Different requirement.',));variants.append(x)
    x=list(copy.deepcopy(original));x[4]=[[11,10]];variants.append(x)
    x=list(copy.deepcopy(original));x[5]='policy-v2';variants.append(x)
    x=list(copy.deepcopy(original));x[1]['output_waveform_sha256']='c'*64;variants.append(x)
    assert all(bind_span_request(*x)[0]!=uid for x in variants)
    assert 'targets' not in payload and 'labels' not in payload['inputs']


def test_teacher_labels_and_unknown_fields_rejected():
    x=list(fixture());x[2]['labels']=torch.tensor([[1]])
    with pytest.raises(ValueError,match='label-free'):bind_span_request(*x)
