import copy
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.intent_behavior_replay import compare_batches

def item():
    return dict(request_id='request',policy_id='policy',candidates=[[1,2]],request_weight=1.,
        positions=torch.tensor([3]),targets=torch.tensor([[1]]),supervised=torch.tensor([[True]]),span_weights=torch.tensor([[1.]]),
        inputs=dict(input_features=torch.tensor([1.,2.]),input_ids=torch.tensor([3,4])),behavior=(torch.log(torch.tensor([[.3,.7]])),))

def test_exact_actor_batch():
    a=[item()];assert compare_batches(a,copy.deepcopy(a),{'ids':['1']},{'ids':['1']})['exact_inputs_and_supervision']

@pytest.mark.parametrize('bad',['audio','positions','targets','mask','weights','policy','request','candidates','behavior','selection'])
def test_changed_consumer_contract_rejected(bad):
    a=[item()];b=copy.deepcopy(a);selection={'ids':['1']};other=copy.deepcopy(selection)
    if bad=='audio':b[0]['inputs']['input_features'][0]+=1
    elif bad=='positions':b[0]['positions'][0]+=1
    elif bad=='targets':b[0]['targets'][0,0]=0
    elif bad=='mask':b[0]['supervised'][0,0]=False
    elif bad=='weights':b[0]['request_weight']=2.
    elif bad=='policy':b[0]['policy_id']='other'
    elif bad=='request':b[0]['request_id']='other'
    elif bad=='candidates':b[0]['candidates']=[[2,1]]
    elif bad=='behavior':b[0]['behavior']=(torch.log(torch.tensor([[.7,.3]])),)
    else:other['ids']=['different']
    with pytest.raises(ValueError):compare_batches(a,b,selection,other)
