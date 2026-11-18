import copy
import pytest
import torch

from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.slurp_training import content_hash
from tools.reports.score_minds_context_pair import score_records


def fixture():
    ids=[f'c{i}' for i in range(14)];inputs=[];receipts=[];truth={};predictions=[]
    for i in range(4):
        uid=str(i);inputs.append(dict(id=uid,audio_sha256='audio'))
        receipts.append(dict(id=uid,audio_sha256='audio',positions=[list(range(14))],tensors={'input_ids':{'sha256':str(i)}}))
        truth[uid]=dict(intent=ids[i],language=['en-US','en-GB','zh-CN','zh-CN'][i])
        identity=dict(policy_id='p',positions=list(range(14)),inputs=receipts[-1]['tensors'])
        logits=torch.full((14,),-2.);logits[i]=2.
        predictions.append(dict(id=uid,policy_id='p',request_identity=identity,request_id=digest(identity),
            prob=logits.sigmoid().tolist(),logits=logits.tolist(),output_protocol=WIRE))
    return predictions,inputs,receipts,truth,ids,'p'


def test_macro_preserves_all_language_and_probability_metrics_without_rewriting_ids():
    args=fixture();original=copy.deepcopy(args[0])
    macro=score_records(*args,request_identity_protocol='macro_compact_json_v1')
    assert args[0]==original
    legacy=copy.deepcopy(args)
    for row in legacy[0]:row['request_id']=content_hash(row['request_identity'])
    assert macro==score_records(*legacy)
    assert len(macro['languages'])==2 and len(macro['variants'])==3


@pytest.mark.parametrize('fault',['legacy_id','logit','input','unknown_protocol'])
def test_protocol_mismatch_or_corruption_refused(fault):
    args=fixture();protocol='macro_compact_json_v1'
    if fault=='legacy_id':args[0][0]['request_id']=content_hash(args[0][0]['request_identity'])
    if fault=='logit':args[0][0]['logits'][0]=0.
    if fault=='input':args[0][0]['request_identity']['positions']=[0]*14
    if fault=='unknown_protocol':protocol='auto_accept_any'
    with pytest.raises(ValueError):score_records(*args,request_identity_protocol=protocol)
