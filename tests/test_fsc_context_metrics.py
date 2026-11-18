import copy,hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from fsc_context_metrics import score,compare


def fixture():
    ids=[f'c{i}' for i in range(31)]
    population=[dict(id='q1',speaker_id='s1',audio_sha256='audio1'),dict(id='q2',speaker_id='s2',audio_sha256='audio2')]
    receipts=[];records=[]
    for index,row in enumerate(population):
        inputs={'input_ids':{'shape':[1,40],'sha256':'input'+str(index)}}
        identity=dict(policy_id='p',inputs=inputs,positions=list(range(9,40)))
        prob=[.1]*31
        if index==0:prob[0]=.8;prob[1]=.7
        else:prob[1]=.4
        receipts.append(dict(id=row['id'],audio_sha256=row['audio_sha256'],positions=[identity['positions']],tensors={**inputs,'positions':{}}))
        records.append(dict(id=row['id'],policy_id='p',request_identity=identity,prob=prob,
            request_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    return records,population,{'q1':'c0','q2':'c1'},ids,receipts


def evaluate(data):
    records,population,targets,ids,receipts=data
    return score(records,population,targets,ids,policy_id='p',input_receipts=receipts)


def test_complete_set_top1_and_speaker_metrics_remain_distinct():
    result=evaluate(fixture())
    assert result['metrics']['top1_event_accuracy']==1
    assert result['metrics']['query_macro_set_f1']==pytest.approx(1/3)
    assert result['empty_selection_rate']==result['multiple_selection_rate']==.5
    assert len(result['per_speaker'])==2
    comparison=compare(result,copy.deepcopy(result))
    assert all(v==0 for v in comparison['delta'].values())


@pytest.mark.parametrize('fault',['missing','duplicate','policy','input','audio','nan','width','labels'])
def test_incomplete_or_changed_evidence_rejected(fault):
    data=fixture();records,population,targets,ids,receipts=data
    if fault=='missing':records.pop()
    elif fault=='duplicate':records[1]=copy.deepcopy(records[0])
    elif fault=='policy':records[1]['policy_id']='other'
    elif fault=='input':records[1]['request_identity']['positions']=[0]*31
    elif fault=='audio':receipts[1]['audio_sha256']='changed'
    elif fault=='nan':records[1]['prob'][0]=float('nan')
    elif fault=='width':records[1]['prob'].pop()
    elif fault=='labels':del targets['q2']
    with pytest.raises(ValueError):evaluate(data)


def test_transfer_pair_rejects_changed_gold():
    initial=evaluate(fixture());later=copy.deepcopy(initial);later['records'][0]['gold']='c1'
    with pytest.raises(ValueError):compare(initial,later)
