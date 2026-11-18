import copy,importlib.util,math
from pathlib import Path
import pytest
pytest.importorskip('torch')
spec=importlib.util.spec_from_file_location('switch_audit',Path(__file__).parents[1]/'tools/jev/verify_simple_sampled_switch.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    refs=[];results=[]
    for i,adapter in enumerate(('old','new')):
        probs=[math.log(.5)]+[math.log(.5/59)]*59 if i else [-math.log(60)]*60
        rows=[dict(id=str(j),request_id=f'{adapter}:{j}',request_identity={'policy_id':'base:'+adapter,'input':j},candidate_logprobs=[probs]) for j in range(256)]
        refs.append(dict(backend='hf',base_id='base',adapter_id=adapter,records=rows))
        worker=dict(registered_adapter_ids=[i+1],lora_dtype='torch.bfloat16',head_projection_dtype='torch.bfloat16',actual_lora_buffer_dtypes={'q':['torch.bfloat16']})
        results.append(dict(passed=True,backend='vllm',base_id='base',adapter_id=adapter,slot=i+1,worker=[worker],records=copy.deepcopy(rows)))
    results.append(copy.deepcopy(results[0]));return results,refs


def test_three_actual_policy_outputs_are_compared_against_distinct_references():
    r,f=fixture();v=m.compare_results(r,f,base_id='base',adapter_ids=['old','new'])
    assert v['exact_switch_back'] and v['maximum_logprob_error']==0


def test_behavior_collection_has_no_reference_but_matches_accepted_execution():
    r,_=fixture();b=copy.deepcopy(r[1]);b['reference_compared']=False
    for row in b['records']:row['parity']=None
    assert m.compare_behavior(b,r[1])['exact_resident_behavior']
    b['records'][0]['candidate_logprobs']=[[-math.log(60)]*60]
    with pytest.raises(ValueError):m.compare_behavior(b,r[1])


@pytest.mark.parametrize('fault',['reference','partial','stale','parity'])
def test_reject_false_reference_free_collection(fault):
    r,_=fixture();b=copy.deepcopy(r[1]);b['reference_compared']=False
    if fault=='reference':b['reference_compared']=True
    elif fault=='partial':b['records'].pop()
    elif fault=='stale':b['adapter_id']='old'
    else:b['records'][0]['parity']={'passed':True}
    with pytest.raises(ValueError):m.compare_behavior(b,r[1])


@pytest.mark.parametrize('fault',['stale','slot','worker','dtype','partial','duplicate','request','probability','switch_back'])
def test_reject_stale_or_partial_switch_even_if_passed_true(fault):
    r,f=fixture()
    if fault=='stale':r[1]['records']=copy.deepcopy(r[0]['records'])
    elif fault=='slot':r[1]['slot']=1
    elif fault=='worker':r[1]['worker'][0]['registered_adapter_ids']=[1]
    elif fault=='dtype':r[1]['worker'][0]['actual_lora_buffer_dtypes']['q']=['torch.float32']
    elif fault=='partial':r[1]['records'].pop()
    elif fault=='duplicate':r[1]['records'][-1]=r[1]['records'][0]
    elif fault=='request':r[1]['records'][0]['request_identity']={'different':True}
    elif fault=='probability':r[1]['records'][0]['candidate_logprobs']=[[-math.log(60)]*60]
    else:
        # Within ordinary parity tolerance but not exact old→new→old recovery.
        import torch
        z=torch.zeros(60,dtype=torch.float64);z[0]=1e-5
        r[2]['records'][0]['candidate_logprobs']=[z.log_softmax(-1).tolist()]
    with pytest.raises(ValueError):m.compare_results(r,f,base_id='base',adapter_ids=['old','new'])
