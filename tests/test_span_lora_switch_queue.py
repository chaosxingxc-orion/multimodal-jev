"""Independent probability and actual adapter-switch acceptance failures."""
import copy
import math
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_span_lora_switch as module


def fixture(monkeypatch):
    data={};policies={};refs={}
    for key,p in [('A',.4),('B',.7)]:
        rows=[dict(id=str(i),request_id=f'{key}{i}',candidate_logprobs=[[math.log(p),math.log(1-p)]]) for i in range(178)]
        refs[key]=rows;policies[key]=dict(gate='/'+key,adapter_id=key)
        data[f'/{key}/run/vllm/result.json']=dict(records=rows)
    data['/probe/run/result.json']=dict(passed=True,sequence=['A','B','A'],a_return_exact=True)
    data['/probe/run/terminal.json']=dict(exit_code=0)
    for phase,key in enumerate(('A','B','A')):
        slot={'A':1,'B':2}[key]
        data[f'/probe/run/phase-{phase}.json']=dict(phase=phase,policy=key,slot=slot,adapter_id=key,
            worker=[dict(registered_adapter_ids=[slot],lora_dtype='torch.bfloat16',head_projection_dtype='torch.bfloat16',actual_lora_buffer_dtypes={'layer':['torch.bfloat16']})],records=copy.deepcopy(refs[key]))
    monkeypatch.setattr(module,'read',lambda p:data[str(p)])
    monkeypatch.setattr(module,'file_hash',lambda p:'hash')
    return dict(policies=policies),data


def test_complete_switch(monkeypatch):
    cfg,_=fixture(monkeypatch);r=module.verify_outputs(cfg,'/probe')
    assert r['passed'] and r['maximum_logprob_error']==0 and r['a_return_exact']


@pytest.mark.parametrize('bad',['stale','missing','duplicate','slot','dtype','request','return','nan','unnormalized','terminal'])
def test_rejects_false_switch(monkeypatch,bad):
    cfg,data=fixture(monkeypatch);phase=data['/probe/run/phase-1.json']
    if bad=='stale':phase['records'][0]['candidate_logprobs']=copy.deepcopy(data['/probe/run/phase-0.json']['records'][0]['candidate_logprobs'])
    elif bad=='missing':phase['records'].pop()
    elif bad=='duplicate':phase['records'][1]['id']='0'
    elif bad=='slot':phase['worker'][0]['registered_adapter_ids']=[1]
    elif bad=='dtype':phase['worker'][0]['actual_lora_buffer_dtypes']['layer']=['torch.float32']
    elif bad=='request':phase['records'][0]['request_id']='wrong'
    elif bad=='return':data['/probe/run/phase-2.json']['records'][0]['candidate_logprobs']=[[math.log(.40001),math.log(.59999)]]
    elif bad=='nan':phase['records'][0]['candidate_logprobs'][0][0]=float('nan')
    elif bad=='unnormalized':phase['records'][0]['candidate_logprobs'][0][0]=0.
    else:data['/probe/run/terminal.json']['exit_code']=1
    with pytest.raises(ValueError):module.verify_outputs(cfg,'/probe')
