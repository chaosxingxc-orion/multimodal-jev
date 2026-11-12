"""Fixed-image CPU contract tests; no actual engine switch asserted."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import probe_span_lora_switch as module


def fixture(monkeypatch):
    configs={};data={};pins={}
    for key in ('A','B'):
        root='/'+key;adapter='/adapter'+key
        c=dict(adapter=adapter,adapter_id=key,base_id='base',model='/base',runtime_profile='fixed',inputs='/same',population={'recordings':89,'requests':178})
        vl=dict(passed=True,base_id='base',adapter_id=key,records=[dict(id=str(i),candidate_logprobs=[[-1.,-2. if key=='A' else -3.]]) for i in range(178)])
        data[root+'/config.json']=c;data[root+'/run/vllm/result.json']=vl
        configs[key]={'gate':root,'adapter_id':key}
        for name in ('config.json','run/vllm/result.json','run/hf/result.json','run/READY.json'):pins[root+'/'+name]='hash'
    monkeypatch.setattr(module,'read',lambda p:data[str(p)])
    monkeypatch.setattr(module,'file_hash',lambda p:'hash')
    monkeypatch.setattr(module,'checkpoint_identity',lambda p:(str(p)[-1],{}))
    import queue_span_dp4_resume
    monkeypatch.setattr(queue_span_dp4_resume,'check_pins',lambda p:None)
    cfg=dict(sequence=['A','B','A'],policies=configs,max_model_len=4096,physical_devices=[4],model='/base',base_id='base',runtime_profile='fixed',pins=pins)
    return cfg,data


def test_requires_distinct_policy_and_complete_same_population(monkeypatch):
    cfg,data=fixture(monkeypatch);assert module.validate(cfg)[2]==1
    data['/B/run/vllm/result.json']['records']=copy.deepcopy(data['/A/run/vllm/result.json']['records'])
    with pytest.raises(ValueError,match='insufficiently distinct'):module.validate(cfg)


@pytest.mark.parametrize('bad',['duplicate','missing','inputs','policy','sequence','pins'])
def test_rejects_invalid_switch_contract(monkeypatch,bad):
    cfg,data=fixture(monkeypatch)
    if bad=='duplicate':data['/B/run/vllm/result.json']['records'][1]['id']='0'
    elif bad=='missing':data['/B/run/vllm/result.json']['records'].pop()
    elif bad=='inputs':data['/B/config.json']['inputs']='/different'
    elif bad=='policy':data['/B/config.json']['adapter_id']='A'
    elif bad=='sequence':cfg['sequence']=['A','B']
    else:cfg['pins'].pop('/A/run/hf/result.json')
    with pytest.raises(ValueError):module.validate(cfg)
