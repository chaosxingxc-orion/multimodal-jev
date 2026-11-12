import copy
import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_span_resident_policy_suite as module
from multimodal_jev.baseline import atomic_json


def fixture():
    c=dict(base_id='base',adapter_id='A',population={'requests':2})
    a=dict(records=[{'id':'one'},{'id':'two'}])
    h=dict(passed=True,backend='hf',base_id='base',adapter_id='A',records=[
        dict(id=k,request_id=k+'-A',request_identity={'policy':'A'},candidate_logprobs=[[math.log(.4),math.log(.6)]]) for k in ['one','two']])
    v=copy.deepcopy(h);v['backend']='vllm'
    return c,a,h,v


@pytest.mark.parametrize('bad',['probability','request','policy','duplicate','missing','nan'])
def test_recompute_rejects_claimed_success(bad):
    c,a,h,v=fixture()
    if bad=='probability':v['records'][0]['candidate_logprobs']=[[math.log(.9),math.log(.1)]]
    elif bad=='request':v['records'][0]['request_id']='wrong'
    elif bad=='policy':v['adapter_id']='B'
    elif bad=='duplicate':v['records'][1]=v['records'][0]
    elif bad=='missing':v['records'].pop()
    else:v['records'][0]['candidate_logprobs'][0][0]=float('nan')
    with pytest.raises(ValueError):module.verify_pair(c,a,h,v)


def test_all_candidates_recomputed():
    assert module.verify_pair(*fixture())==dict(requests=2,maximum_logprob_error=0.)


@pytest.mark.parametrize('failure',[None,'hf','rollout','shutdown'])
def test_all_hf_exit_before_one_engine_and_no_premature_ready(tmp_path,monkeypatch,failure):
    import check_slurp_native_span_policy as native
    import multimodal_jev.span_resident_rollout as rollout
    c,a,h,v=fixture();c.update(adapter='path',pins={})
    configs=[(key,copy.deepcopy(c),a) for key in ['first','second']];events=[]
    monkeypatch.setattr(module,'validate',lambda *args:configs)
    monkeypatch.setattr(native,'prepared',lambda *args:iter([]))
    def hf(cfg,path,out):
        events.append('hf-'+path.parent.name)
        if failure=='hf':raise ValueError('injected HF failure')
        out.mkdir();atomic_json(out/'result.json',h);atomic_json(out/'terminal.json',{'exit_code':0});return h
    monkeypatch.setattr(module,'run_hf',hf)
    class Engine:
        runtime_receipt={'profile':'fixed'}
        def __init__(self,*args):
            assert events==['hf-first','hf-second'];events.append('engine-created')
        def evaluate(self,*args):
            assert not list(tmp_path.glob('*/run/READY.json'))
            events.append('evaluate')
            if failure=='rollout':raise ValueError('injected rollout failure')
            return dict(v,worker=[],slot=1)
        def close(self):
            assert not list(tmp_path.glob('*/run/READY.json'));events.append('closed')
            if failure=='shutdown':raise ValueError('injected shutdown failure')
    monkeypatch.setattr(rollout,'ResidentSpanRollout',Engine)
    if failure:
        with pytest.raises(ValueError,match='injected'):module.run({'pins':{}},tmp_path)
        assert not list(tmp_path.glob('*/run/READY.json'))
        if failure!='hf':assert events[-1]=='closed'
    else:
        result=module.run({'pins':{}},tmp_path)
        assert result['passed'] and len(result['policies'])==2
        assert events==['hf-first','hf-second','engine-created','evaluate','evaluate','closed']
        assert len(list(tmp_path.glob('*/run/READY.json')))==2
