"""Actual tensor/identity checks with a fake engine; no backend acceptance."""
import importlib.util,sys,types
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')


def fixture(monkeypatch):
    # These tests exercise our engine interface, not vLLM internals. Stub its
    # transport classes when the local CPU development environment lacks vLLM.
    if importlib.util.find_spec('vllm') is None:
        v=types.ModuleType('vllm');l=types.ModuleType('vllm.lora');r=types.ModuleType('vllm.lora.request')
        v.lora=l;l.request=r;v.PoolingParams=object;r.LoRARequest=object
        for name,module in [('vllm',v),('vllm.lora',l),('vllm.lora.request',r)]:monkeypatch.setitem(sys.modules,name,module)
    path=Path(__file__).with_name('test_span_resident_rollout.py')
    spec=importlib.util.spec_from_file_location('behavior_test_helpers',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.fixture(monkeypatch)


def test_rollout_needs_no_hf_and_preserves_exact_policy_input(monkeypatch):
    engine,inputs,calls,_=fixture(monkeypatch)
    for adapter in ('A','B','A'):
        config,compiled,reference=inputs(adapter)
        actual=engine.rollout(config,compiled)
        assert actual['passed'] and actual['reference_compared'] is False
        assert actual['records'][0]['parity'] is None
        assert actual['records'][0]['candidate_logprobs']==reference['records'][0]['candidate_logprobs']
        assert actual['records'][0]['request_identity']==reference['records'][0]['request_identity']
    assert [c[0] for c in calls]==[1,2,1]


@pytest.mark.parametrize('fault',['missing','duplicate','wrong_policy','bad_expansion','nonfinite'])
def test_reference_free_does_not_remove_execution_checks(monkeypatch,fault):
    engine,inputs,_,_=fixture(monkeypatch);config,compiled,_=inputs()
    if fault=='missing':compiled=[]
    elif fault=='duplicate':compiled*=2
    elif fault=='wrong_policy':compiled[0][5]['policy_id']='base:foreign'
    else:
        original=engine.engine.encode
        def encode(*a,**kw):
            result=original(*a,**kw)
            if fault=='bad_expansion':result[0].prompt_token_ids=[]
            else:result[0].outputs.data.fill_(float('nan'))
            return result
        engine.engine.encode=encode
    with pytest.raises(ValueError):engine.rollout(config,compiled)
    assert engine.phase=='failed'
    with pytest.raises(ValueError):engine.rollout(*inputs()[:2])


def test_evaluation_still_requires_independent_reference(monkeypatch):
    engine,inputs,calls,_=fixture(monkeypatch);config,compiled,_=inputs()
    with pytest.raises(ValueError):engine.evaluate(config,compiled,None)
    assert not calls
