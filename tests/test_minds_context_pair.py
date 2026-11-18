"""Control-flow tests with model subprocess substitutes; no MINDS score claims."""
import json
from pathlib import Path
import pytest
from tools.reports import run_minds_context_pair as module


@pytest.fixture
def pair(tmp_path,monkeypatch):
    spec=dict(source_package=str(tmp_path/'source'),model=str(tmp_path/'model'),
        inputs=dict(inputs=str(tmp_path/'inputs'),cache_root=str(tmp_path/'cache')),
        source_manifest_sha256='source')
    configs={r:dict(policy_id=r,source_role=r) for r in ('initial','selected')}
    monkeypatch.setattr(module,'configs',lambda c:(configs,[],[]))
    from multimodal_jev import local_runtime_storage
    monkeypatch.setattr(local_runtime_storage,'require_training_storage',lambda p:None)
    monkeypatch.setattr(module,'require_parity',lambda *a:dict(exact=True))
    monkeypatch.setattr(module,'check_full',lambda folder,c,*a:dict(policy_id=c['policy_id']))
    return spec,tmp_path/'output'


@pytest.mark.parametrize('failure',[None,'initial-hf','selected-vllm','initial-inference','selected-inference'])
def test_both_panels_before_full_inference_and_failure_stops(pair,monkeypatch,failure):
    spec,output=pair;calls=[]
    def invoke(path,folder,phase):
        label=folder.name+'-'+phase;calls.append(label)
        if label==failure:raise ValueError('fixture failure')
        if phase=='vllm':module.write(folder/'parity.json',dict(exact=True))
    monkeypatch.setattr(module,'invoke',invoke)
    expected=['initial-hf','initial-vllm','selected-hf','selected-vllm','initial-inference','selected-inference']
    if failure:
        with pytest.raises(ValueError):module.run(spec,output)
        assert calls==expected[:expected.index(failure)+1]
        assert module.read(output/'terminal.json')['exit_code']==1
        assert not (output/'result.json').exists()
    else:
        assert module.run(spec,output)['target_tuning'] is False
        assert calls==expected and module.read(output/'terminal.json')['exit_code']==0
    with pytest.raises(FileExistsError):module.run(spec,output)


def test_false_panel_receipt_blocks_all_full_inference(pair,monkeypatch):
    spec,out=pair;calls=[]
    def invoke(path,folder,phase):
        calls.append(phase)
        if phase=='vllm':module.write(folder/'parity.json',dict(exact=False))
    monkeypatch.setattr(module,'invoke',invoke)
    with pytest.raises(ValueError,match='parity serialization'):module.run(spec,out)
    assert calls==['hf','vllm']


def test_inference_failure_does_not_change_source_or_start_training(pair,monkeypatch):
    spec,out=pair;before=json.dumps(spec,sort_keys=True)
    monkeypatch.setattr(module,'invoke',lambda *a:(_ for _ in ()).throw(RuntimeError('failed')))
    with pytest.raises(RuntimeError):module.run(spec,out)
    assert json.dumps(spec,sort_keys=True)==before
    assert module.read(out/'terminal.json')['phase']=='initial-hf'
