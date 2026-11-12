import json
from pathlib import Path
import sys
import types
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_binding_acceptance as module
from multimodal_jev.baseline import atomic_json,file_hash
from test_intent_binding_resume import configs


def fixture(tmp_path):
    parent=tmp_path/'training';parent.mkdir();suite=tmp_path/'suite';suite.mkdir()
    cfg,_=configs(tmp_path);atomic_json(parent/'config.json',cfg)
    atomic_json(suite/'config.json',dict(training_root=str(parent),adapter=str(parent/'run/adapter'),
        population=dict(recordings=89,requests=712),pins={str(parent/'config.json'):file_hash(parent/'config.json')}))
    return parent,suite


def test_backend_completion_precedes_state_acceptance(tmp_path,monkeypatch):
    parent,suite=fixture(tmp_path);events=[]
    def backend(argv,check):
        assert check and '--output' in argv;events.append('backend')
    def verify(p,s):
        assert p==parent and s==suite and events==['backend'];events.append('state')
        return dict(passed=True,adapter_id='new',scope='synthetic test')
    monkeypatch.setattr(module.subprocess,'run',backend)
    monkeypatch.setitem(sys.modules,'verify_intent_binding_npu',types.SimpleNamespace(verify=verify))
    module.run(suite)
    assert events==['backend','state']
    accepted=json.loads((suite/'ACCEPTED.json').read_text())
    assert accepted['verified_sha256']==file_hash(suite/'verified.json')


@pytest.mark.parametrize('defect',['native_population','wrong_adapter','backend_failure','state_failure'])
def test_failure_never_publishes_acceptance(tmp_path,monkeypatch,defect):
    parent,suite=fixture(tmp_path)
    config=json.loads((suite/'config.json').read_text())
    if defect=='native_population':config['population']['requests']=178
    if defect=='wrong_adapter':config['adapter']='/wrong'
    atomic_json(suite/'config.json',config)
    def backend(*args,**kwargs):
        if defect in ('native_population','wrong_adapter'):raise AssertionError('invalid suite reached backend')
        if defect=='backend_failure':raise RuntimeError('backend failed')
    def verify(*args):raise RuntimeError('state failed')
    monkeypatch.setattr(module.subprocess,'run',backend)
    monkeypatch.setitem(sys.modules,'verify_intent_binding_npu',types.SimpleNamespace(verify=verify))
    with pytest.raises((ValueError,RuntimeError)):module.run(suite)
    assert not (suite/'ACCEPTED.json').exists()
