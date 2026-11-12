import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash

TOOLS=Path(__file__).resolve().parents[1]/'tools/jev'
sys.path.insert(0,str(TOOLS))
import verify_intent_binding_training_gate as module


def fixture(tmp_path):
    path=tmp_path/'launch.json'
    path.write_text(json.dumps(dict(container_id='exact-container',image='fixed-image')))
    return dict(acceptance_root=str(tmp_path),image='fixed-image',pins={str(path):file_hash(path)})


def test_live_gate_cannot_admit_even_with_ready_artifacts(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);observed=[]
    def completed(cid,image):observed.append((cid,image));return False
    def forbidden(*args):raise AssertionError('artifacts alone must not admit')
    monkeypatch.setattr(module,'completed',completed);monkeypatch.setattr(module,'authenticate_artifacts',forbidden)
    with pytest.raises(ValueError,match='successful gate exit'):module.verify(cfg,'policy')
    assert observed==[('exact-container','fixed-image')]


def test_exit_zero_requires_complete_artifact_authentication(tmp_path,monkeypatch):
    cfg=fixture(tmp_path)
    monkeypatch.setattr(module,'completed',lambda *args:True)
    def reject(*args):raise ValueError('full population rejected')
    monkeypatch.setattr(module,'authenticate_artifacts',reject)
    with pytest.raises(ValueError,match='full population rejected'):module.verify(cfg,'policy')


def test_observation_binds_same_gate(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);observed=[]
    monkeypatch.setattr(module,'completed',lambda cid,image:observed.append(cid) or True)
    monkeypatch.setattr(module,'authenticate_artifacts',lambda *args:(None,None,dict(gate_container_id='exact-container',requests=712,actual_exit_observed=False)))
    audit=module.verify(cfg,'policy')
    assert audit['actual_exit_observed'] and audit['actual_exit_code']==0 and observed==['exact-container']
    monkeypatch.setattr(module,'authenticate_artifacts',lambda *args:(None,None,dict(gate_container_id='different')))
    with pytest.raises(ValueError,match='identity changed'):module.verify(cfg,'policy')


def test_unpinned_launch_never_queries_docker(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);cfg['pins']={}
    def forbidden(*args):raise AssertionError('unknown container observation')
    monkeypatch.setattr(module,'completed',forbidden)
    with pytest.raises(ValueError,match='pinned launch'):module.verify(cfg,'policy')


def artifact_fixture(tmp_path,monkeypatch):
    import multimodal_jev.intent_binding_gate as gate
    import multimodal_jev.policy_identity as policy
    (tmp_path/'run/hf').mkdir(parents=True);(tmp_path/'run/vllm').mkdir()
    cfg=dict(acceptance_root=str(tmp_path),image='fixed-image',model_path='base-path',
        initial_adapter='adapter-path',runtime_profile='fixed-profile',binding_release='release',binding_manifest_sha256='manifest',pins={})
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));cfg['pins'][str(path)]=file_hash(path)
        return path
    inputs=put('inputs.json',{})
    put('source-manifest.json',{})
    put('config.json',dict(model='base-path',adapter='adapter-path',runtime_profile='fixed-profile',
        population=dict(recordings=89,requests=712),base_id='base',adapter_id='adapter',inputs=str(inputs),pins={str(inputs):file_hash(inputs)}))
    put('launch.json',dict(image='fixed-image',physical_devices=[4],container_id='exact-container',
        config_sha256=file_hash(tmp_path/'config.json'),source_manifest_sha256=file_hash(tmp_path/'source-manifest.json')))
    for name in ('terminal.json','hf/terminal.json','vllm/terminal.json'):put('run/'+name,dict(exit_code=0))
    put('run/hf/result.json',{});vl=put('run/vllm/result.json',{})
    put('run/READY.json',dict(passed=True,requests=712,base_id='base',adapter_id='adapter',result_sha256=file_hash(vl)))
    monkeypatch.setattr(policy,'checkpoint_identity',lambda path:('base' if path=='base-path' else 'adapter',{}))
    monkeypatch.setattr(gate,'load',lambda *args:([],{}))
    monkeypatch.setattr(gate,'verify_records',lambda *args:([{},{}],dict(passed=True,requests=712)))
    return cfg,put,gate


def test_artifact_check_does_not_claim_process_observation(tmp_path,monkeypatch):
    cfg,put,gate=artifact_fixture(tmp_path,monkeypatch)
    _,_,audit=gate.authenticate_artifacts(cfg,'base:adapter')
    assert audit['passed'] and not audit['actual_exit_observed']


@pytest.mark.parametrize('defect',['unpinned','phase','ready_population','ready_policy','result_hash','actual_weights','native_population'])
def test_artifact_custody_rejects_false_ready(tmp_path,monkeypatch,defect):
    cfg,put,gate=artifact_fixture(tmp_path,monkeypatch)
    if defect=='unpinned':cfg['pins'].pop(str(tmp_path/'run/hf/result.json'))
    elif defect=='phase':put('run/hf/terminal.json',dict(exit_code=1))
    elif defect=='actual_weights':
        import multimodal_jev.policy_identity as policy
        monkeypatch.setattr(policy,'checkpoint_identity',lambda *args:('different',{}))
    elif defect=='native_population':
        data=json.loads((tmp_path/'config.json').read_text());data['population']['requests']=178;put('config.json',data)
        launch=json.loads((tmp_path/'launch.json').read_text());launch['config_sha256']=file_hash(tmp_path/'config.json');put('launch.json',launch)
    else:
        data=json.loads((tmp_path/'run/READY.json').read_text())
        if defect=='ready_population':data['requests']=178
        elif defect=='ready_policy':data['adapter_id']='other'
        else:data['result_sha256']='other'
        put('run/READY.json',data)
    with pytest.raises(ValueError):gate.authenticate_artifacts(cfg,'base:adapter')
