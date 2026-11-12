import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import freeze_intent_binding_training as module
from multimodal_jev.baseline import atomic_json,file_hash
from test_intent_binding_resume import configs


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'train';root.mkdir();gate=tmp_path/'gate';gate.mkdir()
    cfg,_=configs(tmp_path);cfg.update(arm='D',acceptance_root=str(gate))
    names=('launch.json','run/READY.json','run/terminal.json','run/hf/result.json','run/hf/terminal.json',
        'run/vllm/result.json','run/vllm/terminal.json')
    for name in names:
        p=gate/name;p.parent.mkdir(parents=True,exist_ok=True);atomic_json(p,{})
    atomic_json(root/'config-template.json',cfg);atomic_json(root/'source-manifest.json',{})
    plan=dict(root=str(root),physical_devices=[4,5,6,7],image=cfg['image'],
        config_template_sha256=file_hash(root/'config-template.json'),source_manifest_sha256=file_hash(root/'source-manifest.json'),
        pending_pins=[str(gate/n) for n in names],create_argv=['synthetic'])
    atomic_json(root/'prepared.json',plan)
    monkeypatch.setattr(module,'validate_bind_pins',lambda *args:dict(passed=True))
    return root,gate,plan


def test_freeze_only_after_complete_observed_policy(tmp_path,monkeypatch):
    root,gate,plan=fixture(tmp_path,monkeypatch);calls=[]
    def verify(cfg,policy):
        assert not (root/'config.json').exists()
        assert set(plan['pending_pins'])<=cfg['pins'].keys()
        calls.append(policy)
        return dict(actual_exit_observed=True,requests=712,gate_container_id='exact')
    monkeypatch.setattr(module,'verify',verify)
    result=module.freeze(root,'base:adapter')
    assert calls==['base:adapter'] and result['status']=='frozen_not_dispatched'
    assert result['config_sha256']==file_hash(root/'config.json')
    with pytest.raises(ValueError,match='already claimed'):module.freeze(root,'base:adapter')


@pytest.mark.parametrize('defect',['running','bad_population','missing_pending','changed_template','mount_failure','claimed'])
def test_failed_gate_never_publishes_training_config(tmp_path,monkeypatch,defect):
    root,gate,plan=fixture(tmp_path,monkeypatch)
    def verify(*args):
        if defect=='running':raise ValueError('gate running')
        return dict(actual_exit_observed=True,requests=178 if defect=='bad_population' else 712,gate_container_id='exact')
    monkeypatch.setattr(module,'verify',verify)
    if defect=='missing_pending':plan['pending_pins'].pop();atomic_json(root/'prepared.json',plan)
    elif defect=='changed_template':(root/'config-template.json').write_text((root/'config-template.json').read_text()+' ')
    elif defect=='mount_failure':
        def fail(*args):raise ValueError('missing mount')
        monkeypatch.setattr(module,'validate_bind_pins',fail)
    elif defect=='claimed':atomic_json(root/'create-attempt.json',{})
    with pytest.raises(ValueError):module.freeze(root,'base:adapter')
    assert not (root/'config.json').exists()
