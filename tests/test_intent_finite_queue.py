import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
from queue_intent_finite_npu import freeze_training,verify_final


def test_training_freeze_requires_future_behavior_and_refuses_reentry(tmp_path):
    root=tmp_path/'training';root.mkdir();(root/'source').mkdir()
    (root/'source/code.py').write_text('fixed')
    atomic_json(root/'source-manifest.json',{'code.py':file_hash(root/'source/code.py')})
    atomic_json(root/'config-template.json',{'pins':{}})
    behavior=tmp_path/'behavior.json'
    plan=dict(root=str(root),config_template_sha256=file_hash(root/'config-template.json'),
        source_manifest_sha256=file_hash(root/'source-manifest.json'),pending_pins=[str(behavior)],
        create_argv=['docker','create','-v',f'{tmp_path}:{tmp_path}:ro'])
    with pytest.raises(FileNotFoundError):freeze_training(plan)
    assert not (root/'config.json').exists()
    behavior.write_text('actual completed behavior')
    cfg=freeze_training(plan)
    assert cfg['pins'][str(behavior)]==file_hash(behavior)
    with pytest.raises(ValueError,match='already claimed'):freeze_training(plan)


def test_final_receipt_cannot_hide_failure_or_changed_state(tmp_path):
    evidence=tmp_path/'state';evidence.write_text('actual optimizer')
    atomic_json(tmp_path/'config.json',{'pins':{}})
    atomic_json(tmp_path/'controller-terminal.json',{'exit_code':0})
    atomic_json(tmp_path/'verified.json',dict(passed=True,exact_export_reload=True,full_gate_requests=178,
        files={str(evidence):file_hash(evidence)}))
    atomic_json(tmp_path/'ACCEPTED.json',dict(passed=True,verified_sha256=file_hash(tmp_path/'verified.json')))
    assert verify_final(tmp_path)['passed']
    evidence.write_text('changed')
    with pytest.raises(ValueError,match='asset changed'):verify_final(tmp_path)
    atomic_json(tmp_path/'controller-terminal.json',{'exit_code':1})
    with pytest.raises(ValueError,match='controller failed'):verify_final(tmp_path)
