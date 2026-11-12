import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
from queue_intent_finite_npu import freeze_training,verify_final,await_gate_dispatch


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


def test_follow_exact_gate_dispatch_rejects_replaced_receipt(tmp_path):
    queue=tmp_path/'queue';gate=tmp_path/'gate';out=tmp_path/'output'
    for p in (queue/'run',gate,out):p.mkdir(parents=True)
    atomic_json(queue/'launch.json',dict(pid=0,argv=['original']))
    atomic_json(queue/'run/terminal.json',{'exit_code':0})
    receipt=dict(root=str(gate),image='fixed',container_id='exact')
    atomic_json(queue/'run/dispatched.json',receipt);atomic_json(gate/'launch.json',receipt)
    cfg=dict(gate_queue=str(queue),gate_root=str(gate),image='fixed',
        pins={str(queue/'launch.json'):file_hash(queue/'launch.json')})
    assert await_gate_dispatch(cfg,out)==receipt
    atomic_json(gate/'launch.json',dict(receipt,container_id='replacement'))
    with pytest.raises(ValueError,match='receipt differs'):await_gate_dispatch(cfg,out)
    atomic_json(queue/'run/terminal.json',{'exit_code':1})
    with pytest.raises(ValueError,match='queue failed'):await_gate_dispatch(cfg,out)
