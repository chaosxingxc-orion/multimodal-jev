from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
from queue_intent_runtime_admission import verify_suite_terminal


def test_acceptance_binds_actual_verifier_evidence(tmp_path):
    evidence=tmp_path/'rank-state';evidence.write_text('actual snapshot')
    atomic_json(tmp_path/'controller-terminal.json',{'exit_code':0})
    atomic_json(tmp_path/'config.json',{'pins':{str(evidence):file_hash(evidence)}})
    atomic_json(tmp_path/'verified.json',dict(native_actor_verified=True,matched_bf16_export_exact=True,files={str(evidence):file_hash(evidence)}))
    atomic_json(tmp_path/'ACCEPTED.json',dict(passed=True,verified_sha256=file_hash(tmp_path/'verified.json')))
    assert verify_suite_terminal(tmp_path)['passed']
    evidence.write_text('different snapshot')
    with pytest.raises(ValueError):verify_suite_terminal(tmp_path)


def test_success_receipt_cannot_override_failed_controller(tmp_path):
    atomic_json(tmp_path/'controller-terminal.json',{'exit_code':1})
    atomic_json(tmp_path/'ACCEPTED.json',{'passed':True})
    with pytest.raises(ValueError,match='controller failed'):verify_suite_terminal(tmp_path)
