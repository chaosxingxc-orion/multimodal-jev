"""Exercise the fixed vLLM native LoRA parser, without starting an engine."""
import os
from pathlib import Path
import json

import pytest
import torch

pytest.importorskip('vllm')
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from multimodal_jev.context_policy_artifacts import read_execution_bundle, native_peft_payload
from multimodal_jev.context_policy_artifacts import export_native_adapter, verify_native_adapter


def test_real_joint_snapshot_native_parser_mapping_and_scaling(monkeypatch):
    # CPU-only container has no Ascend device for pinned host allocation.
    # Disable this transport allocation only; parser, mapper and tensors are real.
    import vllm.lora.lora_model as native
    monkeypatch.setattr(native, 'PIN_MEMORY', False)
    source = os.environ.get('JEV_CONTEXT_TEST_POLICY')
    if not source:
        pytest.skip('explicit real execution snapshot required')
    root = Path(source)
    manifest = json.loads((root/'MANIFEST.json').read_text())
    bundle = read_execution_bundle(root,expected_policy_id=manifest['policy_id'],
        expected_contract=manifest['contract'],expected_base_id=manifest['contract']['base_id'])
    config, tensors = native_peft_payload(bundle)
    helper = PEFTHelper.from_dict(config)
    loaded = LoRAModel.from_lora_tensors(1,tensors,helper,device='cpu',dtype=torch.bfloat16,
        weights_mapper=Qwen3ASRForConditionalGeneration.hf_to_vllm_mapper)
    specs = bundle['contract']['lora']
    assert len(loaded.loras) == len(specs) == 56
    assert set(loaded.loras) == {n.replace('language_model.','language_model.model.',1) for n in specs}
    for name,spec in specs.items():
        module = loaded.get_lora(name.replace('language_model.','language_model.model.',1))
        assert module.rank == spec['rank'] and module.scaling == spec['scaling']
        for matrix,actual in [('A',module.lora_a),('B',module.lora_b)]:
            expected = bundle['tensors'][f'backbone.{name}.lora_{matrix}.default.weight']
            assert actual.dtype == torch.bfloat16 and torch.equal(actual,expected)


def test_derived_adapter_roundtrip_and_corruption(tmp_path,monkeypatch):
    source = os.environ.get('JEV_CONTEXT_TEST_POLICY')
    if not source:
        pytest.skip('explicit real execution snapshot required')
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage,'check_paths',lambda paths:[])
    root = Path(source)
    manifest = json.loads((root/'MANIFEST.json').read_text())
    bundle = read_execution_bundle(root,expected_policy_id=manifest['policy_id'],
        expected_contract=manifest['contract'],expected_base_id=manifest['contract']['base_id'])
    target = tmp_path/'adapter'
    receipt = export_native_adapter(target,bundle)
    assert verify_native_adapter(target,bundle) == receipt
    assert receipt['complete_context_policy'] is False
    with pytest.raises(FileExistsError): export_native_adapter(target,bundle)
    altered = dict(bundle,policy_id='b'*64)
    with pytest.raises(ValueError): verify_native_adapter(target,altered)
    config = json.loads((target/'adapter_config.json').read_text())
    config['lora_alpha'] *= 2
    (target/'adapter_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError): verify_native_adapter(target,bundle)
