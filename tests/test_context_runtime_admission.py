from types import SimpleNamespace as N
import subprocess
import sys

import pytest
import torch

pytest.importorskip('vllm')
from multimodal_jev.vllm_context_model import validate_context_runtime, register_context_model


def config():
    return N(parallel_config=N(tensor_parallel_size=1,pipeline_parallel_size=1),
        scheduler_config=N(enable_chunked_prefill=False),cache_config=N(enable_prefix_caching=False),
        quant_config=None,lora_config=N(),model_config=N(dtype=torch.bfloat16,enforce_eager=True,
            hf_config=N(jev_context_policy=dict(snapshot='/fixed/policy',policy_id='a'*64,
                contract={},base_id='b'*64,profiling_decisions=60))))


def test_explicit_model_registration_and_runtime_admission():
    register_context_model()
    assert validate_context_runtime(config())['profiling_decisions'] == 60


def test_fresh_worker_extension_import_registers_context_architecture():
    script = '''
from vllm import ModelRegistry
from types import SimpleNamespace
assert 'Qwen3ASRForContextDecisions' not in ModelRegistry.get_supported_archs()
from multimodal_jev.context_worker_extension import ContextBootstrapWorkerExtension
assert 'Qwen3ASRForContextDecisions' in ModelRegistry.get_supported_archs()
model, name = ModelRegistry.resolve_model_cls(['Qwen3ASRForContextDecisions'], SimpleNamespace(model_impl='vllm'))
assert model.__name__ == name == 'Qwen3ASRForContextDecisions'
assert callable(ContextBootstrapWorkerExtension.finish_jev_context_startup)
'''
    subprocess.run([sys.executable,'-c',script],check=True,timeout=120)


@pytest.mark.parametrize('fault',['tp','pp','chunk','cache','quant','dtype','eager','lora','spec','budget'])
def test_incompatible_runtime_rejected(fault):
    c = config()
    if fault == 'tp': c.parallel_config.tensor_parallel_size = 2
    if fault == 'pp': c.parallel_config.pipeline_parallel_size = 2
    if fault == 'chunk': c.scheduler_config.enable_chunked_prefill = True
    if fault == 'cache': c.cache_config.enable_prefix_caching = True
    if fault == 'quant': c.quant_config = object()
    if fault == 'dtype': c.model_config.dtype = torch.float32
    if fault == 'eager': c.model_config.enforce_eager = False
    if fault == 'lora': c.lora_config = None
    if fault == 'spec': c.model_config.hf_config.jev_context_policy['other'] = 1
    if fault == 'budget': c.model_config.hf_config.jev_context_policy['profiling_decisions'] = 0
    with pytest.raises(ValueError): validate_context_runtime(c)
