"""Spawn-worker bootstrap for the explicitly selected context architecture.

vLLM imports worker extensions before init_device, which resolves the model
architecture. Parent-only dynamic registration is not inherited by spawn.
"""
from .vllm_context_model import ContextPolicyWorkerExtension, register_context_model
from .vllm_audio_features import enable_feature_passthrough
from .span_batch_rollout_candidate import BatchCandidateWorkerExtension

register_context_model()
enable_feature_passthrough()


class ContextBootstrapWorkerExtension(ContextPolicyWorkerExtension, BatchCandidateWorkerExtension):
    pass
