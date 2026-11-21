"""Own a single authenticated Omni policy in the shared fixed vLLM runtime."""
from pathlib import Path

from .context_policy_artifacts import export_native_adapter, verify_native_adapter
from .context_resident_rollout import FixedContextRollout
from .omni25_serving_adapter import packed_targets, thinker_payload
from .qwen25_omni_provider import PROFILE


class FixedOmniRollout(FixedContextRollout):
    architecture = 'Qwen25OmniForJev'
    worker_extension = 'multimodal_jev.vllm_omni25_jev.OmniJevWorkerExtension'
    profiles = (PROFILE,)
    return_logits = True
    modality_limits = {'audio': 1, 'image': 0, 'video': 0}
    packed_targets = staticmethod(packed_targets)

    def prepare_adapter(self, config, bundle):
        path = Path(config['adapter'])
        if not path.exists():
            export_native_adapter(path, bundle, payload=thinker_payload)
        verify_native_adapter(path, bundle, payload=thinker_payload)
        return path

    def register_model(self):
        from . import vllm_omni25_jev  # noqa: F401

    def runtime_receipt(self, profile):
        return dict(profile=profile, engine='vLLM native BF16', policy_id=self.bundle['policy_id'],
                    training_dispatch_allowed=False)

    def _score(self, *args):
        result = super()._score(*args)
        if 'adapter' not in self.runtime:
            evidence, = self.engine.llm_engine.collective_rpc(
                'audit_jev_omni_adapter', args=(self.lora.lora_int_id, self.bundle['policy_id']))
            self.runtime['adapter'] = evidence
        return result
