"""Fixed-policy Omni Thinker with cached audio features and typed Jev logits."""
import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder, Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalProcessor, Qwen2_5OmniThinkerProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .context_policy_artifacts import read_execution_bundle
from .vllm_context_model import ContextPolicyModel, ContextPolicyWorkerExtension


class JevOmniFeatureProcessor(Qwen2_5OmniThinkerMultiModalProcessor):
    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs):
        items = mm_items.select({key for key, count in mm_items.get_all_counts().items() if count})
        processor_data, features = self._get_hf_mm_data(items)
        if not processor_data and 'input_audio_features' in features:
            if set(features) != {'input_audio_features', 'audio_feature_lengths', 'feature_attention_mask'}:
                raise ValueError('complete cached Omni audio features required')
            return BatchFeature(dict(features))
        return super()._apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    JevOmniFeatureProcessor, info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen25OmniForJev(ContextPolicyModel, Qwen2_5OmniThinkerForConditionalGeneration):
    jev_return_logits = True


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model('Qwen25OmniForJev', 'multimodal_jev.vllm_omni25_jev:Qwen25OmniForJev')


class OmniJevWorkerExtension(ContextPolicyWorkerExtension):
    def audit_jev_omni_adapter(self, adapter_id, policy_id):
        model = self.get_model()
        spec = model._jev_policy_spec
        if model._jev_policy_id != policy_id or model.pooler.memory_profiling:
            raise ValueError('Omni worker must serve the authenticated frozen policy')
        bundle = read_execution_bundle(spec['snapshot'], expected_policy_id=policy_id,
                                       expected_contract=spec['contract'], expected_base_id=spec['base_id'])
        manager = self.model_runner.lora_manager._adapter_manager
        if manager.lora_index_to_id.count(adapter_id) != 1:
            raise ValueError('the expected adapter is missing from the worker LoRA slots')
        slot = manager.lora_index_to_id.index(adapter_id)
        for name, rule in spec['contract']['lora'].items():
            packed = name.replace('thinker.model.', 'language_model.model.', 1).rsplit('.', 1)[0] + '.qkv_proj'
            layer = manager.modules[packed]
            index = 0 if name.endswith('.q_proj') else 2
            for matrix, storage in (('A', layer.lora_a_stacked), ('B', layer.lora_b_stacked)):
                expected = bundle['tensors'][f'backbone.{name}.lora_{matrix}.default.weight']
                if matrix == 'B':
                    expected = expected * rule['scaling']
                rows, columns = expected.shape
                actual = storage[index][slot, 0, :rows, :columns]
                if actual.dtype != torch.bfloat16 or not torch.equal(actual.cpu(), expected):
                    raise ValueError('active vLLM LoRA weight differs: ' + name + ':' + matrix)
        return dict(policy_id=policy_id, targets=len(spec['contract']['lora']), active_slot=slot)


# A vLLM worker imports the extension in a fresh interpreter.
register()
