"""vLLM 0.30 pooling adapter: native audio, causal Span positions, no decode.

Independent adapter against Apache-2.0 vLLM interfaces. TP=1, no prefix cache
or chunked prefill admitted initially. No upstream source modifications.
"""
import torch
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration, Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo, Qwen3ASRDummyInputsBuilder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class NativeSpanPooler(Pooler):
    def __init__(self, compute_logits):
        super().__init__()
        # Bound method, not another registered reference to the parent module.
        self._compute_logits = compute_logits

    def get_supported_tasks(self):
        return {'token_classify'}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill() or not all(cursor.get_finished_mask()):
            raise ValueError('complete uncached prefill required for Span readout')
        chunks = hidden_states.split(cursor.num_scheduled_tokens_cpu.tolist())
        ids = pooling_metadata.get_prompt_token_ids_cpu()
        outputs = []
        for chunk, prompt, params in zip(chunks, ids, pooling_metadata.pooling_params, strict=True):
            extra = params.extra_kwargs or {}
            # vLLM memory profiling supplies synthetic all-zero token IDs and
            # no request metadata. Exercise the native head, never fake logits.
            if not extra and bool((prompt == 0).all()):
                outputs.append(self._compute_logits(chunk[:3])[:, :2].float())
                continue
            if set(extra) != {'span_positions', 'candidate_ids', 'expected_input_ids'}:
                raise ValueError('explicit Span positions, candidates and prompt identity required')
            if prompt.tolist() != extra['expected_input_ids'] or chunk.shape[0] != len(prompt):
                raise ValueError('vLLM/HF expanded prompt identity mismatch')
            positions, candidates = extra['span_positions'], extra['candidate_ids']
            if (not positions or any(type(p) is not int or p < 0 or p >= len(prompt) for p in positions)
                    or positions != sorted(set(positions)) or len(candidates) != len(positions)):
                raise ValueError('ordered positions and aligned candidates required')
            logits = self._compute_logits(chunk[torch.tensor(positions,device=chunk.device)])
            if logits is None or not torch.isfinite(logits).all():
                raise ValueError('finite native vocabulary logits required')
            width = max(map(len,candidates))
            output = torch.full((len(positions),width), -torch.inf, device=logits.device, dtype=torch.float32)
            for j, options in enumerate(candidates):
                if (len(options)<2 or len(set(options)) != len(options) or
                        any(type(t) is not int or not 0 <= t < logits.shape[-1] for t in options)):
                    raise ValueError('distinct finite candidate token IDs required')
                output[j,:len(options)] = logits[j,options].float()
            outputs.append(output)
        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor, info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForSpanDecisions(Qwen3ASRForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=''):
        if (vllm_config.parallel_config.tensor_parallel_size != 1 or
                vllm_config.scheduler_config.enable_chunked_prefill or
                vllm_config.cache_config.enable_prefix_caching):
            raise ValueError('Span v1 requires TP1, unchunked and uncached prefill')
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        self.pooler = NativeSpanPooler(self.compute_logits)


def register_span_model():
    from vllm import ModelRegistry
    ModelRegistry.register_model('Qwen3ASRForSpanDecisions',
                                 'multimodal_jev.vllm_spans:Qwen3ASRForSpanDecisions')
    ModelRegistry.register_model('Qwen3ASRForSpanDecisionsTP4Diagnostic',
                                 'multimodal_jev.vllm_spans:Qwen3ASRForSpanDecisionsTP4Diagnostic')


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor, info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForSpanDecisionsTP4Diagnostic(Qwen3ASRForConditionalGeneration):
    """Native distributed measurement only; not the accepted RL profile."""
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=''):
        if (vllm_config.parallel_config.tensor_parallel_size != 4 or
                vllm_config.parallel_config.pipeline_parallel_size != 1 or
                vllm_config.scheduler_config.enable_chunked_prefill or
                vllm_config.cache_config.enable_prefix_caching):
            raise ValueError('TP4 diagnostic requires PP1, unchunked and uncached prefill')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Pooling runs on all workers. Native gather may return None outside
        # rank zero; all-gather supplies actual vocabulary logits everywhere.
        self.language_model.logits_processor.use_all_gather = True
        self.pooler = NativeSpanPooler(self.compute_logits)
