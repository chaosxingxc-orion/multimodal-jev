"""Narrow workaround for vLLM 0.30 Qwen3-ASR precomputed feature input.

Upstream accepts DictEmbeddingItems but calls its waveform HF processor with
an audio placeholder and no waveform, raising StopIteration before passthrough.
Only explicit feature requests bypass that invalid call; waveform is unchanged.
"""


def enable_feature_passthrough():
    from transformers.feature_extraction_utils import BatchFeature
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRMultiModalProcessor
    cls=Qwen3ASRMultiModalProcessor
    if getattr(cls,'_jev_feature_passthrough',False):return
    original=cls._apply_hf_processor_main

    def apply(self,mm_items,hf_processor_mm_kwargs):
        valid=mm_items.select({k for k,c in mm_items.get_all_counts().items() if c>0})
        processor_data,passthrough=self._get_hf_mm_data(valid)
        if not processor_data and 'input_audio_features' in passthrough:
            if not {'input_audio_features','audio_feature_lengths'}<=passthrough.keys():
                raise ValueError('incomplete audio feature passthrough')
            return BatchFeature(dict(passthrough))
        return original(self,mm_items,hf_processor_mm_kwargs)

    cls._apply_hf_processor_main=apply
    cls._jev_feature_passthrough=True
