"""Compare actual vLLM processor output with one prepared audio-direct input.

The caller authenticates processor versions, raw waveforms and rendering. This
checks the BF16 model input boundary, not FP32 features or model probabilities.
"""


def compare_prepared(item, processed):
    import torch

    if item['waveform_boundary'] != 'vllm_hop':
        raise ValueError('registered hop-aligned input required')
    mask = item['feature_attention_mask']
    features = item['input_features']
    data = processed['mm_kwargs'].get_data()
    actual = data['input_audio_features']
    lengths = data['audio_feature_lengths']
    if (features.ndim != 3 or mask.ndim != 2
            or mask.shape != (features.shape[0], features.shape[-1])
            or not torch.all((mask == 0) | (mask == 1))
            or not torch.equal(mask, mask.cumprod(-1))
            or not torch.all(mask.sum(-1) > 0)
            or not torch.isfinite(features).all()
            or actual.dtype != torch.bfloat16 or actual.ndim != 2
            or not torch.isfinite(actual).all()
            or lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64)):
        raise ValueError('finite BF16 vLLM features and complete ordered frame masks required')
    if item['input_ids'].tolist() != processed['prompt_token_ids']:
        raise ValueError('actual vLLM prompt tokens differ')
    if mask.sum(-1).tolist() != lengths.tolist():
        raise ValueError('actual vLLM ordered frame lengths differ')
    reference = features.permute(0, 2, 1)[mask.bool()].permute(1, 0).to(torch.bfloat16)
    if actual.shape != reference.shape:
        raise ValueError('actual vLLM feature shape differs')
    # Same boundary and tolerance as the prior complete TRAIN78 processor audit.
    delta = (actual.float() - reference.float()).abs()
    if not torch.allclose(actual.float(), reference.float(), atol=1e-5, rtol=1e-5):
        raise ValueError('actual vLLM ordered audio features differ')
    return dict(token_ids_equal=True, ordered_frames=lengths.tolist(),
                feature_shape=list(actual.shape), model_dtype=str(actual.dtype),
                max_feature_error=float(delta.max()), feature_atol=1e-5, feature_rtol=1e-5)
