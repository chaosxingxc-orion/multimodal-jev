"""Keep existing per-recording acoustics for a future batched language prefill.

This is intentionally serial in the acoustic tower. It accepts the packed
feature layout [mel, sum(frames)], never a padded batch or merged convolution.
Caller supplies the already aligned single-recording tower forward. No install,
weight/policy mutation or inference admission is performed here.
"""
import torch


def segmented_audio_forward(single_forward, input_features, feature_lens, aftercnn_lens):
    if (input_features.ndim != 2 or min(input_features.shape) < 1
            or not input_features.is_floating_point() or not torch.isfinite(input_features).all()
            or feature_lens.ndim != 1 or aftercnn_lens.shape != feature_lens.shape
            or not feature_lens.numel()
            or feature_lens.dtype not in (torch.int32, torch.int64)
            or aftercnn_lens.dtype not in (torch.int32, torch.int64)):
        raise ValueError('finite packed features and aligned integer recording lengths required')
    lengths = feature_lens.detach().cpu().tolist()
    outputs = aftercnn_lens.detach().cpu().tolist()
    if (any(n <= 0 for n in lengths + outputs) or sum(lengths) != input_features.shape[1]):
        raise ValueError('complete positive recording lengths must cover packed features')
    result = []
    for index, features in enumerate(input_features.split(lengths, dim=1)):
        value = single_forward(features.contiguous(), feature_lens[index:index+1],
                               aftercnn_lens[index:index+1])
        if (value.ndim != 2 or len(value) != outputs[index] or value.shape[1] < 1
                or value.dtype != input_features.dtype or value.device != input_features.device
                or not torch.isfinite(value).all()
                or (result and value.shape[1] != result[0].shape[1])):
            raise ValueError('single-recording output does not match audio token boundaries')
        result.append(value)
    return torch.cat(result, dim=0)
