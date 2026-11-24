"""Experimental true-batch inputs for the native Qwen3-ASR Span model.

Operates on existing, individually authenticated processor outputs. It does not
resample audio, read labels or replace request identities. Not a runtime release
or a numerical-parity admission. The production serial actor is unchanged.
"""
import torch
from torch.nn import functional as F


def _cnn_length(length):
    for _ in range(3):
        length = (length + 1) // 2
    return length


def _windows(frames, chunk_size, infer_window, largest_chunk):
    full, tail = divmod(frames, chunk_size)
    total = full * _cnn_length(chunk_size) + _cnn_length(tail)
    width = _cnn_length(largest_chunk) * (infer_window // chunk_size)
    full, tail = divmod(total, width)
    return [width] * full + ([tail] if tail else [])


def collate_span_audio(items, *, pad_token_id, audio_token_id,
                       audio_chunk_size, audio_infer_window):
    """Right-pad one-audio requests sharing the same ordered Span candidates.

    Original feature padding is retained exactly: its values may participate in
    convolution near valid boundaries. Added whole chunks are zero and masked.
    Token positions are unchanged by right padding; explicit position IDs avoid
    batch-dependent rotary offsets. Reject changed per-audio attention windows.
    """
    return _collate_span_audio(items, pad_token_id=pad_token_id, audio_token_id=audio_token_id,
                               audio_chunk_size=audio_chunk_size, audio_infer_window=audio_infer_window,
                               text_only=False)


def _collate_span_audio(items, *, pad_token_id, audio_token_id, audio_chunk_size,
                        audio_infer_window, text_only):
    # text_only is for the card scorer which feeds each owner's original features
    # directly to the tower. Keep every per-request text, span and window check.
    frame_cache = {}
    if (not items or any(type(x) is not int or x < 0 for x in
                        (pad_token_id, audio_token_id)) or pad_token_id == audio_token_id
            or type(audio_chunk_size) is not int or audio_chunk_size != 100
            or type(audio_infer_window) is not int or audio_infer_window < audio_chunk_size
            or audio_infer_window % audio_chunk_size):
        # Pinned upstream output-length helper uses 13 outputs per 100-frame chunk.
        raise ValueError('pinned audio chunk100, integral window and distinct token IDs required')
    candidates = items[0]['candidates']
    if not candidates or any(len(c) < 2 or len(set(c)) != len(c)
                             or any(type(x) is not int or x < 0 for x in c) for c in candidates):
        raise ValueError('distinct candidate tokens required')
    required = {'input_ids', 'attention_mask', 'input_features', 'input_features_mask'}
    frame_lengths = []
    for item in items:
        inputs, positions = item['inputs'], item['positions']
        if (set(inputs) - required - {'position_ids'} or not required <= inputs.keys()
                or item['candidates'] != candidates):
            raise ValueError('known inputs and identical ordered candidates required')
        if any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu'
               for v in [*inputs.values(), positions]):
            raise ValueError('CPU processor tensors required')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        features, fmask = inputs['input_features'], inputs['input_features_mask']
        if (ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1 or not ids.numel()
                or (ids < 0).any() or mask.shape != ids.shape or not (mask == 1).all()
                or positions.dtype != torch.long or positions.shape != (1, len(candidates))
                or (positions < 0).any() or (positions >= ids.shape[1]).any()
                or (positions[:, 1:] <= positions[:, :-1]).any()):
            raise ValueError('one unpadded sequence and ordered valid spans required')
        if 'position_ids' in inputs and not torch.equal(inputs['position_ids'], torch.arange(ids.shape[1])[None]):
            raise ValueError('canonical original position IDs required')
        owner = (id(features), id(fmask))
        if text_only and owner in frame_cache:
            frames = frame_cache[owner]
        else:
            if (features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1
                    or features.shape[-1] == 0 or features.shape[-1] % audio_chunk_size
                    or not features.is_floating_point() or not torch.isfinite(features).all()
                    or fmask.shape != (1, features.shape[-1])
                    or not ((fmask == 0) | (fmask == 1)).all()
                    or not torch.equal(fmask, fmask.cumprod(-1)) or fmask.sum() < 1):
                raise ValueError('finite chunk-padded audio and contiguous prefix mask required')
            frames = int(fmask.sum())
            if text_only:
                frame_cache[owner] = frames
        full, tail = divmod(frames, audio_chunk_size)
        expected = full * _cnn_length(audio_chunk_size) + _cnn_length(tail)
        audio_positions = (ids[0] == audio_token_id).nonzero().flatten()
        if audio_positions.numel() != expected or audio_positions[-1] >= positions[0, 0]:
            raise ValueError('per-request audio placeholders must match features and precede spans')
        frame_lengths.append(frames)
    signature = lambda x: (x['inputs']['input_features'].shape[1],
                           tuple(x['inputs'][k].dtype for k in sorted(required)))
    if any(signature(item) != signature(items[0]) for item in items):
        raise ValueError('same mel dimensions and processor dtypes required')
    largest = min(max(frame_lengths), audio_chunk_size)
    for frames in frame_lengths:
        single = _windows(frames, audio_chunk_size, audio_infer_window, min(frames, audio_chunk_size))
        batched = _windows(frames, audio_chunk_size, audio_infer_window, largest)
        if single != batched:
            raise ValueError('batch changes per-audio attention windows')
    token_width = max(item['inputs']['input_ids'].shape[-1] for item in items)
    feature_width = max(item['inputs']['input_features'].shape[-1] for item in items)
    packed = {}
    for key in sorted(required if not text_only else ('input_ids', 'attention_mask')):
        width = feature_width if key.startswith('input_features') else token_width
        packed[key] = torch.cat([F.pad(item['inputs'][key],
                                      (0, width - item['inputs'][key].shape[-1]),
                                      value=pad_token_id if key == 'input_ids' else 0)
                                 for item in items], dim=0)
    packed['position_ids'] = (packed['attention_mask'].long().cumsum(-1) - 1).clamp_min(0)
    packed['position_ids'].masked_fill_(packed['attention_mask'] == 0, 0)
    return packed, torch.cat([item['positions'] for item in items]), [list(c) for c in candidates]


def forward_span_audio_batch(module, items, *, device, dtype, **collation):
    """One model call, split back into the original request order with autograd."""
    inputs, positions, candidates = collate_span_audio(items, **collation)
    inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}
    logprobs = module(inputs, positions.to(device), candidates)[1]
    return [tuple(p[i:i+1] for p in logprobs) for i in range(len(items))]
