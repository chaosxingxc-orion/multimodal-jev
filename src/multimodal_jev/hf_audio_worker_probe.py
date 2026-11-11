"""Diagnostic-only HF acoustic tower inside an offline vLLM worker.

Uses current checkpoint weights and live mel features. No saved activations,
transcripts, external model or modifications to sampled probabilities.
"""
import torch


def unpack_audio_features(features, lengths, chunk_size):
    """Reconstruct the HF contiguous mask and zero-padded chunk layout."""
    if features.ndim != 2 or lengths.ndim != 1 or not len(lengths):
        raise ValueError('packed features and nonempty lengths required')
    sizes = lengths.tolist()
    if any(type(n) is not int or n <= 0 for n in sizes) or sum(sizes) != features.shape[-1]:
        raise ValueError('invalid packed feature lengths')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('positive chunk size required')
    padded = ((max(sizes)+chunk_size-1)//chunk_size)*chunk_size
    batch = features.new_zeros((len(sizes), features.shape[0], padded))
    mask = torch.zeros((len(sizes), padded), device=features.device, dtype=torch.long)
    offset = 0
    for index, length in enumerate(sizes):
        batch[index, :, :length] = features[:, offset:offset+length]
        mask[index, :length] = 1
        offset += length
    return batch, mask


class HFAudioTowerProbe(torch.nn.Module):
    def __init__(self, tower, projector):
        super().__init__()
        self.tower = tower
        self.projector = projector

    @property
    def dtype(self):
        return next(self.tower.parameters()).dtype

    def forward(self, input_features, feature_lens, aftercnn_lens):
        features, mask = unpack_audio_features(input_features, feature_lens, self.tower.n_window*2)
        result = self.tower(input_features=features, input_features_mask=mask, return_dict=True)
        projected = self.projector(result.last_hidden_state)
        if projected.shape[0] != int(aftercnn_lens.sum()):
            raise ValueError('HF/vLLM audio token lengths differ')
        return projected


def install_hf_audio_probe(model, checkpoint):
    from .asr import load_runtime
    from .policy_identity import checkpoint_identity
    if model.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError('diagnostic supports TP1 only')
    if getattr(model, '_jev_hf_audio_probe', False):
        raise ValueError('probe already installed')
    device = next(model.audio_tower.parameters()).device
    dtype = next(model.audio_tower.parameters()).dtype
    if dtype != torch.bfloat16:
        raise ValueError('current diagnostic requires BF16')
    previous_dtype = torch.get_default_dtype()
    try:
        # vLLM worker creation defaults must not change HF nonpersistent buffers.
        torch.set_default_dtype(torch.float32)
        with torch.device('cpu'):
            _, hf = load_runtime(checkpoint, 'cpu', 'bfloat16')
        replacement = HFAudioTowerProbe(hf.model.audio_tower, hf.model.multi_modal_projector)
        replacement = replacement.to(device=device).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    model.audio_tower = replacement
    model._jev_hf_audio_probe = True
    policy, files = checkpoint_identity(checkpoint)
    return {'backend':'hf_eager_live_audio_diagnostic', 'policy_id':policy,
            'checkpoint_files':files, 'device':str(device), 'dtype':str(dtype),
            'position_buffer_dtype':str(replacement.tower.positional_embedding.positional_embedding.dtype),
            'scope':'TP1 read-only probe; weight reload/update not supported'}
