"""Align short-audio CNN boundaries, retaining native vLLM weights."""


def short_audio_forward(tower,input_features,feature_lens,aftercnn_lens):
    import torch
    import torch.nn.functional as F
    if feature_lens.numel()!=1:raise ValueError('one complete recording required')
    frames=int(feature_lens.item());window=tower.n_window*2
    if not 0<frames<window or input_features.shape[-1]!=frames:
        raise ValueError('short packed recording required')
    # HF pads to a complete window before all three convolutions. Truncating
    # after each convolution instead changes later boundary activations.
    padded=F.pad(input_features,(0,window-frames))[None,None]
    hidden=F.gelu(tower.conv2d1(padded))
    hidden=F.gelu(tower.conv2d2(hidden))
    hidden=F.gelu(tower.conv2d3(hidden))
    batch,channels,freq,steps=hidden.shape
    hidden=tower.conv_out(hidden.permute(0,3,1,2).contiguous().view(batch,steps,channels*freq))
    hidden=hidden+tower.positional_embedding.positional_embedding[:steps].to(hidden.dtype)
    valid=int(tower._get_cnn_output_lengths(feature_lens).item())
    if valid!=int(aftercnn_lens.item()):raise ValueError('short audio output length differs')
    hidden=hidden[0,:valid]
    boundaries=torch.tensor([0,valid],device=hidden.device,dtype=torch.int32)
    maximum=tower.compute_attn_mask_seqlen(boundaries)
    for layer in tower.layers:hidden=layer(hidden,boundaries,maximum)
    hidden=tower.ln_post(hidden)
    return tower.proj2(tower.act(tower.proj1(hidden)))


def install_short_audio_padding(model):
    from types import MethodType
    if model.vllm_config.parallel_config.tensor_parallel_size!=1:
        raise ValueError('short audio alignment requires TP1')
    tower=model.audio_tower
    if getattr(tower,'_jev_short_padding',False):raise ValueError('short audio alignment already installed')
    original=tower.forward
    def forward(self,input_features,feature_lens,aftercnn_lens):
        if feature_lens.numel()!=1:raise ValueError('single-recording alignment profile required')
        if int(feature_lens.item())<self.n_window*2:
            return short_audio_forward(self,input_features,feature_lens,aftercnn_lens)
        return original(input_features,feature_lens,aftercnn_lens)
    tower.forward=MethodType(forward,tower);tower._jev_short_padding=True
    return dict(window_frames=tower.n_window*2,scope='TP1 one-recording short audio: full HF CNN window, native vLLM weights; longer audio unchanged')
