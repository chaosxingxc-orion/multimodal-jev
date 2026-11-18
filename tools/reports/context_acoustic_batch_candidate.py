"""Bounded multi-recording acoustic CNN and packed encoder candidate.

Keep complete 100-frame convolution windows, reset positions per chunk and
preserve per-recording attention windows. No weights or global profile change.
"""
from types import MethodType
import torch
from torch.nn import functional as F


def audio_layout(lengths, window, infer_window):
    if (type(window) is not int or window != 100 or type(infer_window) is not int
            or infer_window < window or infer_window % window
            or not lengths or any(type(n) is not int or n <= 0 for n in lengths)):
        raise ValueError('positive complete recordings and pinned 100-frame windows required')
    chunks=[]; outputs=[]; attention=[]
    for frames in lengths:
        full, tail=divmod(frames, window)
        parts=[window]*full+([tail] if tail else [])
        valid=[(n+7)//8 for n in parts]
        count=sum(valid); outputs.append(count); chunks.extend(parts)
        # Existing single-short profile attends to the entire short recording.
        width=((window+7)//8)*(infer_window//window)
        whole, remainder=divmod(count, width)
        attention.extend([width]*whole+([remainder] if remainder else []))
    return chunks, outputs, attention


def batched_audio_forward(tower, features, feature_lens, aftercnn_lens, *, conv_batch_chunks):
    if (features.ndim != 2 or min(features.shape)<1 or not features.is_floating_point()
            or not torch.isfinite(features).all() or feature_lens.ndim != 1
            or aftercnn_lens.shape != feature_lens.shape
            or feature_lens.dtype not in (torch.int32,torch.int64)
            or aftercnn_lens.dtype not in (torch.int32,torch.int64)
            or type(conv_batch_chunks) is not int or not 1 <= conv_batch_chunks <= 128):
        raise ValueError('finite packed audio and bounded convolution batch required')
    lengths=feature_lens.detach().cpu().tolist()
    chunks, outputs, attention=audio_layout(lengths,tower.n_window*2,tower.n_window_infer)
    if sum(lengths)!=features.shape[-1] or outputs!=aftercnn_lens.detach().cpu().tolist():
        raise ValueError('audio lengths and output boundaries differ')
    pieces=features.split(chunks,dim=-1); projected=[]
    # Bound both padded convolution input and intermediate activations.
    for offset in range(0,len(chunks),conv_batch_chunks):
        group=pieces[offset:offset+conv_batch_chunks]
        x=torch.stack([F.pad(p,(0,100-p.shape[-1])) for p in group])[:,None]
        for conv in (tower.conv2d1,tower.conv2d2,tower.conv2d3):
            x=F.gelu(conv(x))
        b,c,f,t=x.shape
        x=tower.conv_out(x.permute(0,3,1,2).contiguous().view(b,t,c*f))
        x=x+tower.positional_embedding.positional_embedding[:t].to(x.dtype)
        for i,part in enumerate(group):
            projected.append(x[i,:(part.shape[-1]+7)//8])
    hidden=torch.cat(projected,dim=0)
    bounds=torch.tensor([0,*attention],device=hidden.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
    maximum=tower.compute_attn_mask_seqlen(bounds)
    for layer in tower.layers:
        hidden=layer(hidden,bounds,maximum)
    hidden=tower.proj2(tower.act(tower.proj1(tower.ln_post(hidden))))
    if hidden.ndim!=2 or len(hidden)!=sum(outputs) or not torch.isfinite(hidden).all():
        raise ValueError('complete finite acoustic output required')
    return hidden


def install(model, *, conv_batch_chunks=32):
    tower=model.audio_tower
    if (not getattr(model,'_jev_segmented_candidate',False)
            or not getattr(tower,'_jev_short_padding',False)
            or getattr(tower,'_jev_acoustic_batch_candidate',False)
            or model.vllm_config.parallel_config.tensor_parallel_size!=1
            or tower.conv2d1.weight.dtype!=torch.bfloat16
            or type(conv_batch_chunks) is not int or not 1<=conv_batch_chunks<=128):
        raise ValueError('fresh aligned BF16 TP1 segmented tower required')
    audio_layout([1],tower.n_window*2,tower.n_window_infer)
    def forward(self,input_features,feature_lens,aftercnn_lens):
        return batched_audio_forward(self,input_features,feature_lens,aftercnn_lens,
                                     conv_batch_chunks=conv_batch_chunks)
    tower.forward=MethodType(forward,tower);tower._jev_acoustic_batch_candidate=True
    return dict(audio='batched_cnn_packed_encoder_recording_boundaries_v1',
                conv_batch_chunks=conv_batch_chunks,training_dispatch_allowed=False)
