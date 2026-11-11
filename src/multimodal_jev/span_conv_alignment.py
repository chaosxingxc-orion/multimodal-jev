"""Single-recording diagnostic matching HF chunk tensor strides for conv1."""


def hf_chunk_layout(x):
    if x.ndim!=4 or x.shape[1]!=1:
        raise ValueError('chunk x channel=1 x mel x frames required')
    chunks,_,mel,frames=x.shape
    contiguous_mel=x[:,0].permute(1,0,2).contiguous().view(1,mel,chunks*frames)
    return contiguous_mel.view(1,mel,chunks,frames).permute(0,2,1,3).reshape(chunks,1,mel,frames)


def install_hf_conv_layout(model):
    if getattr(model,'_jev_conv_layout_handle',None) is not None:
        raise ValueError('conv layout already installed')
    def change_layout(_module,args):
        return (hf_chunk_layout(args[0]),*args[1:])
    model._jev_conv_layout_handle=model.audio_tower.conv2d1.register_forward_pre_hook(change_layout)
    return {'scope':'Single recording complete prefill diagnostic only; tensor values unchanged',
            'operation':'HF contiguous mel -> chunk view/permute/reshape before conv2d1'}
