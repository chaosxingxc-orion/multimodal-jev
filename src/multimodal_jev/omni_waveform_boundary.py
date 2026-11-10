"""Explicit vLLM-compatible zero padding, after mono 16 kHz resampling."""


def align_to_hop(waves, hop_length):
    import numpy as np
    if type(hop_length) is not int or hop_length<=0 or not waves:
        raise ValueError('positive hop length and nonempty waveform list required')
    result=[]
    for wave in waves:
        if (not isinstance(wave,np.ndarray) or wave.ndim!=1 or wave.dtype!=np.float32
                or not len(wave) or not np.isfinite(wave).all()):
            raise ValueError('finite nonempty mono float32 waveforms required')
        result.append(np.pad(wave,(0,(-len(wave))%hop_length)))
    return result
