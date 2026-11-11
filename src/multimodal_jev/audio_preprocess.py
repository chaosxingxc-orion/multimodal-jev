"""Deterministic waveform transforms shared by actor and rollout.

No transcript, labels or model outputs are used. Originals are never modified.
Speed perturbation here resamples at a fixed output rate and changes pitch too;
it is not a pitch-preserving tempo algorithm. Silence trimming is edge-only.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
import numpy as np


def waveform_hash(audio):
    return hashlib.sha256(np.asarray(audio,dtype='<f4').tobytes()).hexdigest()


@dataclass(frozen=True)
class AudioPreprocess:
    preemphasis: float = 0.0
    trim_threshold_dbfs: float | None = None
    trim_frame_ms: float = 20.0
    trim_margin_ms: float = 100.0
    target_rms_dbfs: float | None = None
    max_gain_db: float = 12.0
    peak_limit: float = 0.99
    speed_factor: float = 1.0
    version: str = 'jev-waveform-v1'

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name=='version' or value is None:continue
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('finite numeric preprocessing option required: '+name)
        if self.version!='jev-waveform-v1':raise ValueError('unknown preprocessing version')
        if not 0<=self.preemphasis<1:raise ValueError('preemphasis must be in [0,1)')
        if self.trim_threshold_dbfs is not None and not -120<=self.trim_threshold_dbfs<0:
            raise ValueError('invalid edge silence threshold')
        if not 1<=self.trim_frame_ms<=100 or not 0<=self.trim_margin_ms<=1000:
            raise ValueError('invalid trim timing')
        if self.target_rms_dbfs is not None and not -80<=self.target_rms_dbfs<0:
            raise ValueError('invalid target RMS')
        if not 0<=self.max_gain_db<=24 or not 0<self.peak_limit<=1 or not .5<=self.speed_factor<=2:
            raise ValueError('invalid gain, peak or speed bound')


def preprocess_audio(audio, config=None, sample_rate=16000):
    """Order: edge trim -> speed/pitch -> preemphasis -> bounded gain.

    Returns float32 waveform and an audit receipt. Identity config is byte-exact.
    All-silent input is retained; internal pauses are never removed. Gain uses
    whole-waveform RMS (not a loudness or speech-active RMS estimate).
    """
    config = config if isinstance(config,AudioPreprocess) else AudioPreprocess(**(config or {}))
    original=np.asarray(audio)
    if sample_rate!=16000 or original.ndim!=1 or not original.size or not np.isfinite(original).all():
        raise ValueError('finite nonempty mono 16 kHz waveform required')
    x=np.asarray(original,dtype=np.float32).copy()
    start,end=0,len(x)
    if config.trim_threshold_dbfs is not None:
        frame=max(1,round(sample_rate*config.trim_frame_ms/1000))
        active=[]
        threshold=10**(config.trim_threshold_dbfs/20)
        for left in range(0,len(x),frame):
            block=x[left:left+frame].astype(np.float64)
            if np.sqrt(np.mean(block*block))>=threshold:active.append(left)
        if active:
            margin=round(sample_rate*config.trim_margin_ms/1000)
            start=max(0,active[0]-margin)
            end=min(len(x),active[-1]+frame+margin)
            x=x[start:end].copy()
    if config.speed_factor!=1:
        from scipy.signal import resample_poly
        ratio=Fraction(1/config.speed_factor).limit_denominator(1000)
        x=resample_poly(x,ratio.numerator,ratio.denominator).astype(np.float32)
    if config.preemphasis:
        previous=x.copy()
        x[1:]=previous[1:]-config.preemphasis*previous[:-1]
    gain=1.
    if config.target_rms_dbfs is not None:
        rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
        if rms>0:
            gain=min(10**(config.target_rms_dbfs/20)/rms,10**(config.max_gain_db/20))
    # Avoid clipping after enabled amplitude-changing transforms; identity leaves
    # even original peaks untouched and never silently changes the baseline.
    if config.target_rms_dbfs is not None or config.preemphasis or config.speed_factor!=1:
        peak=float(np.max(np.abs(x)))
        if peak>0:gain=min(gain,config.peak_limit/peak)
        x=(x*gain).astype(np.float32)
    if not len(x) or not np.isfinite(x).all():raise ValueError('invalid processed waveform')
    receipt=dict(config=asdict(config),sample_rate=sample_rate,
                 input_samples=len(original),output_samples=len(x),
                 input_waveform_sha256=waveform_hash(original),output_waveform_sha256=waveform_hash(x),
                 retained_original_interval=[start,end],applied_gain=gain,
                 speed_changes_pitch=config.speed_factor!=1,
                 internal_silence_removed=False)
    return np.ascontiguousarray(x),receipt
