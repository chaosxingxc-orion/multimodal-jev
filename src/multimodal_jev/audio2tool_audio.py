"""Authenticate and decode the same audio bytes for the frozen native actor."""
import hashlib
from io import BytesIO
from pathlib import Path

from .audio2tool_input import messages, validate
from .omni_waveform_boundary import align_to_hop
from .sfc_audio import load_sfc_audio


def load_audio(inputs, root):
    """Full ordered prefix, no truncation, source replacement or transcript reads.

    Hashing and decoding one byte buffer avoids a hash-then-reopen race. Source
    format is fixed by the full release header audit (24 kHz mono PCM16 WAV).
    """
    import soundfile as sf
    validate(inputs)
    root = Path(root).resolve(strict=True)
    waves, receipts = [], []
    for audio in inputs['audios']:
        path = (root / audio['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('audio escapes dataset root')
        payload = path.read_bytes()
        if (len(payload) != audio['bytes']
                or hashlib.sha256(payload).hexdigest() != audio['sha256']):
            raise ValueError('source audio bytes differ')
        info = sf.info(BytesIO(payload))
        if (info.format != 'WAV' or info.subtype != 'PCM_16'
                or info.samplerate != 24000 or info.channels != 1 or info.frames <= 0):
            raise ValueError('audited source audio format differs')
        wave, conversion = load_sfc_audio(BytesIO(payload))
        expected = (info.frames * 16000 + 23999) // 24000
        if len(wave) != expected or conversion['source_samples'] != info.frames:
            raise ValueError('complete resampled audio required')
        aligned = align_to_hop([wave], 160)[0]
        receipts.append(dict(path=audio['path'], sha256=audio['sha256'], bytes=len(payload),
            conversion=conversion, right_zero_padding_samples=len(aligned)-len(wave),
            actor_samples=len(aligned), expected_frames=len(aligned)//160,
            actor_waveform_sha256=hashlib.sha256(aligned.astype('<f4', copy=False).tobytes()).hexdigest()))
        waves.append(aligned)
    return waves, receipts


def prepare_request(processor, inputs, root, registry_bytes):
    """vLLM request and audio provenance; engine/feature acceptance is separate."""
    # Validate the independent registry before reading any waveform.
    from .audio2tool_input import system_text
    system_text(registry_bytes)
    if processor.feature_extractor.hop_length != 160:
        raise ValueError('registered processor hop required')
    waves, receipts = load_audio(inputs, root)
    prompt = processor.apply_chat_template(messages(inputs, waves, registry_bytes),
        tokenize=False, add_generation_prompt=True)
    if prompt.count('<|audio_pad|>') != len(waves):
        raise ValueError('complete ordered audio placeholders required')
    return dict(prompt=prompt, multi_modal_data={'audio': [(w, 16000) for w in waves]}), receipts
