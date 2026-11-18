import hashlib
import io

import numpy as np
import pytest
import soundfile as sf

from tools.reports.materialize_minds_audio import validate_record


def example():
    stream = io.BytesIO()
    sf.write(stream, np.array([0, .25, -.5], dtype='float32'), 8000, format='WAV', subtype='PCM_16')
    payload = stream.getvalue()
    path = 'en-US~FREEZE/label_bearing.wav'
    waveform, _ = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
    row = dict(path=path, audio=dict(bytes=payload, path='SECRET_GOLD.wav'),
               transcription='SECRET_TRANSCRIPT', intent_class=9)
    expected = dict(id='minds14:' + hashlib.sha256(('en-US\0' + path).encode()).hexdigest(),
                    language='en-US', source_path_sha256=hashlib.sha256(path.encode()).hexdigest(),
                    encoded_audio_sha256=hashlib.sha256(payload).hexdigest(), sample_rate=8000,
                    channels=1, samples=3,
                    waveform_sha256=hashlib.sha256(b'8000:1:' + waveform.astype('<f4').tobytes()).hexdigest())
    return row, expected


def test_consumer_inputs_exclude_gold_and_transcript_and_preserve_bytes():
    row, expected = example()
    payload, value = validate_record(row, 'en-US', expected)
    assert payload == row['audio']['bytes']
    assert set(value) == {'id', 'audio', 'audio_sha256', 'sample_rate', 'channels', 'samples', 'waveform_sha256'}
    assert value['audio'] == 'audio/' + expected['id'].split(':')[1] + '.wav'
    assert not any(x in str(value) for x in ['FREEZE', 'SECRET', 'en-US'])


@pytest.mark.parametrize('field', ['id', 'language', 'source_path_sha256', 'encoded_audio_sha256',
                                  'waveform_sha256', 'sample_rate', 'samples', 'channels'])
def test_audit_identity_mismatch_is_rejected(field):
    row, expected = example()
    expected[field] = 'wrong'
    with pytest.raises(ValueError):
        validate_record(row, 'en-US', expected)
