from pathlib import Path
import sys
import wave
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from audit_audio2tool_headers import header


def test_pcm_header_duration_and_size_binding(tmp_path):
    path=tmp_path/'a.wav'
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(8000)
        stream.writeframes(b'\0'*(8000*2*2))
    audio=dict(path='a.wav',bytes=path.stat().st_size,sha256='a'*64)
    result=header(tmp_path,audio)
    assert result['seconds']==1. and result['channels']==2 and result['sample_rate']==8000
    audio['bytes']+=1
    with pytest.raises(ValueError,match='byte length'):header(tmp_path,audio)


def test_escape_rejected(tmp_path):
    with pytest.raises(ValueError,match='relative'):
        header(tmp_path,dict(path='../other.wav'))
