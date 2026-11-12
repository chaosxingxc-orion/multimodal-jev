from pathlib import Path
import sys
import pytest
np=pytest.importorskip('numpy');sf=pytest.importorskip('soundfile')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_intent_development_audio import inspect_audio
from multimodal_jev.baseline import file_hash


def test_signal_validity_is_strict_but_not_semantic_admission(tmp_path):
    path=tmp_path/'clip.wav';sf.write(path,np.ones(1600,dtype=np.float32)*.2,16000)
    receipt=inspect_audio(path,file_hash(path));assert receipt['seconds']==.1
    with pytest.raises(ValueError,match='bytes differ'):inspect_audio(path,'changed')
    sf.write(path,np.zeros(1600,dtype=np.float32),16000)
    with pytest.raises(ValueError,match='all-zero'):inspect_audio(path,file_hash(path))
    sf.write(path,np.ones((1600,2),dtype=np.float32)*.2,16000)
    with pytest.raises(ValueError,match='mono'):inspect_audio(path,file_hash(path))
