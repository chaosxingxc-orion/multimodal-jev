import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip('soundfile')
pytest.importorskip('scipy')
from multimodal_jev.baseline import atomic_json, file_hash


def test_persisted_audio_preserves_duration_tone_source_and_missing_record(tmp_path):
    script = Path(__file__).resolve().parents[1]/'tools/jev/normalize_mac_slu_audio.py'
    spec = importlib.util.spec_from_file_location('normalize_mac_slu_audio', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path/'source'; source.mkdir(); output = tmp_path/'normalized'; output.mkdir()
    raw = source/'id_1.wav'
    t = np.arange(2400)/24000
    sf.write(raw, np.sin(2*np.pi*1000*t)*0.5, 24000, subtype='FLOAT')
    source_digest = file_hash(raw)
    rows = [dict(id='1', audio=dict(path=str(raw),sha256=source_digest)), dict(id='missing',audio=None)]
    (source/'train.inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (source/'train.targets.jsonl').write_text('{"not_for_model":true}\n')
    (source/'bank.json').write_text('{}')
    files={name:file_hash(source/name) for name in ('train.inputs.jsonl','train.targets.jsonl','bank.json')}
    atomic_json(source/'MANIFEST.json',dict(files=files,training_admitted=False))
    result=module.run(source,output)
    converted=[json.loads(line) for line in (output/'train.inputs.jsonl').open()]
    wave,rate=sf.read(converted[0]['audio']['path'],dtype='float32')
    assert rate==16000 and len(wave)==1600
    assert np.max(np.abs(wave[30:-30]-0.5*np.sin(2*np.pi*1000*np.arange(1600)[30:-30]/16000)))<0.002
    assert converted[1]['audio'] is None
    assert converted[0]['audio']['source']['sha256']==source_digest==file_hash(raw)
    assert result['native_audio_compatible']==1 and result['training_admitted'] is False
    assert file_hash(output/'train.targets.jsonl')==files['train.targets.jsonl']
