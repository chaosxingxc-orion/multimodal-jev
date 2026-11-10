import importlib.util
from pathlib import Path
import pytest
from multimodal_jev.end_to_end import require_audio_task_route


def test_direct_audio_route():
    require_audio_task_route({'route':'audio-direct','audio_model':'audio-checkpoint'})


@pytest.mark.parametrize('change', [
    {'route':'asr-text'}, {'route':'oracle'}, {'text_model':'text-checkpoint'},
    {'asr_evidence':{'path':'cache'}}, {'reference_transcripts':'gold'}, {'audio_model':None}])
def test_cascade_and_transcript_bypass_rejected(change):
    with pytest.raises(ValueError):
        require_audio_task_route({'route':'audio-direct','audio_model':'audio-checkpoint',**change})


def test_runner_rejects_cascade_before_importing_runtime_or_writing_output(tmp_path):
    spec=importlib.util.spec_from_file_location('frozen_runner',
        Path(__file__).parents[1]/'tools/jev/infer_frozen_requests.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='retired'):module.run({'route':'asr-text'},tmp_path)
    assert not list(tmp_path.iterdir())
