import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('teacher_prepare',Path(__file__).parents[1]/'tools/jev/prepare_teacher_feedback.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('route', ['asr-text', 'audio-direct'])
def test_retired_teacher_entry_never_prepares_or_dispatches(tmp_path, route):
    run=tmp_path/'run';run.mkdir()
    (run/'identity.json').write_text(json.dumps({'config':{'route':route,'audio_model':'base'}}))
    output=tmp_path/'output'
    with pytest.raises(ValueError,match='retired'):
        module.prepare(run,tmp_path/'gold',output,tmp_path/'executor',tmp_path/'config')
    assert not output.exists()
