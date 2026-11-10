import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('selected_supervisor',Path(__file__).parents[1]/'tools/jev/run_selected_frozen_test.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('route', ['asr-text', 'audio-direct'])
def test_retired_asr_reuse_supervisor_never_waits_or_dispatches(tmp_path,monkeypatch,route):
    (tmp_path/'config-template.json').write_text(json.dumps({'route':route,'audio_model':'base'}))
    monkeypatch.setattr(module,'wait_container',lambda *a: pytest.fail('must not wait'))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k: pytest.fail('must not dispatch'))
    with pytest.raises(ValueError,match='retired'):module.execute(tmp_path)
    assert not (tmp_path/'dispatch-intent.json').exists()
