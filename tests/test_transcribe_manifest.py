import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('transcribe_manifest',
                                              Path(__file__).parents[1] / 'tools/jev/transcribe_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
edit_distance, normalize = module.edit_distance, module.normalize


def test_transcription_metrics_are_token_based():
    assert normalize("Delete My To-Do list!") == ['delete', 'my', 'to', 'do', 'list']
    assert edit_distance(['a', 'b'], ['a', 'c', 'b']) == 1
    assert edit_distance([], ['a']) == 1
