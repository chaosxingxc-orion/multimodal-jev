import importlib.util
import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash

spec = importlib.util.spec_from_file_location('omni_teacher_config',
    Path(__file__).parents[1]/'tools/jev/configure_omni_teacher.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_bounded_configuration_no_dispatch_or_duplicate_attempt(tmp_path, monkeypatch):
    context = dict(schema='jev-generative-audio-ace-feedback-v1', split='train',
        selection_or_test_feedback=False, weight_updates=0, zeros_semantic_gold_available=False,
        actor_model='Qwen/Qwen3-Omni-30B-A3B-Instruct')
    messages = tmp_path/'messages.private.json'
    messages.write_text(json.dumps([{'role':'system','content':'instruction'},
                                    {'role':'user','content':json.dumps(context)}]))
    prepared = dict(messages_sha256=file_hash(messages), API_sent=False, learning_groups=64,
        summary_records=384, selection_or_test_feedback_used=False, max_candidates=2,
        max_instruction_characters=1500)
    (tmp_path/'prepared.json').write_text(json.dumps(prepared))
    executor = tmp_path/'executor.py'; executor.write_text('raise RuntimeError("must not execute")')
    monkeypatch.setattr(module, 'EXECUTOR_SHA', file_hash(executor))
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(model='glm-5.3-flash', endpoint='https://example.test/chat/completions',
                                        api_key_env='JEV_TEST_CREDENTIAL')))
    monkeypatch.delenv('JEV_TEST_CREDENTIAL', raising=False)
    prepared['selection_or_test_feedback_used'] = True
    (tmp_path/'prepared.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='bounded'):
        module.configure(tmp_path, reference, executor)
    assert not (tmp_path/'executor-config.private.json').exists()
    prepared['selection_or_test_feedback_used'] = False
    (tmp_path/'prepared.json').write_text(json.dumps(prepared))
    receipt = module.configure(tmp_path, reference, executor)
    assert receipt['API_sent'] is False and receipt['max_new_requests'] == 1
    assert receipt['credential_available'] is False
    assert not (tmp_path/'response').exists()
    assert (tmp_path/'executor-config.private.json').stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        module.configure(tmp_path, reference, executor)
