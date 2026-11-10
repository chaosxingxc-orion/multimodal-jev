import json
from pathlib import Path

import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.direct_test_lock import validate_population


def fixture(tmp_path):
    requests = tmp_path/'test.requests.jsonl'
    requests.write_text('{}\n')
    instruction = tmp_path/'prompt.txt'; instruction.write_text('listen')
    cfg = dict(split='test', audio_condition='actual', source_groups=['a','b'],
               requests=str(requests), instruction=str(instruction),
               purpose='direct-audio-locked-full-test', heldout_lock=str(tmp_path/'lock.json'),
               dtype='bfloat16', parameters_frozen=True)
    lock = dict(schema='jev-direct-heldout-lock-v1', test_feedback_used=False,
                source_selection_split='train', requests=2, source_groups=2,
                requests_sha256=file_hash(requests), parent_prompt_sha256=file_hash(instruction),
                selected_prompt_sha256='other', actor_config=dict(audio_condition='actual',
                dtype='bfloat16', parameters_frozen=True))
    Path(cfg['heldout_lock']).write_text(json.dumps(lock))
    rows = [dict(id=g, source_group=g, split='test') for g in ['a','b']]
    return cfg, rows


def test_full_test_requires_locked_prompt_population_and_actor(tmp_path):
    cfg,rows=fixture(tmp_path)
    assert validate_population(cfg,rows)==rows
    for change in [dict(source_groups=['a']),dict(source_groups=['a','a']),
                   dict(dtype='float16'),dict(audio_condition='zeros'),dict(split='valid')]:
        with pytest.raises(ValueError): validate_population({**cfg,**change},rows)
    Path(cfg['instruction']).write_text('new instruction')
    with pytest.raises(ValueError,match='prompt'): validate_population(cfg,rows)


def test_modified_requests_and_split_rejected(tmp_path):
    cfg,rows=fixture(tmp_path)
    with pytest.raises(ValueError):validate_population(cfg,rows+[rows[0]])
    with pytest.raises(ValueError):validate_population(cfg,[{**r,'split':'train'} for r in rows])
    Path(cfg['requests']).write_text('changed')
    with pytest.raises(ValueError,match='coverage'):validate_population(cfg,rows)


def test_train_does_not_require_test_lock(tmp_path):
    cfg,rows=fixture(tmp_path)
    cfg.update(split='train',source_groups=['a']);cfg.pop('heldout_lock')
    rows=[{**r,'split':'train'} for r in rows]
    assert validate_population(cfg,rows)==rows[:1]
