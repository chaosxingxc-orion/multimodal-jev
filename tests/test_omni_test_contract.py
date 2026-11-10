import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev import omni_test_contract as module


def fixture(tmp_path, monkeypatch):
    rows = [dict(id=f'{g}:{v}', source_group=str(g), split='test')
            for g in range(3793) for v in range(6)]
    def write(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return path
    request = tmp_path/'test.requests.jsonl'
    request.write_text('\n'.join(json.dumps(r) for r in rows))
    digest = file_hash(request)
    monkeypatch.setattr(module, 'REQUEST_SHA', digest)
    instruction = tmp_path/'instruction.txt'; instruction.write_text('frozen parent')
    parent_sha = file_hash(instruction)
    audit = write('audit.json', {})
    prior = write('selection-protocol.json', {'prompts':{'parent':parent_sha,'p1':'candidate'}})
    report = write('selection-result.json', dict(schema='jev-omni-prompt-selection-v1',
        selected_id='parent', selected_instruction_sha256=parent_sha,
        protocol_sha256=file_hash(prior), test_feedback_used=False, weight_updates=0))
    write('MANIFEST.json', dict(release_id=module.RELEASE, splits={'test':dict(
        requests_sha256=digest, targets_sha256=module.TARGET_SHA, requests=22758, source_groups=3793)}))
    cfg = dict(split='test',evaluation_phase='test',audio_condition='actual',parameters_frozen=True,
               requests=str(request),instruction=str(instruction),asset_audit=str(audit),seed=17)
    protocol = dict(schema='jev-omni-test-protocol-v1', actor_config=module.test_actor_config(cfg),
        prompts={'parent':parent_sha}, selection_result=str(report),selection_result_sha256=file_hash(report),
        selection_protocol=str(prior),selection_protocol_sha256=file_hash(prior),
        asset_audit_sha256=file_hash(audit), requests_sha256=digest,
        targets_sha256=module.TARGET_SHA,test_feedback_used=False)
    path = write('test-protocol.json',protocol)
    cfg.update(test_protocol=str(path),test_protocol_sha256=file_hash(path))
    return cfg, rows


def test_full_test_population_and_parent_deduplication(tmp_path, monkeypatch):
    cfg, rows = fixture(tmp_path, monkeypatch)
    assert len(module.select_test(cfg,rows)) == 22758
    for changed in [rows[:-1], rows[:-1]+[rows[0]],
                    rows[:-1]+[{**rows[-1],'split':'train'}],
                    rows[:-1]+[{**rows[-1],'source_group':'extra'}]]:
        with pytest.raises(ValueError,match='all3793'):
            module.select_test(cfg,changed)
    for change in [dict(source_groups=['0']),dict(split='train'),dict(seed=42),
                   dict(audio_condition='zeros'),dict(parameters_frozen=False)]:
        with pytest.raises(ValueError):
            module.select_test({**cfg,**change},rows)


@pytest.mark.parametrize('artifact',['instruction','asset_audit','test_protocol','requests',
                                    'selection-result.json','selection-protocol.json'])
def test_changed_prompt_or_selection_custody_rejected(tmp_path,monkeypatch,artifact):
    cfg, rows = fixture(tmp_path,monkeypatch)
    path = Path(cfg[artifact]) if artifact in cfg else tmp_path/artifact
    with path.open('a') as handle: handle.write(' ')
    with pytest.raises(ValueError): module.select_test(cfg,rows)
