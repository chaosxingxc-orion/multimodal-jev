import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from multimodal_jev.task_schema import DecisionSchema
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from test_omni_selection_contract import fixture

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
spec = importlib.util.spec_from_file_location('omni_select',
    Path(__file__).parents[1]/'tools/jev/select_omni_prompt.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_invalid_output_never_becomes_correct_unknown():
    request = dict(id='x', source_group='g', input={'schema':asdict(
        DecisionSchema('q', 'noul', 'Is it requested?', allow_unknown=True))})
    target = dict(id='x', target=None, metadata={'variant':'missing_candidate'})
    parent = {'x':{'raw_text':'```json\n{"value":null}\n```'}}
    candidate = {'x':{'raw_text':'{"value":null}'}}
    report = module.paired([request], [target], parent, candidate)
    assert report['by_view']['missing_candidate']['fixed'] == 1
    assert report['by_view']['missing_candidate']['both_correct'] == 0
    target['target'] = True
    report = module.paired([request], [target], {'x':{'raw_text':'{"value":1}'}},
                           {'x':{'raw_text':'{"value":true}'}})
    assert report['by_view']['missing_candidate']['fixed'] == 1


def test_selection_requires_all_preregistered_candidates(tmp_path):
    path = tmp_path/'protocol.json'
    path.write_text(json.dumps({'prompts':{'parent':'a', 'p1':'b', 'p2':'c'}}))
    with pytest.raises(ValueError, match='every registered'):
        module.select(path, {'parent':tmp_path, 'p1':tmp_path})


def test_loader_requires_container_success_and_unique_full_coverage(tmp_path):
    cfg, rows = fixture(tmp_path)
    selected = [r for r in rows if r['source_group'].startswith('s')]
    root = tmp_path/'parent'/'run'; root.mkdir(parents=True)
    def write(path, value):
        path.write_bytes(canonical(value))
    identity = dict(config=cfg, selection_protocol_sha256=cfg['selection_protocol_sha256'],
                    instruction_sha256=file_hash(cfg['instruction']),
                    requests_sha256=file_hash(cfg['requests']),
                    asset_audit_sha256=file_hash(cfg['asset_audit']))
    write(root/'identity.json', identity)
    write(root/'terminal.json', {'exit_code':0})
    write(root.parent/'evaluation-terminal.json', {'exit_code':0})
    write(root.parent/'launch.json', {'container_id':'cid', 'image':'image'})
    terminal = dict(container_id='cid', state={'Status':'exited','ExitCode':0})
    write(root.parent/'docker-terminal.json', terminal)
    write(root/'selection.json', {'ids':[r['id'] for r in selected]})
    predictions = [dict(example_id=r['id'], replicate=0, audio_condition='actual', split='train')
                   for r in selected]
    pred_path = root/'predictions.jsonl'
    pred_path.write_bytes(b''.join(canonical(p) for p in predictions))
    result = dict(status='passed', weight_updates=0, model_files_unchanged=True,
                  audio_condition='actual', predictions=1536, requests=1536,
                  predictions_sha256=file_hash(pred_path))
    write(root/'result.json', result)
    protocol = Path(cfg['selection_protocol'])
    assert len(module.load(root, protocol, 'parent')[2]) == 1536
    terminal['state']['ExitCode'] = 143
    write(root.parent/'docker-terminal.json', terminal)
    with pytest.raises(ValueError, match='successful'):
        module.load(root, protocol, 'parent')
    terminal['state']['ExitCode'] = 0
    write(root.parent/'docker-terminal.json', terminal)
    predictions[-1] = predictions[0]
    pred_path.write_bytes(b''.join(canonical(p) for p in predictions))
    result['predictions_sha256'] = file_hash(pred_path)
    write(root/'result.json', result)
    with pytest.raises(ValueError, match='unique'):
        module.load(root, protocol, 'parent')
