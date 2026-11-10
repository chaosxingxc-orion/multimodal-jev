import importlib.util
import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from multimodal_jev.task_schema import DecisionSchema
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from test_omni_selection_contract import fixture
from test_six_view_score import fixture as six_view_fixture
from multimodal_jev.omni_selection_contract import actor_config

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


def test_full256_selection_scores_raw_outputs_and_rejects_runtime_drift(tmp_path):
    cfg, _ = fixture(tmp_path)
    rows, targets, predictions = [], [], []
    for group in cfg['source_groups']:
        rr, tt, pp = copy.deepcopy(six_view_fixture())
        for r,t,p in zip(rr,tt,pp):
            uid=group+':'+t['metadata']['variant']
            r.update(id=uid,source_group=group)
            t.update(id=uid);t['metadata']['source_group']=group
            digest=hashlib.sha256(canonical(r)).hexdigest()
            t['request_sha256']=digest
            p.update(example_id=uid,request_sha256=digest,split='train',audio_condition='actual')
        rows.extend(rr);targets.extend(tt);predictions.extend(pp)
    request_path=Path(cfg['requests']);request_path.write_bytes(b''.join(map(canonical,rows)))
    request_path.with_name('train.targets.jsonl').write_bytes(b''.join(map(canonical,targets)))
    candidate=tmp_path/'candidate.txt';candidate.write_text('return exact task JSON')
    protocol_path=Path(cfg['selection_protocol']);protocol=json.loads(protocol_path.read_text())
    protocol.update(requests_sha256=file_hash(request_path),actor_config=actor_config(cfg),rule='registered paired rule')
    protocol['prompts']['p1']=file_hash(candidate)
    protocol_path.write_bytes(canonical(protocol));cfg['selection_protocol_sha256']=file_hash(protocol_path)
    runs={}
    for key in ['parent','p1']:
        root=tmp_path/key/'run';root.mkdir(parents=True);runs[key]=root
        local_cfg={**cfg,'instruction':str(candidate) if key=='p1' else cfg['instruction']}
        identity=dict(config=local_cfg,selection_protocol_sha256=file_hash(protocol_path),
            instruction_sha256=file_hash(local_cfg['instruction']),requests_sha256=file_hash(request_path),
            asset_audit_sha256=file_hash(cfg['asset_audit']),model_files=['fixed'],runner_sha256='runner',
            source_sha256={'module':'fixed'},versions={'vllm':'fixed'},population_selector_sha256='selector')
        def write(path,value):path.write_bytes(canonical(value))
        write(root/'identity.json',identity)
        write(root/'terminal.json',{'exit_code':0})
        write(root.parent/'evaluation-terminal.json',{'exit_code':0})
        write(root.parent/'launch.json',{'container_id':key,'image':'fixed-image'})
        write(root.parent/'docker-terminal.json',dict(container_id=key,state={'Status':'exited','ExitCode':0}))
        write(root/'selection.json',{'ids':[r['id'] for r in rows]})
        pp=copy.deepcopy(predictions)
        if key=='parent':
            for p in pp:
                if p['example_id'].endswith('missing_candidate'):
                    p['raw_text']='```json\n'+p['raw_text']+'\n```'
        (root/'predictions.jsonl').write_bytes(b''.join(map(canonical,pp)))
        write(root/'result.json',dict(status='passed',weight_updates=0,model_files_unchanged=True,
            audio_condition='actual',requests=1536,predictions=1536,predictions_sha256=file_hash(root/'predictions.jsonl')))
    report=module.select(protocol_path,runs)
    assert report['selected_id']=='p1'
    by_id={r['id']:r for r in report['comparisons']}
    assert by_id['parent']['metrics']['all_six_correct']==0
    assert by_id['p1']['metrics']['all_six_correct']==256
    assert by_id['p1']['paired_to_parent']['by_view']['missing_candidate']['fixed']==256
    assert report['test_feedback_used'] is False
    path=runs['p1']/'identity.json';identity=json.loads(path.read_text())
    identity['versions']['vllm']='changed';path.write_bytes(canonical(identity))
    with pytest.raises(ValueError,match='actor identity differs'):
        module.select(protocol_path,runs)
