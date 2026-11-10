import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev import omni_test_contract as contract
from test_omni_test_contract import fixture as contract_fixture
from test_six_view_score import fixture as six_fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import compare_omni_test as comparison
import register_omni_test as registration
import run_omni_test_queue as queue


def test_full_test_strict_scoring_and_runtime_drift(tmp_path,monkeypatch):
    cfg,_=contract_fixture(tmp_path,monkeypatch)
    rows,targets,predictions=[],[],[]
    base=six_fixture()
    for group in range(3793):
        rr,tt,pp=copy.deepcopy(base)
        for r,t,p in zip(rr,tt,pp):
            uid=str(group)+':'+t['metadata']['variant']
            r.update(id=uid,source_group=str(group),split='test')
            t.update(id=uid);t['metadata']['source_group']=str(group)
            digest=hashlib.sha256(canonical(r)).hexdigest();t['request_sha256']=digest
            p.update(example_id=uid,request_sha256=digest,split='test',audio_condition='actual')
            if uid.endswith('missing_candidate'):
                p['raw_text']='```json\n'+p['raw_text']+'\n```'
        rows.extend(rr);targets.extend(tt);predictions.extend(pp)
    Path(cfg['requests']).write_bytes(b''.join(map(canonical,rows)))
    target=tmp_path/'test.targets.jsonl';target.write_bytes(b''.join(map(canonical,targets)))
    monkeypatch.setattr(contract,'REQUEST_SHA',file_hash(cfg['requests']))
    monkeypatch.setattr(contract,'TARGET_SHA',file_hash(target))
    manifest=json.loads((tmp_path/'MANIFEST.json').read_text())
    manifest['splits']['test'].update(requests_sha256=contract.REQUEST_SHA,targets_sha256=contract.TARGET_SHA)
    (tmp_path/'MANIFEST.json').write_bytes(canonical(manifest))
    protocol_path=Path(cfg['test_protocol']);protocol=json.loads(protocol_path.read_text())
    protocol.update(requests_sha256=contract.REQUEST_SHA,targets_sha256=contract.TARGET_SHA,
        image='fixed-image',actor_identity=dict(runner_sha256='runner',source_sha256={'source':'fixed'},
                                             versions={'vllm':'fixed'},model_files=['model']))
    protocol_path.write_bytes(canonical(protocol));cfg['test_protocol_sha256']=file_hash(protocol_path)
    root=tmp_path/'parent/run';root.mkdir(parents=True)
    def write(path,data):path.write_bytes(canonical(data))
    identity=dict(config=cfg,test_protocol_sha256=file_hash(protocol_path),
        instruction_sha256=file_hash(cfg['instruction']),asset_audit_sha256=file_hash(cfg['asset_audit']),
        requests_sha256=contract.REQUEST_SHA,population_selector_sha256=file_hash(inspect.getfile(contract.select_test)),
        **protocol['actor_identity'])
    write(root/'identity.json',identity)
    write(root/'terminal.json',{'exit_code':0})
    write(root.parent/'evaluation-terminal.json',{'exit_code':0})
    write(root.parent/'launch.json',dict(container_id='cid',image='fixed-image',physical_devices=[4,5]))
    terminal=dict(container_id='cid',state={'Status':'exited','ExitCode':0})
    write(root.parent/'docker-terminal.json',terminal)
    write(root/'selection.json',{'ids':[r['id'] for r in rows]})
    (root/'predictions.jsonl').write_bytes(b''.join(map(canonical,predictions)))
    write(root/'result.json',dict(status='passed',weight_updates=0,model_files_unchanged=True,
        audio_condition='actual',requests=22758,predictions=22758,length_limited=0,
        predictions_sha256=file_hash(root/'predictions.jsonl')))
    report=comparison.compare(tmp_path)
    metrics=report['comparisons'][0]['metrics']
    assert metrics['by_variant']['native']['correct']==3793
    assert metrics['by_variant']['missing_candidate']['correct']==0
    assert metrics['all_six_correct']==0
    assert len(report['comparisons'])==1  # Selected parent never duplicated.
    identity['versions']['vllm']='drift';write(root/'identity.json',identity)
    with pytest.raises(ValueError,match='identity changed'):comparison.compare(tmp_path)
    identity['versions']['vllm']='fixed';write(root/'identity.json',identity)
    terminal['state']['ExitCode']=143;write(root.parent/'docker-terminal.json',terminal)
    with pytest.raises(ValueError,match='successful'):comparison.compare(tmp_path)


def test_registration_requires_completed_reproducible_selection(tmp_path,monkeypatch):
    (tmp_path/'evaluation-terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError,match='successful'):registration.register(tmp_path,tmp_path/'out.json')
    (tmp_path/'evaluation-terminal.json').write_text('{"exit_code":0}')
    (tmp_path/'selection-protocol.json').write_text('{"prompts":{"parent":"a","p1":"b"}}')
    (tmp_path/'selection-result.json').write_text('{"selected_id":"p1"}')
    monkeypatch.setattr(registration,'select',lambda *args:{'selected_id':'parent'})
    with pytest.raises(ValueError,match='reproduce'):registration.register(tmp_path,tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()


def test_test_dispatch_preserves_device_mapping_and_mounts_selection(tmp_path):
    selection=tmp_path/'selection';parent=selection/'parent';root=tmp_path/'test';child=root/'parent'
    argv=['docker','run','--name','selection-parent','--device','/dev/davinci4:/dev/davinci4',
          '--device','/dev/davinci5:/dev/davinci5','-v',str(parent)+':'+str(parent)+':rw',
          'image','timeout','--signal=TERM','--kill-after=30s','7200s','python3',
          str(parent/'source/tools/jev/infer_qwen3_omni_selection.py')]
    changed=queue.test_argv(argv,parent,child,selection,root,'parent')
    assert changed[:2]==['docker','run']
    assert changed[changed.index('--name')+1]=='test-parent'
    assert str(selection)+':'+str(selection)+':ro' in changed
    assert str(child)+':'+str(child)+':rw' in changed
    assert changed[-1]==str(child/'source/tools/jev/infer_qwen3_omni_test.py')
    assert changed[-3]=='86400s'
    assert '/dev/davinci4:/dev/davinci4' in changed and '/dev/davinci5:/dev/davinci5' in changed


def test_no_automatic_test_replay(tmp_path,monkeypatch):
    protocol=tmp_path/'test-protocol.json';protocol.write_text('{}')
    (tmp_path/'queue-intent.json').write_text(json.dumps(dict(order=['parent'],protocol_sha256=file_hash(protocol))))
    (tmp_path/'parent').mkdir();(tmp_path/'parent/dispatch.log').write_text('ambiguous')
    monkeypatch.setattr(queue,'dispatch_and_score',lambda *_:pytest.fail('must not dispatch'))
    with pytest.raises(FileExistsError,match='replay'):queue.run(tmp_path)
