import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_slurp_transfer_contract import select_transfer_shard
from test_omni_slurp_transfer import fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import compare_omni_slurp_transfer as module


def test_loader_rejects_failed_container_missing_predictions_and_changed_runtime(tmp_path):
    cfg,rows=fixture(tmp_path)
    path=Path(cfg['transfer_protocol']);protocol=json.loads(path.read_text())
    protocol.update(image='image',versions={'vllm':'fixed'},model_files=['fixed'])
    path.write_bytes(canonical(protocol));cfg['transfer_protocol_sha256']=file_hash(path)
    run=tmp_path/'shard/run';run.mkdir(parents=True)
    def write(p,data):p.write_bytes(canonical(data))
    identity=dict(config=cfg,transfer_protocol_sha256=file_hash(path),instruction_sha256=file_hash(cfg['instruction']),
        requests_sha256=file_hash(cfg['requests']),asset_audit_sha256=file_hash(cfg['asset_audit']),
        versions=protocol['versions'],model_files=protocol['model_files'],
        population_selector_sha256=file_hash(inspect.getfile(select_transfer_shard)))
    write(run/'identity.json',identity);write(run/'terminal.json',{'exit_code':0})
    write(run.parent/'evaluation-terminal.json',{'exit_code':0})
    write(run.parent/'launch.json',dict(container_id='cid',image='image',physical_devices=[6,7]))
    terminal=dict(container_id='cid',state={'Status':'exited','ExitCode':0})
    write(run.parent/'docker-terminal.json',terminal)
    write(run/'selection.json',{'ids':[r['id'] for r in rows]})
    predictions=[dict(example_id=r['id'],replicate=0,split='test',audio_condition='actual',
        length_limited=False,request_sha256=hashlib.sha256(canonical(r)).hexdigest(),raw_text='invalid') for r in rows]
    (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,predictions)))
    result=dict(status='passed',weight_updates=0,model_files_unchanged=True,audio_condition='actual',
        requests=len(rows),predictions=len(rows),length_limited=0,predictions_sha256=file_hash(run/'predictions.jsonl'))
    write(run/'result.json',result)
    assert len(module.load(run,path,'parent','shard-0000')[1])==840
    terminal['state']['ExitCode']=143;write(run.parent/'docker-terminal.json',terminal)
    with pytest.raises(ValueError,match='successful'):module.load(run,path,'parent','shard-0000')
    terminal['state']['ExitCode']=0;write(run.parent/'docker-terminal.json',terminal)
    identity['versions']={'vllm':'drift'};write(run/'identity.json',identity)
    with pytest.raises(ValueError,match='identity changed'):module.load(run,path,'parent','shard-0000')
    identity['versions']=protocol['versions'];write(run/'identity.json',identity)
    predictions[-1]=predictions[0]
    (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,predictions)))
    result['predictions_sha256']=file_hash(run/'predictions.jsonl');write(run/'result.json',result)
    with pytest.raises(ValueError,match='unique'):module.load(run,path,'parent','shard-0000')


def test_aggregate_rejects_incomplete_roster_before_gold_access(tmp_path):
    cfg,_=fixture(tmp_path)
    # This source fixture has no target files. Missing roster must fail first.
    with pytest.raises(ValueError,match='every registered'):
        module.compare(Path(cfg['transfer_protocol']),{'parent':{}},tmp_path/'scorer')
