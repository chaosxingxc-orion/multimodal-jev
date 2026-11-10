import hashlib
import inspect
import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev import omni_slurp_transfer_contract as contract

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import verify_omni_transfer_train as verification


def fixture(tmp_path,monkeypatch):
    rows=[dict(id=f'{i}::{j}',split='train',source_group='slurp:12507') for i in range(2) for j in range(168)]
    requests=tmp_path/'train.requests.jsonl';requests.write_bytes(b''.join(map(canonical,rows)))
    prompt=tmp_path/'prompt.txt';prompt.write_text('parent')
    audit=tmp_path/'audit.json';audit.write_text('{}')
    monkeypatch.setattr(contract,'TRAIN_REQUEST_SHA',file_hash(requests))
    monkeypatch.setattr(contract,'PARENT_PROMPT_SHA',file_hash(prompt))
    cfg=dict(split='train',evaluation_phase='slurp-jev-transfer-train',task_format='jev',audio_condition='actual',
        parameters_frozen=True,max_new_tokens=128,max_context_tokens=4096,dtype='bfloat16',seed=17,
        sampling={'temperature':0.,'n':1},source_groups=['slurp:12507'],requests=str(requests),
        instruction=str(prompt),asset_audit=str(audit),audio_model='/model',route='audio-direct',model_family='qwen3_omni_moe')
    return cfg,rows


def test_fixed_train_gate_rejects_subset_and_task_changes(tmp_path,monkeypatch):
    cfg,rows=fixture(tmp_path,monkeypatch)
    assert contract.select_transfer_train(cfg,rows)==rows
    for changes in [dict(split='test'),dict(max_new_tokens=256),dict(source_groups=['other'])]:
        with pytest.raises(ValueError):contract.select_transfer_train({**cfg,**changes},rows)
    for altered in [rows[:-1],rows[:-1]+[rows[0]],list(reversed(rows))]:
        with pytest.raises(ValueError,match='all168'):contract.select_transfer_train(cfg,altered)


def test_train_validation_requires_real_success_custody_and_context(tmp_path,monkeypatch):
    cfg,rows=fixture(tmp_path,monkeypatch);run=tmp_path/'run';run.mkdir()
    def write(path,data):path.write_bytes(canonical(data))
    identity=dict(config=cfg,runner_sha256=file_hash(inspect.getfile(verification.inference_run)),
        population_selector_sha256=file_hash(inspect.getfile(contract.select_transfer_train)),
        requests_sha256=file_hash(cfg['requests']),instruction_sha256=file_hash(cfg['instruction']),
        asset_audit_sha256=file_hash(cfg['asset_audit']),source_sha256={'omni_slurp_transfer_contract.py':file_hash(inspect.getfile(contract.select_transfer_train))},
        versions={'vllm':'fixed'},model_files=['fixed'])
    write(run/'identity.json',identity);write(run/'terminal.json',{'exit_code':0})
    write(tmp_path/'evaluation-terminal.json',{'exit_code':0})
    write(tmp_path/'launch.json',dict(container_id='cid',image='image',physical_devices=[6,7],source_commit='commit'))
    terminal=dict(container_id='cid',state={'Status':'exited','ExitCode':0})
    write(tmp_path/'docker-terminal.json',terminal)
    write(run/'selection.json',{'ids':[r['id'] for r in rows]})
    predictions=[dict(example_id=r['id'],request_sha256=hashlib.sha256(canonical(r)).hexdigest(),
        split='train',replicate=0,audio_condition='actual',input_tokens=1000,raw_text='invalid') for r in rows]
    (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,predictions)))
    result=dict(status='passed',weight_updates=0,model_files_unchanged=True,audio_condition='actual',requests=336,predictions=336,predictions_sha256=file_hash(run/'predictions.jsonl'))
    write(run/'result.json',result)
    # No quality threshold: even invalid JSON is valid engineering evidence.
    assert verification.verify(tmp_path)['requests']==336
    terminal['state']['ExitCode']=143;write(tmp_path/'docker-terminal.json',terminal)
    with pytest.raises(ValueError,match='successful'):verification.verify(tmp_path)
    terminal['state']['ExitCode']=0;write(tmp_path/'docker-terminal.json',terminal)
    predictions[0]['input_tokens']=4096
    (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,predictions)))
    result['predictions_sha256']=file_hash(run/'predictions.jsonl');write(run/'result.json',result)
    with pytest.raises(ValueError,match='context budget'):verification.verify(tmp_path)
