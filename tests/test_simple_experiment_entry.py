"""Exercise actual CLI imports and config pins from a private source copy, on CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports import run_composed_stage2 as stage2

ROOT=Path(__file__).resolve().parents[1]


def test_cli_frozen_plan_config_change_and_source_change(tmp_path):
    spec=freeze(ROOT,tmp_path/'source',patterns=['tools/reports/run_simple_experiment.py',
        'configs/portable/operations/simple-experiment.json','configs/portable/operations/stage2-dataset.json',
        'tools/reports/prepare_stage2_dataset.py','src/multimodal_jev/*.py'],io_workers=2)
    root=Path(spec['root'])
    local=tmp_path/'local.json';env=dict(os.environ,PYTHONPATH='src:.:tools/jev',PYTHONDONTWRITEBYTECODE='1')
    def save(macros):local.write_text(json.dumps({'schema':'jev-local-macros-v1','macros':macros}))
    macros=dict(EXPERIMENT_STAGE='stage2_dataset',EXPERIMENT_IO_WORKERS=2,EXPERIMENT_SOURCE_SNAPSHOT=spec,
        EXPERIMENT_DRIVER_ROOT=str(tmp_path/'driver'),AUDITED_STAGE1_REFERENCE_JOB={},STAGE1_DATASET_ENDPOINT={},
        STAGE2_DATASET_OUTPUT=str(tmp_path/'data'),STAGE2_DATASET_SHA256='',STAGE2_DATASET_RECEIPTS={'release':str(tmp_path/'receipt')},
        STAGE2_INPUT_CONFIG={},STAGE2_BOUND_CONFIG=str(tmp_path/'bound'))
    save(macros)
    _,p=load_config(root/'configs/portable/operations/stage2-dataset.json',local)
    macros['EXPERIMENT_BACKEND_SHA256']=p['resolved_semantic_sha256'];save(macros)
    argv=[sys.executable,'-B','-m','tools.reports.run_simple_experiment','--config',
        'configs/portable/operations/simple-experiment.json','--local',str(local)]
    def call(*args):return subprocess.run(argv+list(args),cwd=root,env=env,capture_output=True,text=True)
    result=call();assert result.returncode==0,result.stderr
    assert not json.loads(result.stdout)['executed'] and not (tmp_path/'driver').exists()
    save(dict(macros,STAGE2_DATASET_OUTPUT=str(tmp_path/'other')))
    result=call('--action','run');assert result.returncode and 'configuration changed' in result.stderr
    save(macros);root.chmod(0o755);(root/'extra.py').touch()
    result=call('--action','run');assert result.returncode and 'snapshot membership changed' in result.stderr
    terminal=json.loads((tmp_path/'driver/terminal.json').read_bytes());assert not terminal['restart'] and terminal['exit_code']==1
    assert not (tmp_path/'data').exists()
    result=call('--action','run');assert result.returncode and 'FileExistsError' in result.stderr


def test_full_stage2_refuses_view_before_expensive_checks(monkeypatch):
    monkeypatch.setattr(stage2,'verify',lambda *a,**kw:pytest.fail('snapshot read before scope check'))
    with pytest.raises(ValueError,match='data-only'):
        stage2.run(dict(prefix_spec=dict(training_dataset={'root':'wrong'})),{})


def test_stage2_detach_leaves_evidence_validation_to_owner(tmp_path,monkeypatch):
    import types
    c=dict(output=str(tmp_path/'stage2'));p=dict(resolved_semantic_sha256='a'*64)
    calls=[]
    monkeypatch.setattr(stage2,'load_config',lambda *a:(c,p))
    monkeypatch.setattr(stage2,'validate_request',lambda cfg:calls.append(cfg))
    monkeypatch.setattr(stage2,'validate_evidence',lambda cfg:pytest.fail('heavy evidence repeated at detach'))
    monkeypatch.setattr(stage2.subprocess,'Popen',lambda *a,**kw:types.SimpleNamespace(pid=os.getpid()))
    record=stage2.detach('template','local')
    assert record['argv'][-2:]==['--request-sha256','a'*64] and calls==[c]


def test_stage1_plan_rejects_changed_design_and_unadmitted_microbatch(tmp_path,monkeypatch):
    import hashlib
    from tools.reports import run_simple_experiment as entry
    monkeypatch.setattr(entry,'ROOT',tmp_path);monkeypatch.chdir(tmp_path)
    src=ROOT/'configs/experiments/context-two-stage-margin-v0.0.123.json'
    design=json.loads(src.read_bytes());path=tmp_path/'design.json'
    spec=dict(schema='jev-simple-experiment-v1',stage='stage1',source_snapshot=dict(root=str(tmp_path)),
        io_workers=2,backend_sha256='key',driver_root=str(tmp_path.parent/'driver'))
    cfg=dict(package=dict(design=str(path),physical_devices=[0,1,2,3],package='/pkg',remote_root='/remote'),
        finite=dict(package='/pkg',remote_root='/remote'),output=str(tmp_path.parent/'out'))
    def bind():
        path.write_text(json.dumps(design));cfg['design_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    bind()
    monkeypatch.setattr(entry,'load_config',lambda t,l:(spec,{}) if t=='spec' else (cfg,dict(resolved_semantic_sha256='key')))
    assert entry.request('spec','local')[1]['io_workers']==2
    for microbatch in (2,4):
        design['stage1']['batch_config']['microbatch']=microbatch
        bind()
        assert entry.request('spec','local')[1]['package']['design']==str(path)
    design['stage1']['batch_config']['microbatch']=16;path.write_text(json.dumps(design))
    with pytest.raises(ValueError,match='design changed'):entry.request('spec','local')
    bind()
    with pytest.raises(ValueError,match='microbatch'):entry.request('spec','local')


def test_stage_monitor_passes_exact_backend_without_import_rebinding(tmp_path):
    from tools.reports.context_cached_formal_successor import monitored_training
    ops=dict(host='h',tag='run',operations_root='unused',manifest_sha256='f'*64)
    calls=[]
    def backend(c,p):calls.append((c,p));return 21
    assert monitored_training(ops,{},tmp_path,run_stage=backend)==21
    assert calls==[(ops,{})]
    assert not json.loads((tmp_path/'monitor-terminal.json').read_text())['training_success_verified']
