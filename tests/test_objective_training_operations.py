import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_objective_training_operations as module
from tools.reports import context_objective_storage as storage
from tests.test_context_cached_training_operations import fixture as cached_fixture,write


def fixture(tmp_path,monkeypatch):
    c,proof=cached_fixture(tmp_path,monkeypatch)
    parent=copy.deepcopy(c['training_package']);spec=dict(parent=parent,
        output=parent['output'],manifest_sha256=parent['manifest_sha256'],remote_root=parent['remote_root'])
    c['training_package']=spec;p=Path(c['package']);config=module.read(p/'config.json')
    config['objective_extension']=dict(root=spec['remote_root']+'/objective-extension',files={'bootstrap.py':'a'*64})
    config['pins'][config['objective_extension']['root']+'/bootstrap.py']='a'*64
    write(p/'config.json',config)
    monkeypatch.setattr(module,'audit_package',lambda *a:dict(status='objective_training_candidate_reconstructed_not_admitted'))
    monkeypatch.setattr(module,'bootstrap_evidence',lambda *a:dict(tests=18))
    return c,proof


def test_original_reference_gate_and_extension_mount(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    result=module.admission(c);m,cfg,_=module.validate(c)
    assert result['cached_prerequisites']['actor_updates']==2
    argv=module.command(c,m,cfg);root=cfg['objective_extension']['root']
    binds=storage.mounts(argv)
    assert (root,root,True) in binds
    assert not any(Path(dst).is_relative_to(root) and dst!=root for _,dst,_ in binds)
    assert (cfg['reference_dataset']['root'],cfg['reference_dataset']['root'],True) in binds


@pytest.mark.parametrize('fault',['runtime','population','reference','endpoint'])
def test_candidate_cannot_bypass_original_admission(tmp_path,monkeypatch,fault):
    c,proof=fixture(tmp_path,monkeypatch)
    if fault=='runtime':proof['numerical_admission']['runtime_identity']={}
    elif fault=='population':proof['output_audit']['recordings']=32
    elif fault=='reference':c['training_package']['parent']['reference_job']['collection']='wrong'
    else:
        write(Path(c['package'])/'initial/endpoint-commit.json',{'substitution':True})
    with pytest.raises(ValueError):module.admission(c)


def test_dispatch_disabled_does_not_launch():
    with pytest.raises(ValueError,match='two-update'):
        module.launch({'finite_training_dispatch_allowed':False},{})


def test_bootstrap_report_binds_actual_tested_extension(tmp_path):
    root=tmp_path/'package';snapshot=tmp_path/'snapshot';source=snapshot/'tools/reports/context_objective_bootstrap.py'
    source.parent.mkdir(parents=True);source.write_text('test bytes')
    h=module.file_hash(source);write(snapshot/'FILES.json',{'tools/reports/context_objective_bootstrap.py':h})
    write(root/'PREPARED.json',{'image':'fixed'})
    write(root/'config.json',{'objective_extension':{'files':{'bootstrap.py':h}}})
    report=tmp_path/'report.json'
    write(report,dict(counts=dict(tests=18,failures=0,errors=0,skipped=0),image='fixed',
        state=dict(ExitCode=0,OOMKilled=False),terminal=dict(exit_code=0),
        source_manifest_sha256=module.file_hash(snapshot/'FILES.json')))
    c=dict(training_package={'output':str(root)},bootstrap_cpu_source=str(snapshot),
        bootstrap_cpu_report=dict(path=str(report),sha256=module.file_hash(report)))
    assert module.bootstrap_evidence(c)['tests']==18
    source.write_text('different code')
    with pytest.raises(ValueError,match='different extension'):module.bootstrap_evidence(c)


@pytest.mark.parametrize('failure',[False,True])
def test_run_collects_then_independently_audits(tmp_path,monkeypatch,failure):
    import multimodal_jev.local_runtime_storage as local
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(local,'check_paths',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    c=dict(operations_root=str(tmp_path/'operations'),tag='objective-test',training_package={},objective_loss_atol=0.)
    order=[]
    def launch(*a):order.append('collect');return dict(status='collected_independent_objective_training_audit_required')
    def audit(cfg,prov):
        order.append('audit');assert cfg['transport'].endswith('-collected.transport.json')
        if failure:raise ValueError('output rejected')
        return dict(status='fixture_complete')
    monkeypatch.setattr(module,'launch',launch);monkeypatch.setattr(module,'audit_training',audit)
    if failure:
        with pytest.raises(ValueError,match='output rejected'):module.run(c,{})
    else:assert module.run(c,{})['status']=='fixture_complete'
    terminal=module.read(Path(c['operations_root'])/'finalization/terminal.json')
    assert terminal['exit_code']==int(failure) and order==['collect','audit']
    if failure:assert terminal['restart'] is False and terminal['preserve_handles'] is True
    else:assert terminal['formal_training_admitted'] is False
