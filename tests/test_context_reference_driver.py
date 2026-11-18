import json
from pathlib import Path

import pytest

from tools.reports import context_reference_job_operations as module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def predecessor(tmp_path):
    base=tmp_path/'parity'
    deployment=dict(operations_root=str(base),host='fixture',lease='fixture-lease')
    audit_config=dict(deployment=deployment,parity_package={'fixture':True})
    prior={key:str(base/path) for key,path in {
        'driver':'driver/started.json','config':'finalization/config.json',
        'terminal':'finalization/terminal.json','audit_config':'finalization/audit-config.json'}.items()}
    write(Path(prior['driver']),{'pid':123,'fixture':True})
    write(Path(prior['config']),deployment)
    write(Path(prior['terminal']),dict(exit_code=0,status='train_parity_collected_and_audited'))
    write(Path(prior['audit_config']),audit_config)
    prior.update({key+'_sha256':module.file_hash(prior[key]) for key in ('driver','config')})
    return dict(predecessor=prior,parity_audit=audit_config,host='fixture',lease='fixture-lease')


@pytest.mark.parametrize('fault',[None,'path','driver','config','deployment','host','lease','failure','audit'])
def test_successor_uses_exact_pinned_predecessor(tmp_path,monkeypatch,fault):
    c=predecessor(tmp_path);p=c['predecessor'];seen=[]
    monkeypatch.setattr(module,'wait_exact',lambda record,terminal:seen.append((record,terminal)))
    if fault=='path':p['terminal']=str(tmp_path/'other.json')
    elif fault in ('driver','config'):write(Path(p[fault]),{'changed':True})
    elif fault=='deployment':c['parity_audit']['deployment']['different']=True
    elif fault in ('host','lease'):c[fault]='other'
    elif fault=='failure':write(Path(p['terminal']),dict(exit_code=1))
    elif fault=='audit':write(Path(p['audit_config']),{'wrong':True})
    if fault:
        with pytest.raises(ValueError):module.wait_for_parity(c)
        if fault not in ('failure','audit'):assert seen==[]
    else:
        module.wait_for_parity(c)
        assert seen==[({'pid':123,'fixture':True},p['terminal'])]


@pytest.mark.parametrize('failure',[None,'predecessor','launch','collection','audit'])
def test_reference_driver_never_turns_collection_into_admission(tmp_path,monkeypatch,failure):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    calls=[]
    def wait(c):
        calls.append('wait')
        if failure=='predecessor':raise ValueError('predecessor failed')
    def launch(c,p):
        calls.append('launch')
        if failure=='launch':raise ValueError('launch failed')
        return dict(status='bad' if failure=='collection' else 'collected_independent_reference_output_audit_required')
    def audit(c,p):
        calls.append('audit')
        assert c['deployment']['tag']=='reference'
        assert c['archive']==str(tmp_path/'operations/reference-collected.tar')
        if failure=='audit':raise ValueError('output failed')
        return dict(output_audit={'fixture':'independent-output'},training_dispatch_allowed=False)
    monkeypatch.setattr(module,'wait_for_parity',wait)
    monkeypatch.setattr(module,'launch',launch)
    monkeypatch.setattr(module,'audit',audit)
    c=dict(operations_root=str(tmp_path/'operations'),reference_dispatch_allowed=True,
        tag='reference',reference_job={'fixture':True},parity_audit={'fixture':True})
    if failure:
        with pytest.raises(ValueError):module.run_and_audit(c,{})
    else:module.run_and_audit(c,{})
    root=Path(c['operations_root'])/'finalization';t=module.read(root/'terminal.json')
    assert t['exit_code']==int(failure is not None)
    assert (root/'audit.json').exists()==(failure is None)
    if failure:
        assert t['preserve_handles'] and t['restart'] is False
    else:
        assert calls==['wait','launch','audit']
        assert t['training_dispatched'] is False
        assert module.read(root/'reference-output-audit.json')=={'fixture':'independent-output'}
    if failure=='predecessor':assert calls==['wait']
    if failure=='collection':assert calls==['wait','launch']
    with pytest.raises(FileExistsError):module.run_and_audit(c,{})


def test_reference_driver_requires_reference_authorization(tmp_path):
    with pytest.raises(ValueError,match='explicit full TRAIN'):
        module.run_and_audit(dict(operations_root=str(tmp_path/'unused'),finite_inference_dispatch_allowed=True),{})
    assert not (tmp_path/'unused').exists()
