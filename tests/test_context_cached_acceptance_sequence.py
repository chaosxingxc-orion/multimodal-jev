import copy
from pathlib import Path

import pytest

from tools.reports import context_cached_acceptance_sequence as module
from test_context_reference_driver import write


def setup(tmp_path,monkeypatch):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    base=tmp_path/'reference'
    ops=dict(operations_root=str(base),package=str(tmp_path/'reference-package'),
        reference_job=dict(remote_root='/data/reference',parent=dict(output=str(tmp_path/'bound'),remote_root='/data/bound')),
        reference_dispatch_allowed=True,parity_audit={'fixture':True},host='fixture',lease='lease',
        shared_root='/shared',driver_root='/driver',
        release_evidence={'root':'/shared/evidence','destination':'/original'})
    prior={key:str(base/path) for key,path in {
        'driver':'driver/started.json','config':'finalization/config.json',
        'terminal':'finalization/terminal.json','audit_config':'finalization/audit-config.json'}.items()}
    write(Path(prior['config']),ops);write(Path(prior['driver']),{'pid':123,'fixture':True})
    prior.update({key+'_sha256':module.file_hash(prior[key]) for key in ('driver','config')})
    bound=dict(deployment=ops,reference_job=ops['reference_job'],parity_audit=ops['parity_audit'])
    write(Path(prior['audit_config']),bound)
    write(Path(prior['terminal']),dict(exit_code=0,status='reference_collected_and_audited'))
    c=dict(output=str(tmp_path/'sequence'),reference_predecessor=prior,finite_training_dispatch_allowed=True,
        stages={n:dict(package=str(tmp_path/(n+'-package')),remote_root='/data/'+n,
            operations_root=str(tmp_path/(n+'-operations')),tag='finite-'+n) for n in module.STAGES})
    return c,ops,bound


@pytest.mark.parametrize('fault',[None,'authorization','stages','keys','local-overlap','remote-overlap',
    'protected','tag','pin','path','terminal','deployment','audit'])
def test_sequence_admission_and_exact_reference_predecessor(tmp_path,monkeypatch,fault):
    c,ops,bound=setup(tmp_path,monkeypatch);p=c['reference_predecessor'];seen=[]
    monkeypatch.setattr(module,'wait_exact',lambda *a:seen.append('wait'))
    def audit(*a):
        seen.append('audit')
        if fault=='audit':raise ValueError('independent reference failed')
        return dict(status='reference_job_and_same_endpoint_parity_audited')
    monkeypatch.setattr(module.reference,'audit',audit)
    if fault=='authorization':c['finite_training_dispatch_allowed']=False
    elif fault=='stages':del c['stages']['tail']
    elif fault=='keys':c['stages']['prefix']['stop_update']=100
    elif fault=='local-overlap':c['stages']['tail']['package']=c['stages']['prefix']['operations_root']+'/child'
    elif fault=='remote-overlap':c['stages']['tail']['remote_root']='/data/prefix/child'
    elif fault=='protected':c['stages']['prefix']['package']=ops['package']
    elif fault=='tag':c['stages']['tail']['tag']='not safe'
    elif fault=='pin':p['driver_sha256']='wrong'
    elif fault=='path':p['terminal']=str(tmp_path/'wrong')
    elif fault=='terminal':write(Path(p['terminal']),dict(exit_code=1))
    elif fault=='deployment':
        bound['deployment']['host']='other';write(Path(p['audit_config']),bound)
    if fault:
        with pytest.raises(ValueError):module.wait_reference(c)
        if fault not in ('terminal','deployment','audit'):assert seen==[]
        elif fault!='audit':assert seen==['wait']
    else:
        actual=module.wait_reference(c)
        assert actual[:2]==(ops,bound) and seen==['wait','audit']


@pytest.mark.parametrize('stage',module.STAGES)
@pytest.mark.parametrize('failure',[None,'prepare','launch','collect','audit','status'])
def test_stage_only_advances_after_independent_matching_audit(tmp_path,monkeypatch,stage,failure):
    c,ops,ref=setup(tmp_path,monkeypatch);calls=[];prefix={'fixture':'audited-prefix'}
    prep=module.prepare_recovery if stage=='recovery' else module.prepare_training
    runtime=module.recovery if stage=='recovery' else module.training
    monkeypatch.setattr(module,'producer_spec',lambda spec:{'fixture':'reference-producer'})
    def prepare(spec,p):
        calls.append('prepare')
        if stage=='recovery':assert spec['parent_audit']==prefix
        else:assert spec['acceptance_kind']==('engineering_prefix' if stage=='prefix' else 'cached_reference_tail_admission')
        if failure=='prepare':raise ValueError('prepare failed')
        return dict(manifest_sha256='fixture')
    def launch(config,p):
        calls.append('launch')
        assert config['finite_training_dispatch_allowed'] is True and config['host']==ops['host']
        assert config['release_evidence']==ops['release_evidence']
        if failure=='launch':raise ValueError('launch failed')
        return dict(status='wrong' if failure=='collect' else 'collected_independent_cached_training_audit_required')
    def audit(config,p):
        calls.append('audit')
        assert config['deployment']['training_package']==config['training_package']
        assert config['reference_audit']==ref
        if failure=='audit':raise ValueError('audit failed')
        return dict(status='wrong' if failure=='status' else module.STATUSES[stage],formal_training_admitted=False)
    monkeypatch.setattr(prep,'prepare',prepare);monkeypatch.setattr(runtime,'launch',launch);monkeypatch.setattr(runtime,'audit',audit)
    args=(stage,c['stages'][stage],ops,ref,prefix,{})
    if failure:
        with pytest.raises(ValueError):module.run_stage(*args)
    else:module.run_stage(*args)
    root=Path(c['stages'][stage]['operations_root'])/'finalization'
    terminal=module.read(root/'terminal.json')
    assert terminal['exit_code']==int(failure is not None)
    assert (root/'audit.json').exists()==(failure is None)
    if failure=='prepare':assert calls==['prepare']
    elif failure in ('launch','collect'):assert calls==['prepare','launch']
    else:assert calls==['prepare','launch','audit']
    if failure:assert terminal['restart'] is False and terminal['preserve_handles']
    with pytest.raises(FileExistsError):module.run_stage(*args)


@pytest.mark.parametrize('failure',[None,'prefix','tail','recovery','reference'])
def test_sequence_stops_on_first_failed_gate_without_retry(tmp_path,monkeypatch,failure):
    c,ops,ref=setup(tmp_path,monkeypatch);calls=[]
    def wait(c):
        if failure=='reference':raise ValueError('reference failed')
        return ops,ref,{'fixture':True}
    def stage(name,stage,ops,ref,prefix,p):
        calls.append(name)
        assert prefix== (None if name=='prefix' else {'source':'prefix-audit'})
        if failure==name:raise ValueError('finite gate failed')
        return {'source':name+'-audit'},{'fixture':name}
    monkeypatch.setattr(module,'wait_reference',wait);monkeypatch.setattr(module,'run_stage',stage)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    expected=[] if failure=='reference' else list(module.STAGES)
    if failure in module.STAGES:expected=expected[:expected.index(failure)+1]
    assert calls==expected
    terminal=module.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None)
    if not failure:assert terminal['efficiency_and_effects_pending'] and not terminal['formal_training_dispatched']
