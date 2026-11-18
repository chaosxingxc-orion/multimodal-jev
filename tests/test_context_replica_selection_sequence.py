"""Deterministic sequence tests; remote model execution explicitly substituted."""
import copy
from pathlib import Path

import pytest

from multimodal_jev import local_checkpoint
from tools.reports import context_replica_selection_sequence as module
from test_context_cached_efficiency_successor import setup as efficiency_setup
from test_prepare_context_cached_training import write


def setup(tmp_path,monkeypatch):
    eff,finite,evidence=efficiency_setup(tmp_path,monkeypatch)
    root=Path(eff['output']);write(root/'config.json',eff)
    driver=root.with_name(root.name+'-driver')/'started.json';write(driver,{'pid':123,'fixture':True})
    write(root/'terminal.json',dict(exit_code=0,status='cached_efficiency_collected_audited_and_measured',
        updates=100,formal_training_dispatched=False))
    refops=module.read(finite['reference_predecessor']['config']);deployment=module.inherited(refops)
    source=tmp_path/'source-plan.json';write(source,{'fixture':'source'})
    c=dict(output=str(tmp_path/'selection'),selection_dispatch_allowed=True,deployment=deployment,
        predecessor=dict(config=str(root/'config.json'),driver=str(driver),terminal=str(root/'terminal.json')),
        point_template=dict(parent=refops['reference_job']['parent'],source_plan=str(source),
            source_plan_sha256=module.file_hash(source),source_finalization={},selection_inputs={}),
        points=[dict(step=s,package=str(tmp_path/f'select-{s}-package'),remote_root=f'/data/select-{s}',
            operations_root=str(tmp_path/f'select-{s}-ops'),tag=f'select-{s}') for s in module.STEPS])
    for k in ('driver','config'):c['predecessor'][k+'_sha256']=module.file_hash(c['predecessor'][k])
    ops=dict(deployment,**eff['experiment']);bound=dict(deployment=ops,
        training_package={'remote_root':eff['experiment']['remote_root']})
    write(root/'audit-config.json',bound)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'wait_exact',lambda *a:None)
    monkeypatch.setattr(module.efficiency,'audit',lambda *a:dict(status='cached_100step_efficiency_outputs_audited'))
    return c,eff


@pytest.mark.parametrize('fault',[None,'authorization','missing','order','pin','path','lease','release_mount',
    'source','parent','overlap','remote','tag','terminal','audit'])
def test_fixed_predecessor_full_scope_and_isolation(tmp_path,monkeypatch,fault):
    c,eff=setup(tmp_path,monkeypatch)
    if fault=='authorization':c['selection_dispatch_allowed']=False
    elif fault=='missing':c['points'].pop()
    elif fault=='order':c['points'].reverse()
    elif fault=='pin':c['predecessor']['driver_sha256']='wrong'
    elif fault=='path':c['predecessor']['terminal']='/wrong'
    elif fault=='lease':c['deployment']['lease']='other'
    elif fault=='release_mount':c['deployment'].pop('release_evidence')
    elif fault=='source':c['point_template']['source_plan_sha256']='wrong'
    elif fault=='parent':c['point_template']['parent']={}
    elif fault=='overlap':c['points'][1]['package']=c['points'][0]['operations_root']+'/nested'
    elif fault=='remote':c['points'][1]['remote_root']=eff['experiment']['remote_root']
    elif fault=='tag':c['points'][1]['tag']=c['points'][0]['tag']
    elif fault=='terminal':write(Path(c['predecessor']['terminal']),{'exit_code':1})
    elif fault=='audit':
        def fail(*a):raise ValueError('independent efficiency audit failed')
        monkeypatch.setattr(module.efficiency,'audit',fail)
    if fault:
        with pytest.raises(ValueError):module.wait_efficiency(c)
    else:assert module.wait_efficiency(c)['status']=='cached_100step_efficiency_outputs_audited'


@pytest.mark.parametrize('failure',[None,'predecessor','prepare','point','curve','status'])
def test_serial_points_stop_on_failure_and_never_retry(tmp_path,monkeypatch,failure):
    c,_=setup(tmp_path,monkeypatch);calls=[]
    # Reuse one previously sealed package, independently audited on consumption.
    existing=c['points'][-1];write(Path(existing['package'])/'PREPARED.json',{'fixture':'endpoint'})
    existing['manifest_sha256']=module.file_hash(Path(existing['package'])/'PREPARED.json')
    def wait(c):
        calls.append('wait')
        if failure=='predecessor':raise ValueError('failed predecessor')
        return {'fixture':'independent audit'}
    def prepare(spec,p):
        calls.append(('prepare',spec['source_step']))
        if failure=='prepare':raise ValueError('invalid source')
        return {'manifest_sha256':'fixture'}
    def audit(spec,p):calls.append(('reuse',spec['source_step']))
    def run(ops,p):
        step=ops['selection_package']['source_step'];calls.append(('run',step))
        assert ops['release_evidence']==c['deployment']['release_evidence']
        if failure=='point' and step==32:raise ValueError('numeric failure')
        write(Path(ops['operations_root'])/'finalization/audit-config.json',{'fixture':step})
        return dict(status='bad' if failure=='status' else 'complete_replica_infonce_selection_audited',
            source_step=step,runtime_provenance_verified=True)
    def collect(spec,p):
        calls.append('curve')
        assert [e['step'] for e in spec['evaluations']]==module.STEPS
        for e in spec['evaluations']:assert module.file_hash(e['audit_config'])==e['audit_config_sha256']
        if failure=='curve':raise ValueError('incomparable points')
        return {'status':'complete_infonce_source_curve_audited'}
    monkeypatch.setattr(module,'wait_efficiency',wait)
    monkeypatch.setattr(module,'reconstruct_plan',lambda c:{'fixture':'actual-source-plan'})
    monkeypatch.setattr(module.preparation,'prepare',prepare);monkeypatch.setattr(module.preparation,'audit',audit)
    monkeypatch.setattr(module.point,'run',run);monkeypatch.setattr(module.curve,'collect',collect)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    terminal=module.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None) and terminal['restart'] is False
    runs=[v[1] for v in calls if isinstance(v,tuple) and v[0]=='run']
    if failure in ('predecessor','prepare'):assert runs==[]
    elif failure=='point':assert runs==[0,32]
    elif failure=='status':assert runs==[0]
    else:assert runs==module.STEPS and ('reuse',318) in calls
    before=copy.deepcopy(calls)
    with pytest.raises(FileExistsError):module.run(c,{})
    assert calls==before
