"""Real scheduling gates with explicitly substituted training/inference."""
import copy
from pathlib import Path

import pytest

from multimodal_jev import local_checkpoint
from tools.reports import context_cached_grpo_evaluation_sequence as module
from test_context_cached_formal_successor import setup as formal_setup
from test_prepare_context_cached_training import write


def setup(tmp_path,monkeypatch):
    previous,selection,eff,bound=formal_setup(tmp_path,monkeypatch);base=Path(previous['output'])
    write(base/'config.json',previous);driver=base.with_name(base.name+'-driver')/'started.json'
    write(driver,{'pid':123,'fixture':True});write(base/'terminal.json',dict(exit_code=0,
        status='formal_cached_successor_collected_and_audited',updates=634))
    stage=previous['experiment'];spec=dict(parent=selection['point_template']['parent'],remote_root=stage['remote_root'])
    ops=dict(selection['deployment'],training_package=spec,**{k:stage[k] for k in ('package','operations_root','tag')})
    root=Path(stage['operations_root'])/'finalization';write(root/'audit-config.json',dict(deployment=ops,training_package=spec))
    write(root/'terminal.json',dict(exit_code=0,status='formal_cached_training_collected_and_audited'))
    c=dict(output=str(tmp_path/'grpo-evaluation'),selection_dispatch_allowed=True,
        predecessor=dict(config=str(base/'config.json'),driver=str(driver),terminal=str(base/'terminal.json')),
        points=[dict(step=s,package=str(tmp_path/f'grpo-{s}-package'),remote_root=f'/data/grpo-{s}',
            operations_root=str(tmp_path/f'grpo-{s}-ops'),tag=f'grpo-{s}') for s in module.formal.STEPS])
    for k in ('driver','config'):c['predecessor'][k+'_sha256']=module.file_hash(c['predecessor'][k])
    monkeypatch.setattr(module,'check_paths',lambda _:None);monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'wait_exact',lambda *a:None)
    monkeypatch.setattr(module.source,'build',lambda *a:{'fixture':'independently audited actual GRPO'})
    return c,previous,selection,ops


@pytest.mark.parametrize('fault',[None,'permission','pin','path','missing','order','overlap','remote','tag',
    'terminal','deployment','parent','collection','audit','mutation'])
def test_exact_formal_source_and_full_curve_gates(tmp_path,monkeypatch,fault):
    c,previous,selection,ops=setup(tmp_path,monkeypatch);base=Path(previous['output'])
    final=Path(previous['experiment']['operations_root'])/'finalization'
    if fault=='permission':c['selection_dispatch_allowed']=False
    elif fault=='pin':c['predecessor']['config_sha256']='wrong'
    elif fault=='path':c['predecessor']['terminal']='/wrong'
    elif fault=='missing':c['points'].pop()
    elif fault=='order':c['points'].reverse()
    elif fault=='overlap':c['points'][0]['package']=selection['points'][0]['operations_root']
    elif fault=='remote':c['points'][0]['remote_root']=previous['experiment']['remote_root']
    elif fault=='tag':c['points'][1]['tag']=c['points'][0]['tag']
    elif fault=='terminal':write(base/'terminal.json',{'exit_code':1})
    elif fault in ('deployment','parent'):
        v=module.read(final/'audit-config.json')
        if fault=='deployment':v['deployment']['lease']='other'
        else:
            v['training_package']['parent']={};v['deployment']['training_package']=v['training_package']
        write(final/'audit-config.json',v)
    elif fault=='collection':write(final/'terminal.json',{'status':'needs_collect'})
    elif fault=='audit':
        def reject(*a):raise ValueError('actual training invalid')
        monkeypatch.setattr(module.source,'build',reject)
    elif fault=='mutation':monkeypatch.setattr(module,'wait_exact',lambda *a:write(base/'config.json',{}))
    if fault:
        with pytest.raises(ValueError):module.wait_formal(c)
    else:
        spec,plan,actual,inputs=module.wait_formal(c)
        assert actual==ops and inputs==selection['point_template']['selection_inputs']
        assert module.file_hash(spec['training_audit_config'])==spec['training_audit_config_sha256']


@pytest.mark.parametrize('failure',[None,'predecessor','prepare','point','curve','status'])
def test_single_attempt_serial_full_curve_then_role_freeze(tmp_path,monkeypatch,failure):
    c,_,_,ops=setup(tmp_path,monkeypatch);calls=[];actual_wait=module.wait_formal
    def wait(c):
        calls.append('wait')
        if failure=='predecessor':raise ValueError('failed formal')
        return actual_wait(c)
    def prepare(spec,p):
        calls.append(('prepare',spec['source_step']))
        if failure=='prepare':raise ValueError('invalid origin')
        assert module.file_hash(spec['source_plan'])==spec['source_plan_sha256']
        return {'manifest_sha256':'fixture'}
    def run(deployment,p):
        step=deployment['selection_package']['source_step'];calls.append(('run',step))
        assert module.inherited(deployment)==module.inherited(ops)
        if failure=='point' and step==64:raise ValueError('failed parity')
        path=Path(deployment['operations_root'])/'finalization/audit-config.json';write(path,{'fixture':step})
        return dict(status='wrong' if failure=='status' else 'complete_replica_cached_grpo_selection_audited',
            source_step=step,runtime_provenance_verified=True)
    def collect(spec,p):
        calls.append('curve');assert [e['step'] for e in spec['evaluations']]==module.formal.STEPS
        if failure=='curve':raise ValueError('inconsistent curve')
        return {'status':'complete_source_curve_audited_and_roles_frozen'}
    monkeypatch.setattr(module,'wait_formal',wait);monkeypatch.setattr(module.preparation,'prepare',prepare)
    monkeypatch.setattr(module.point,'run',run);monkeypatch.setattr(module.curve,'collect',collect)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    terminal=module.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None) and terminal['restart'] is False
    runs=[v[1] for v in calls if isinstance(v,tuple) and v[0]=='run']
    if failure in ('predecessor','prepare'):assert runs==[]
    elif failure=='point':assert runs==[0,64]
    elif failure=='status':assert runs==[0]
    else:assert runs==module.formal.STEPS
    before=copy.deepcopy(calls)
    with pytest.raises(FileExistsError):module.run(c,{})
    assert calls==before
