"""Formal scheduling with explicit NPU/curve substitutes and real path gates."""
import copy
from pathlib import Path

import pytest

from multimodal_jev import local_checkpoint
from tools.reports import context_cached_formal_successor as module
from test_context_replica_selection_sequence import setup as selection_setup
from test_prepare_context_cached_training import write


def setup(tmp_path,monkeypatch):
    previous,eff=selection_setup(tmp_path,monkeypatch);base=Path(previous['output'])
    write(base/'config.json',previous)
    driver=base.with_name(base.name+'-driver')/'started.json';write(driver,{'pid':123,'fixture':True})
    write(base/'terminal.json',dict(exit_code=0,status='replica_infonce_curve_collected_and_audited',points=11,training_dispatched=False))
    entries=[]
    for p in previous['points']:
        path=Path(p['operations_root'])/'finalization/audit-config.json';write(path,{'fixture':p['step']})
        entries.append(dict(step=p['step'],audit_config=str(path),audit_config_sha256=module.file_hash(path)))
    cc=dict(previous['point_template'],evaluations=entries,output=str(base/'curve'));write(base/'curve-config.json',cc)
    curve=dict(complete=True,selected_checkpoint={'step':64,'policy_id':'selected'})
    endpoint={'step':318,'policy_id':'endpoint'}
    write(base/'curve/curve.json',curve);write(base/'curve/source-evaluation.json',{'roles':{'stage2_initial_and_reference':endpoint}})
    bound=module.read(Path(eff['output'])/'audit-config.json')
    bound['training_package'].update(parent=previous['point_template']['parent'],reference_job={'fixture':'fixed-reference'})
    bound['reference_audit']={'fixture':'reference-provenance'}
    write(Path(eff['output'])/'audit-config.json',bound)
    c=dict(output=str(tmp_path/'formal-successor'),formal_training_dispatch_allowed=True,
        predecessor=dict(config=str(base/'config.json'),driver=str(driver),terminal=str(base/'terminal.json')),
        experiment=dict(package=str(tmp_path/'formal-package'),remote_root='/data/formal',
            operations_root=str(tmp_path/'formal-ops'),tag='formal-test'))
    for k in ('config','driver'):c['predecessor'][k+'_sha256']=module.file_hash(c['predecessor'][k])
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'wait_exact',lambda *a:None)
    monkeypatch.setattr(module.curve,'reconstruct',lambda c:dict(plan={'reference_endpoint':endpoint},curve=copy.deepcopy(curve)))
    return c,previous,eff,bound


@pytest.mark.parametrize('fault',[None,'permission','pin','path','terminal','curve-config','curve','role',
    'parent','overlap','remote','reference-overlap','tag','source-audit','efficiency-audit','mutation'])
def test_exact_curve_and_efficiency_must_precede_same_endpoint_training(tmp_path,monkeypatch,fault):
    c,previous,eff,bound=setup(tmp_path,monkeypatch);base=Path(previous['output'])
    if fault=='permission':c['formal_training_dispatch_allowed']=False
    elif fault=='pin':c['predecessor']['driver_sha256']='wrong'
    elif fault=='path':c['predecessor']['terminal']='/other'
    elif fault=='terminal':write(base/'terminal.json',{'exit_code':1})
    elif fault=='curve-config':write(base/'curve-config.json',{})
    elif fault=='curve':write(base/'curve/curve.json',{})
    elif fault=='role':write(base/'curve/source-evaluation.json',{'roles':{'stage2_initial_and_reference':{'step':64}}})
    elif fault=='parent':
        bound['training_package']['parent']={};write(Path(eff['output'])/'audit-config.json',bound)
    elif fault=='overlap':c['experiment']['package']=previous['points'][0]['operations_root']+'/child'
    elif fault=='remote':c['experiment']['remote_root']=previous['points'][0]['remote_root']
    elif fault=='reference-overlap':
        finite=module.selection.efficiency_sequence.validate(eff)
        c['experiment']['remote_root']=module.read(finite['reference_predecessor']['config'])['reference_job']['remote_root']
    elif fault=='tag':c['experiment']['tag']='unsafe tag'
    elif fault in ('source-audit','efficiency-audit'):
        def fail(*a):raise ValueError('independent evidence failed')
        monkeypatch.setattr(module.curve if fault=='source-audit' else module.selection,
            'reconstruct' if fault=='source-audit' else 'wait_efficiency',fail)
    elif fault=='mutation':monkeypatch.setattr(module,'wait_exact',lambda *a:write(base/'config.json',{}))
    if fault:
        with pytest.raises(ValueError):module.wait_source(c)
    else:
        spec,actual,source=module.wait_source(c)
        assert spec['parent']==previous['point_template']['parent']
        assert spec['evaluation_steps']==module.formal.STEPS and spec['acceptance_kind']=='formal_source'
        assert source['reference_endpoint']['step']==318 and source['selected_checkpoint']['step']==64
        assert source['source_selection_used_for_initialization'] is False and actual==bound


@pytest.mark.parametrize('failure',[None,'predecessor','prepare','run','partial'])
def test_one_formal_transaction_preserves_deployment_and_stops_on_failure(tmp_path,monkeypatch,failure):
    c,_,_,bound=setup(tmp_path,monkeypatch);calls=[]
    actual_wait=module.wait_source
    def wait(c):
        calls.append('wait')
        if failure=='predecessor':raise ValueError('prior failure')
        return actual_wait(c)
    def prepare(spec,p):
        calls.append('prepare')
        assert spec['acceptance_kind']=='formal_source' and spec['evaluation_steps']==module.formal.STEPS
        if failure=='prepare':raise ValueError('gate failed')
        return {'manifest_sha256':'new-formal'}
    def run(ops,p):
        calls.append('run')
        assert module.inherited(ops)==module.inherited(bound['deployment'])
        assert ops['reference_audit']==bound['reference_audit']
        assert ops['formal_training_dispatch_allowed'] and ops['training_package']['efficiency_audit']==bound
        if failure=='run':raise ValueError('actual training failure')
        return dict(status='formal_cached_reference_training_outputs_audited',real_query_exposures=81026,
            updates=[{}]*(633 if failure=='partial' else 634))
    monkeypatch.setattr(module,'wait_source',wait);monkeypatch.setattr(module.formal,'package_action',prepare)
    monkeypatch.setattr(module.formal,'run',run)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    assert calls==(['wait'] if failure=='predecessor' else ['wait','prepare'] if failure=='prepare' else ['wait','prepare','run'])
    terminal=module.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None) and terminal['restart'] is False
    if not failure:assert terminal['selection_evaluation_pending'] and terminal['transfer_evaluation_pending']
    with pytest.raises(FileExistsError):module.run(c,{})


@pytest.mark.parametrize('failure',[False,True])
def test_monitor_waits_900_seconds_and_never_restarts_training(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    waits=[];calls=[]
    class Event:
        def wait(self,seconds):waits.append(seconds);return len(waits)>1
        def set(self):calls.append('stop')
    class Thread:
        def __init__(self,**kw):self.target=kw['target']
        def start(self):self.target()
        def join(self,timeout):assert timeout==60
        def is_alive(self):return False
    monkeypatch.setattr(module.threading,'Event',Event);monkeypatch.setattr(module.threading,'Thread',Thread)
    def observe(c):
        calls.append('observe');assert c['interval_seconds']==900
        return {'status':'not_due_no_remote_call'}
    def run(*a):
        calls.append('train')
        if failure:raise ValueError('actual training failed')
        return {'fixture':'audited'}
    monkeypatch.setattr(module,'observe',observe);monkeypatch.setattr(module.formal,'run',run)
    ops=dict(host='host',tag='tag',operations_root='/data/ops',manifest_sha256='manifest')
    if failure:
        with pytest.raises(ValueError):module.monitored_training(ops,{},tmp_path)
    else:assert module.monitored_training(ops,{},tmp_path)=={'fixture':'audited'}
    assert waits==[900,900] and calls==['observe','train','stop']
    assert module.read(tmp_path/'monitor-terminal.json')['training_success_verified'] is False
