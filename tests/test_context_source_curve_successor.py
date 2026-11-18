import json
from pathlib import Path

import pytest

from tools.reports import run_context_source_curve_successor as m


def fixture(tmp_path,monkeypatch,fault=None):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return str(p)
    operations=dict(prepared=str(tmp_path/'training-package'),manifest_sha256='formal',
        operations_root=str(tmp_path/'training-ops'),remote_root='/data/train',host='npu',
        model='/shared/model',driver_root='/driver',lease='/local/lease')
    save('training-package/package/training.json',{'release':'/shared/release'})
    prior={'output':str(tmp_path/'formal-successor')}
    save('formal-successor/launch-config.json',{'operations':operations})
    terminal=save('formal-successor/terminal.json',dict(exit_code=int(fault=='prior'),
        status='formal_source_collected_and_plan_bound',updates=634))
    plan=dict(source_operations=operations,experiment={'release_sha256':'release'},
        checkpoints=[dict(step=s,policy_id='p'+str(s)) for s in m.STEPS])
    plan_path=save('formal-successor/source-evaluation-plan.json',plan)
    if fault=='plan':save('formal-successor/source-evaluation-plan.json',dict(plan,extra=True))
    c=dict(output=str(tmp_path/'successor'),prepared_root=str(tmp_path/'prepared'),
        operations_root=str(tmp_path/'point-ops'),curve_output=str(tmp_path/'curve'),remote_root='/data/curve',
        tag_prefix='source',source_evaluation_dispatch_allowed=True,
        predecessor_driver=save('driver.json',{'pid':123}),predecessor_config=save('prior.json',prior),
        predecessor_terminal=terminal,source_plan=plan_path,
        operations_template=save('operations-template.json','${SOURCE_POINT_OPERATIONS}'),
        inputs={k:'fixture' for k in m.INPUT_KEYS})
    c['inputs'].update(release='/shared/release',release_sha256='release')
    c['pins']={c[k]:m.file_hash(c[k]) for k in ('predecessor_driver','predecessor_config','operations_template')}
    calls=[]
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    def wait(*a):
        calls.append('wait')
        if fault=='pins':Path(c['operations_template']).write_text('{}')
    monkeypatch.setattr(m,'wait_driver',wait)
    def audit(*a):
        calls.append('training-audit')
        if fault=='training':raise ValueError('training failure')
        return {'status':'formal_dp2_outputs_audited'}
    monkeypatch.setattr(m,'audit_training',audit)
    monkeypatch.setattr(m,'build',lambda *a:plan)
    def prepare(c,p):
        calls.append(('prepare',c['source_step']))
        if fault=='prepare' and c['source_step']==64:raise ValueError('prepare failure')
        return {'files_sha256':'point'}
    monkeypatch.setattr(m,'prepare',prepare)
    monkeypatch.setattr(m.point,'validate',lambda _:None)
    def run(c,p,**kw):
        step=int(c['tag'].split('-')[-1]);calls.append(('run',step))
        if fault=='run' and step==64:raise ValueError('run failure')
    monkeypatch.setattr(m.point.shared,'lifecycle',run)
    def collect(c,**kw):
        step=int(c['tag'].split('-')[-1]);calls.append(('collect',step))
        if fault=='collect' and step==64:raise ValueError('collect failure')
        return dict(status='complete_source_point_macro_selection_audited',result={'policy_id':'p'+str(step)})
    monkeypatch.setattr(m,'collect_and_audit',collect)
    def curve(c,p):
        calls.append('curve')
        assert [e['step'] for e in c['evaluations']]==m.STEPS
        for item in c['evaluations']:
            value=json.loads(Path(item['local']).read_text())['macros']['SOURCE_POINT_OPERATIONS']
            assert value['finite_inference_dispatch_allowed'] and value['lease']==operations['lease']
        if fault=='curve':raise ValueError('curve failed')
        return dict(status='complete_source_curve_audited_and_roles_frozen',source_evaluation={'path':'actual','sha256':'hash'})
    monkeypatch.setattr(m,'collect_curve',curve)
    return c,calls


def test_all11_points_full_curve_then_freeze_without_targets(tmp_path,monkeypatch):
    c,calls=fixture(tmp_path,monkeypatch);m.run(c,{})
    expected=['wait','training-audit']
    for step in m.STEPS:expected.extend([(k,step) for k in ('prepare','run','collect')])
    assert calls==expected+['curve']
    t=m.read(Path(c['output'])/'terminal.json')
    assert t['exit_code']==0 and t['target_evaluated'] is False


@pytest.mark.parametrize('fault',['prior','pins','training','plan','prepare','run','collect','curve'])
def test_failure_preserves_outputs_never_retries_or_selects_partial_curve(tmp_path,monkeypatch,fault):
    c,calls=fixture(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):m.run(c,{})
    t=m.read(Path(c['output'])/'terminal.json')
    assert t['exit_code']==1 and t['restart'] is False
    assert calls.count(('run',64))<=1
    if fault!='curve':assert 'curve' not in calls
    if fault in ('prepare','run','collect'):assert ('prepare',128) not in calls


def test_existing_output_or_disabled_dispatch_rejected_before_wait(tmp_path,monkeypatch):
    c,calls=fixture(tmp_path,monkeypatch)
    c['source_evaluation_dispatch_allowed']=False
    with pytest.raises(ValueError):m.run(c,{})
    c['source_evaluation_dispatch_allowed']=True
    Path(c['prepared_root']).mkdir()
    with pytest.raises(ValueError):m.run(c,{})
    assert not calls


def test_detach_is_one_fixed_child_and_records_intent(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'load_config',lambda *a:(c,{}))
    calls=[]
    def spawn(argv,**kw):
        assert (tmp_path/'driver/intent.json').is_file() and kw['start_new_session']
        calls.append(argv);return SimpleNamespace(pid=55)
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    assert m.detach('template','local',tmp_path/'driver')['pid']==55
    with pytest.raises(FileExistsError):m.detach('template','local',tmp_path/'driver')
    assert len(calls)==1
