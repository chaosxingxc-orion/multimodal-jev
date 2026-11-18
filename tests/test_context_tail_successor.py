import json
from pathlib import Path

import pytest

from tools.reports import run_context_tail_successor as m


def fixture(tmp_path,monkeypatch,failure=None):
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    ops=dict(operations_root=str(tmp_path/'tail'),host='npu',lease='lease',
        tail_diagnostic_dispatch_allowed=True,finite_training_dispatch_allowed=False)
    parent=dict(manifest_sha256='parent',host='npu',lease='lease')
    comparison=dict(schema='jev-service-batch-efficiency-comparison-v1',matched_actual_queries=True,
        matched_reference_misses=True,candidate=dict(optimizer_steps=100,all_updates={'real_query_exposures':12800}),
        source_audits=[{},dict(status='finite_dp2_efficiency_audited',files_sha256='parent')])
    if failure=='short_run':comparison['candidate']['optimizer_steps']=99
    if failure=='wrong_parent':comparison['source_audits'][1]['files_sha256']='other'
    if failure=='cache':comparison['matched_reference_misses']=False
    c=dict(output=str(tmp_path/'successor'),operations=save('ops.json',ops),
        predecessor_driver=save('driver.json',{'pid':123}),
        predecessor_terminal=save('terminal.json',dict(exit_code=int(failure=='predecessor'),
            status='service_batch32_100_steps_compared')),
        comparison=save('comparison.json',comparison),
        study_config=save('study.json',dict(output=str(tmp_path),training_operations=save('parent.json',parent))))
    c['pins']={c[k]:m.file_hash(c[k]) for k in ['operations','predecessor_driver','study_config']}
    calls=[]
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(m.tail,'validate',lambda _:dict(parent_manifest_sha256='parent'))
    def wait(*a):
        calls.append('wait')
        if failure=='changed_pin':Path(c['operations']).write_text('{}')
    monkeypatch.setattr(m,'wait_driver',wait)
    def train(*a):
        calls.append('train')
        if failure=='training':raise ValueError('fixture failure')
    monkeypatch.setattr(m.tail,'run',train)
    def collect(*a,**kw):
        calls.append('collect')
        if failure=='audit':raise ValueError('fixture audit failure')
        return {'status':'finite_dp2_tail_outputs_audited'}
    monkeypatch.setattr(m,'collect_and_audit',collect)
    return c,calls


@pytest.mark.parametrize('failure',[None,'predecessor','short_run','wrong_parent','cache','changed_pin','training','audit'])
def test_exact_wait_one_attempt_and_no_formal_launch(tmp_path,monkeypatch,failure):
    c,calls=fixture(tmp_path,monkeypatch,failure)
    if failure:
        with pytest.raises(ValueError):m.run(c,{})
    else:m.run(c,{})
    expected=['wait']
    if failure in (None,'training','audit'):expected+=['train']
    if failure in (None,'audit'):expected+=['collect']
    assert calls==expected
    terminal=m.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None)
    if not failure:assert terminal['formal_training_dispatched'] is False
    else:assert terminal['restart'] is False


def test_existing_attempt_refused_without_wait_or_remote_call(tmp_path,monkeypatch):
    c,calls=fixture(tmp_path,monkeypatch)
    p=tmp_path/'tail';p.mkdir();(p/'attempt.json').write_text('{}')
    with pytest.raises(ValueError,match='already attempted'):m.run(c,{})
    assert not calls and not Path(c['output']).exists()


def test_detach_records_intent_and_never_reuses_driver_root(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'load_config',lambda *a:(c,{}))
    commands=[]
    def spawn(argv,**kw):
        assert (tmp_path/'driver/intent.json').exists()
        assert kw['start_new_session'] is True
        commands.append(argv);return SimpleNamespace(pid=12345)
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    result=m.detach('template.json','local.json',tmp_path/'driver')
    assert result['pid']==12345 and len(commands)==1
    with pytest.raises(FileExistsError):m.detach('template.json','local.json',tmp_path/'driver')
    assert len(commands)==1
