import json
from pathlib import Path

import pytest

from tools.reports import run_context_transfer_parity_sequence as m


def fixture(tmp_path,monkeypatch,failure=None):
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    tail_ops=save('tail.json',dict(host='npu',lease='lease'))
    c=dict(output=str(tmp_path/'sequence'),predecessor_driver=save('driver.json',dict(pid=123)),
        predecessor_config=save('prior.json',dict(output=str(tmp_path),operations=tail_ops)),
        predecessor_terminal=save('terminal.json',dict(exit_code=int(failure=='prior'),status='tail_diagnostic_collected_audited')))
    c['operations']=[save(str(i)+'.json',dict(dataset=d,host='npu',lease='lease',remote_root='/data/'+str(i),
        operations_root=str(tmp_path/str(i)),tag=str(i),finite_inference_dispatch_allowed=True)) for i,d in enumerate(m.DATASETS)]
    c['pins']={p:m.file_hash(p) for p in [tail_ops,c['predecessor_driver'],c['predecessor_config'],*c['operations']]}
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(m.tail,'validate',lambda _:dict(parent_manifest_sha256='parent'))
    monkeypatch.setattr(m.parity,'validate',lambda o:dict(dataset=o['dataset'],parent_manifest_sha256='parent',
        policy_id='p',runtime_identity={},image='image'))
    calls=[]
    def wait(*args):
        calls.append('wait')
        if failure=='pin':Path(c['operations'][0]).write_text('{}')
    monkeypatch.setattr(m,'wait_driver',wait)
    def audit(*args):
        calls.append('tail-audit')
        if failure=='tail':raise ValueError('tail failed')
        return {'status':'finite_dp2_tail_outputs_audited'}
    monkeypatch.setattr(m.tail,'audit_collected',audit)
    def run(o,*args,**kwargs):
        calls.append('run:'+o['dataset'])
        if failure=='run' and o['dataset']==m.DATASETS[1]:raise ValueError('uncertain execution')
    monkeypatch.setattr(m.parity,'lifecycle',run)
    def collect(o,**kwargs):
        calls.append('collect:'+o['dataset'])
        if failure=='collect':raise ValueError('audit failed')
        return dict(status='target_width_macro_parity_outputs_audited',comparison=dict(dataset=o['dataset'],exact=True))
    monkeypatch.setattr(m,'collect_and_audit',collect)
    return c,calls


@pytest.mark.parametrize('failure',[None,'prior','pin','tail','run','collect'])
def test_serial_exact_wait_and_fail_fast_without_retry(tmp_path,monkeypatch,failure):
    c,calls=fixture(tmp_path,monkeypatch,failure)
    if failure:
        with pytest.raises(ValueError):m.run(c,{})
    else:m.run(c,{})
    expected=['wait']
    if failure not in ('prior','pin'):expected.append('tail-audit')
    if failure not in ('prior','pin','tail'):
        for d in m.DATASETS:
            expected.append('run:'+d)
            if failure=='run' and d==m.DATASETS[1]:break
            expected.append('collect:'+d)
            if failure=='collect':break
    assert calls==expected
    terminal=m.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None)
    if not failure:assert terminal['target_scored'] is False


def test_existing_attempt_refused_before_wait(tmp_path,monkeypatch):
    c,calls=fixture(tmp_path,monkeypatch);p=tmp_path/'1';p.mkdir();(p/'attempt.json').write_text('{}')
    with pytest.raises(ValueError,match='fresh serial'):m.run(c,{})
    assert not calls


def test_detach_is_single_attempt(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c,_=fixture(tmp_path,monkeypatch);monkeypatch.setattr(m,'load_config',lambda *args:(c,{}))
    spawned=[]
    def spawn(argv,**kw):
        assert (tmp_path/'driver/intent.json').exists() and kw['start_new_session']
        spawned.append(argv);return SimpleNamespace(pid=321)
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    assert m.detach('c','l',tmp_path/'driver')['pid']==321
    with pytest.raises(FileExistsError):m.detach('c','l',tmp_path/'driver')
    assert len(spawned)==1
