import json
from pathlib import Path

import pytest

from tools.reports import run_context_formal_successor as m


def fixture(tmp_path,monkeypatch,fault=None):
    def save(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)
    policy=dict(policy_id='p',contract={},base_id='base')
    parity=[]
    for i,dataset in enumerate(m.DATASETS):
        p=tmp_path/f'parity-{i}'
        save(f'parity-{i}/package/evaluation.json',dict(policy,dataset=dataset))
        parity.append(save(f'parity-{i}.json',dict(host='npu',lease='lease',prepared=str(p),dataset=dataset)))
    preparation=dict(output=str(tmp_path/'prepared'),operations_root=str(tmp_path/'training'),
        comparison_config=save('comparison-template.json',{}),comparison_local=save('comparison-local.json',{}))
    config=dict(output=str(tmp_path/'successor'),formal_training_dispatch_allowed=True,
        predecessor_config=save('parity-config.json',dict(output=str(tmp_path/'prior'),operations=parity)),
        predecessor_driver=save('driver.json',dict(pid=123)),predecessor_terminal=str(tmp_path/'prior/terminal.json'),
        preparation_config=save('preparation-template.json',{}),preparation_local=save('preparation-local.json',{}),
        tail_operations=save('tail.json',dict(host='npu',lease='lease')))
    save('prior/terminal.json',dict(exit_code=int(fault=='prior'),status='three_target_width_parity_audited',completed=list(m.DATASETS)))
    names=[config[k] for k in ('predecessor_config','predecessor_driver','preparation_config','preparation_local','tail_operations')]
    names+=parity+[preparation['comparison_config'],preparation['comparison_local']]
    config['pins']={n:m.file_hash(n) for n in names}
    monkeypatch.setattr(m,'load_config',lambda *a:(preparation,{}))
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    calls=[]
    def wait(*a):
        calls.append('wait')
        if fault=='changed_pin':Path(config['tail_operations']).write_text('{}')
    monkeypatch.setattr(m,'wait_driver',wait)
    def audit(o,**kw):
        calls.append('audit:'+o['dataset'])
        if fault=='parity':raise ValueError('failed parity')
        return dict(status='target_width_macro_parity_outputs_audited',comparison=dict(dataset=o['dataset'],exact=True))
    monkeypatch.setattr(m,'reaudit',audit)
    def prepare(c,p):
        calls.append('prepare')
        if fault=='prepare':raise ValueError('failed prepare')
        save('prepared/operations.json',dict(prepared=c['output'],operations_root=c['operations_root']))
        save('prepared/package/training.json',dict(source_policy_id='wrong' if fault=='policy' else 'p',source_contract={},base_id='base'))
        return {'status':'prepared'}
    monkeypatch.setattr(m,'prepare',prepare)
    def launch(*a):
        calls.append('launch')
        if fault=='launch':raise ValueError('failed launch')
    monkeypatch.setattr(m,'launch',launch)
    def collect(*a,**kw):
        calls.append('collect')
        if fault=='collect':raise ValueError('failed collect')
        return dict(status='formal_dp2_outputs_audited',updates=list(range(634)))
    monkeypatch.setattr(m,'collect_and_audit',collect)
    def build(*a):
        calls.append('plan')
        if fault=='plan':raise ValueError('failed plan')
        return {'actual':True}
    monkeypatch.setattr(m,'build',build)
    return config,calls


@pytest.mark.parametrize('fault',[None,'prior','changed_pin','parity','prepare','policy','launch','collect','plan'])
def test_exact_serial_sequence_failure_stops_without_retry(tmp_path,monkeypatch,fault):
    c,calls=fixture(tmp_path,monkeypatch,fault)
    if fault:
        with pytest.raises(ValueError):m.run(c,{})
    else:m.run(c,{})
    expected=['wait']
    if fault not in ('prior','changed_pin'):
        expected+=['audit:'+d for d in (m.DATASETS[:1] if fault=='parity' else m.DATASETS)]
        if fault!='parity':expected+=['prepare']
        if fault not in ('parity','prepare','policy'):expected+=['launch']
        if fault not in ('parity','prepare','policy','launch'):expected+=['collect']
        if fault in (None,'plan'):expected+=['plan']
    assert calls==expected
    terminal=m.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(fault is not None)
    if fault:assert terminal['restart'] is False
    else:assert terminal['source_evaluated'] is terminal['target_evaluated'] is False


@pytest.mark.parametrize('existing',['prepared','training/attempt.json'])
def test_existing_attempt_prevents_wait_or_launch(tmp_path,monkeypatch,existing):
    c,calls=fixture(tmp_path,monkeypatch)
    p=tmp_path/existing
    if p.suffix:p.parent.mkdir();p.write_text('{}')
    else:p.mkdir()
    with pytest.raises(ValueError):m.run(c,{})
    assert not calls


def test_detach_single_child_and_immutable_driver_directory(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'load_config',lambda *a:(c,{}))
    monkeypatch.setattr(m,'validate',lambda c:None)
    calls=[]
    def spawn(argv,**kw):
        assert (tmp_path/'driver/intent.json').is_file() and kw['start_new_session']
        calls.append(argv);return SimpleNamespace(pid=42)
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    assert m.detach('template','local',tmp_path/'driver')['pid']==42
    with pytest.raises(FileExistsError):m.detach('template','local',tmp_path/'driver')
    assert len(calls)==1
