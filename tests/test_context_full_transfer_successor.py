import json
from pathlib import Path

import pytest

from tools.reports import run_context_full_transfer_successor as m


def fixture(tmp_path,monkeypatch,*,alias=False,fault=None):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return str(p)
    source={'path':'source','sha256':'source'}
    roles={role:dict(role=role,policy_id='p0' if alias and i==1 else f'p{i}') for i,role in enumerate(m.ROLES)}
    prior=dict(output=str(tmp_path/'prior'),curve_output=str(tmp_path/'source-curve'))
    save('prior/terminal.json',dict(exit_code=int(fault=='prior'),status='source_curve_completed_roles_frozen',source_evaluation=source))
    save('prior/curve-config.json',dict(output=prior['curve_output']))
    save('source-curve/result.json',dict(status='complete_source_curve_audited_and_roles_frozen',source_evaluation=source,roles=roles))
    c=dict(output=str(tmp_path/'successor'),prepared_root=str(tmp_path/'prepared'),operations_root=str(tmp_path/'ops'),
        scores_root=str(tmp_path/'scores'),remote_root='/data/transfer',tag_prefix='target',project='/project',
        full_transfer_dispatch_allowed=True,predecessor_driver=save('driver.json',{'pid':1}),
        predecessor_config=save('prior-config.json',prior),predecessor_terminal=str(tmp_path/'prior/terminal.json'),
        datasets={d:dict(portable_full_inputs={'dataset':d},portable_parity_inputs={},
            original_evaluation={'dataset':d},scoring={}) for d in m.DATASETS})
    c['pins']={c[k]:m.file_hash(c[k]) for k in ('predecessor_driver','predecessor_config')}
    calls=[]
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    def wait(*a):
        calls.append('wait')
        if fault=='pins':Path(c['predecessor_driver']).write_text('{}')
    monkeypatch.setattr(m,'wait_driver',wait)
    def admit(request):
        calls.append('admit:'+request['policy_role'])
        if fault=='source':raise ValueError('bad source')
        return dict(binding=roles[request['policy_role']],source_operations={},training={},source_evaluation=source)
    monkeypatch.setattr(m,'admit',admit)
    def prepare(p,provenance):
        calls.append('prepare')
        if fault=='prepare':raise ValueError('bad prepare')
        out=Path(p['output']);out.mkdir(parents=True)
        (out/'operations.json').write_text(json.dumps(dict(dataset=p['portable_full_inputs']['dataset'],
            policy_id=p['source_admission']['policy_id'],source_admission=p['source_admission'],prepared=p['output'])))
    monkeypatch.setattr(m,'prepare',prepare)
    monkeypatch.setattr(m.full,'validate',lambda _:None)
    def lifecycle(*a,**kw):
        calls.append('run')
        if fault=='run':raise ValueError('bad run')
    monkeypatch.setattr(m.full,'lifecycle',lifecycle)
    def collect(ops):
        calls.append('collect')
        if fault=='collect':raise ValueError('bad collect')
        return dict(status='full_portable_transfer_outputs_audited',runtime_provenance_verified=True,
                    policy_id=ops['policy_id'],dataset=ops['dataset'])
    monkeypatch.setattr(m.full,'collect',collect)
    def score(config,provenance):
        calls.append('score')
        assert calls.count('collect')==(6 if alias else 9)
        assert set(config['entries'])==set(m.ROLES)
        if alias:assert config['entries'][m.ROLES[0]]['operations']==config['entries'][m.ROLES[1]]['operations']
        if fault=='score':raise ValueError('bad score')
        out=Path(config['output']);out.mkdir(parents=True)
        result=dict(dataset=config['original_evaluation']['dataset'],runtime_provenance_verified=True)
        (out/'scores.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(m,'score',score)
    return c,calls


@pytest.mark.parametrize('alias',[False,True])
def test_unique_policy_dispatch_all_inference_before_scoring(tmp_path,monkeypatch,alias):
    c,calls=fixture(tmp_path,monkeypatch,alias=alias);result=m.run(c,{})
    assert set(result)==set(m.DATASETS)
    assert calls[:4]==['wait',*['admit:'+r for r in m.ROLES]]
    assert calls[4:]==['prepare','run','collect']*(6 if alias else 9)+['score']*3
    terminal=m.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==0 and not terminal['target_tuning'] and not terminal['training_dispatched']


@pytest.mark.parametrize('fault',['prior','pins','source','prepare','run','collect','score'])
def test_failures_stop_preserving_handles_without_retry(tmp_path,monkeypatch,fault):
    c,calls=fixture(tmp_path,monkeypatch,fault=fault)
    with pytest.raises(ValueError):m.run(c,{})
    terminal=m.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==1 and not terminal['restart']
    if fault!='score':assert 'score' not in calls
    if fault in ('prepare','run','collect'):assert calls.count('prepare')==1


def test_refuse_existing_outputs_and_target_policy_override(tmp_path,monkeypatch):
    c,calls=fixture(tmp_path,monkeypatch)
    c['datasets'][m.DATASETS[0]]['policy_id']='target-picked'
    with pytest.raises(ValueError):m.run(c,{})
    del c['datasets'][m.DATASETS[0]]['policy_id']
    Path(c['scores_root']).mkdir()
    with pytest.raises(ValueError):m.run(c,{})
    assert not calls


def test_detach_never_spawns_twice(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'load_config',lambda *a:(c,{}));calls=[]
    def spawn(argv,**kw):
        assert (tmp_path/'driver/intent.json').exists() and kw['start_new_session']
        calls.append(argv);return SimpleNamespace(pid=11)
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    assert m.detach('template','local',tmp_path/'driver')['pid']==11
    with pytest.raises(FileExistsError):m.detach('template','local',tmp_path/'driver')
    assert len(calls)==1
