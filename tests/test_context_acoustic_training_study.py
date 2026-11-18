import json
from pathlib import Path

import pytest
from tools.reports import run_context_acoustic_training_study as s


@pytest.mark.parametrize('failure',[None,'predecessor','admission','training','audit'])
def test_ordered_single_attempt_stops_on_failure(tmp_path,monkeypatch,failure):
    events=[]
    def write(name,value):
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));return str(p)
    c=dict(output=str(tmp_path/'out'),operations=write('ops',dict(parent_operations={'baseline':True})),
        probe_operations=write('probe',dict(operations_root=str(tmp_path),tag='a')),
        predecessor_driver=write('driver',{'pid':1}),
        predecessor_terminal=write('terminal',dict(exit_code=1 if failure=='predecessor' else 0,
            status='kv_capacity_independently_compared')))
    monkeypatch.setattr(s,'pins',lambda c:None)
    monkeypatch.setattr(s,'check_paths',lambda c:None)
    monkeypatch.setattr(s.training,'validate',lambda c:None)
    monkeypatch.setattr(s.acoustic,'validate',lambda c:None)
    monkeypatch.setattr(s,'wait_driver',lambda *a:events.append('wait'))
    def stage(name):
        events.append(name)
        if failure==name:raise ValueError('failure '+name)
        return dict(exact_probabilities_and_logits=True,exact_sampling=True)
    monkeypatch.setattr(s.acoustic,'audit',lambda *a:stage('admission'))
    monkeypatch.setattr(s,'audit_collected',lambda *a:stage('baseline_audit'))
    def train(c,p):
        assert c['finite_training_dispatch_allowed'] is True
        return stage('training')
    monkeypatch.setattr(s.training,'run',train)
    monkeypatch.setattr(s,'collect_and_audit',lambda *a,**kw:stage('audit'))
    monkeypatch.setattr(s,'compare',lambda *a:stage('compare'))
    if failure:
        with pytest.raises(ValueError):s.run(c,{})
    else:s.run(c,{})
    expected=['wait','admission','baseline_audit','training','audit','compare']
    if failure=='predecessor':expected=['wait']
    elif failure:expected=expected[:expected.index(failure)+1]
    assert events==expected
    result=json.loads((tmp_path/'out/terminal.json').read_text())
    assert result['exit_code']==int(failure is not None)
    if failure:assert result['restart'] is False
    else:assert result['automatic_promotion'] is False


@pytest.mark.parametrize('fault',[None,'order','cache','image','incomplete'])
def test_comparison_requires_actual_matching100_and_full_audits(tmp_path,monkeypatch,fault):
    baseline=dict(operations_root=str(tmp_path/'base'),manifest_sha256='base')
    candidate=dict(operations_root=str(tmp_path/'new'),manifest_sha256='new',parent_operations=baseline)
    for ops in (baseline,candidate):
        root=Path(ops['operations_root'])/'collected/run';root.mkdir(parents=True)
        for i in range(100):
            rows=[i, i+100]
            if fault=='order' and ops is candidate and i==99:rows.reverse()
            (root/f'batch-{i:04d}.json').write_text(json.dumps(dict(records=rows)))
    monkeypatch.setattr(s.training,'validate',lambda c:None)
    def summary(root,**kwargs):
        assert kwargs==dict(start_update=0,stop_update=100,warmup_updates=2)
        new='new' in str(root)
        return dict(all_updates=dict(complete_seconds=80 if new else 100),
            measured=dict(complete_seconds=60 if new else 90),
            steps=[dict(reference_misses=127 if fault=='cache' and new else 128)]*100)
    monkeypatch.setattr(s,'summarize',summary)
    audits=[dict(status='finite_dp2_efficiency_audited',files_sha256=n,image='same') for n in ('base','new')]
    if fault=='image':audits[1]['image']='other'
    if fault=='incomplete':audits[1]['status']='partial'
    if fault:
        with pytest.raises(ValueError):s.compare(baseline,candidate,audits)
    else:
        r=s.compare(baseline,candidate,audits)
        assert r['all_updates_speedup']==1.25 and r['after_warmup_speedup']==1.5
        assert r['saved_seconds']==20 and r['matched_actual_queries']
