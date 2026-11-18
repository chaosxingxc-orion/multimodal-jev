import json
from pathlib import Path

import pytest

from test_prepare_context_macro_replay import fixture
from tools.reports import prepare_context_macro_efficiency as m
from tools.reports.prepare_context_macro_replay import seal


def setup(tmp_path, monkeypatch):
    rc,parent,a=fixture(tmp_path)
    r=seal(rc,{},parent,a)
    replay=dict(parent,prepared=rc['output'],remote_root=rc['remote_root'],manifest_sha256=r['files_sha256'])
    admission=dict(status='finite_dp2_replay_audited',fresh_process_replay_verified=True)
    monkeypatch.setattr(m,'load_config',lambda *a:(replay,{}))
    monkeypatch.setattr(m,'audit_collected',lambda c:admission)
    # Synthetic fixture omits the full trainer; packaging/validation remain real.
    monkeypatch.setattr(m,'generate',lambda root:None)
    c=dict(replay_config='config',replay_local='local',output=str(tmp_path/'arms'),
        remote_root='/local/efficiency',project=str(Path.cwd()),steps=100,
        operations_root=str(tmp_path/'ops'),tag_prefix='efficiency')
    return c,replay,admission


def test_three_arms_keep_runtime_objective_schedule_and_start_identical(tmp_path,monkeypatch):
    c,replay,a=setup(tmp_path,monkeypatch)
    result=m.prepare(c,{})
    configs=[];sources=[]
    for micro in (8,16,32):
        arm=result['arms']['micro'+str(micro)];p=Path(arm['prepared'])/'package'
        t=json.loads((p/'training.json').read_text())
        assert t['microbatch']==micro and t['stop_update']==100 and t['start_update']==0
        assert t['reference_cache']==arm['remote_root']+'/reference-cache'
        assert not (p/'resume').exists()
        configs.append(t);sources.append((p/'RUNTIME-SOURCES.json').read_bytes())
        for rel in m.REFRESH:
            assert (p/rel).read_bytes()==(Path.cwd()/rel).read_bytes()
        ops=json.loads(Path(arm['operations']).read_text())
        assert m.validate(ops)['status']=='macro_efficiency_prepared_only'
    assert sources[0]==sources[1]==sources[2]
    for key in ('source_policy_id','source_contract','algorithm','schedule_sha256','release_sha256','global_batch'):
        assert configs[0][key]==configs[1][key]==configs[2][key]


@pytest.mark.parametrize('fault',['budget','not_replayed','audit_failure'])
def test_no_package_without_budget_and_actual_replay_audit(tmp_path,monkeypatch,fault):
    c,replay,a=setup(tmp_path,monkeypatch)
    if fault=='budget':c['steps']=101
    if fault=='not_replayed':a['fresh_process_replay_verified']=False
    if fault=='audit_failure':
        def fail(_):raise ValueError('actual audit failed')
        monkeypatch.setattr(m,'audit_collected',fail)
    with pytest.raises(ValueError):m.prepare(c,{})
    assert not Path(c['output']).exists()
