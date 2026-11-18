import json
from pathlib import Path
import pytest
from tools.reports import run_objective_acceptance as module


def fixture(tmp_path,monkeypatch):
    import multimodal_jev.local_checkpoint as ckpt
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(ckpt,'check_paths',lambda _:None)
    c=dict(output=str(tmp_path/'sequence'),monitor_interval_seconds=900,pins=[],finite_training_dispatch_allowed=True,
        prefix=dict(package=str(tmp_path/'prefix-package'),operations_root=str(tmp_path/'prefix-ops'),
            training_package={'remote_root':'/data/prefix'},tag='prefix',finite_training_dispatch_allowed=True),
        recovery=dict(package=str(tmp_path/'replay-package'),operations_root=str(tmp_path/'replay-ops'),
            remote_root='/data/replay',tag='replay'))
    return c


@pytest.mark.parametrize('fail',[None,'prefix','prepare','recovery'])
def test_sequence_requires_audited_prefix_and_recovery(tmp_path,monkeypatch,fail):
    c=fixture(tmp_path,monkeypatch);calls=[]
    def prefix(ops,prov):
        calls.append('prefix')
        if fail=='prefix':raise ValueError('prefix rejected')
        root=Path(ops['operations_root'])/'finalization';root.mkdir(parents=True)
        (root/'audit-config.json').write_text(json.dumps({'complete_parent':'fixture'}))
        return dict(status='finite_objective_training_outputs_audited')
    def prepare(spec,prov):
        calls.append('prepare');assert spec['parent_audit']=={'complete_parent':'fixture'}
        if fail=='prepare':raise ValueError('prepare rejected')
        return dict(manifest_sha256='fixture')
    def recovery(ops,prov):
        calls.append('recovery');assert ops['training_package']['manifest_sha256']=='fixture'
        if fail=='recovery':raise ValueError('recovery rejected')
        return dict(status='finite_objective_recovery_audited',fresh_process_npu_resume_verified=True,
                    replay={'exact_objective_traces':True})
    monkeypatch.setattr(module.prefix,'run',prefix)
    monkeypatch.setattr(module.preparation,'prepare',prepare)
    monkeypatch.setattr(module.recovery,'run',recovery)
    if fail:
        with pytest.raises(ValueError):module.run(c,{})
    else:assert not module.run(c,{})['formal_training_admitted']
    expected=['prefix','prepare','recovery']
    assert calls==expected[:expected.index(fail)+1] if fail else calls==expected
    terminal=json.loads((Path(c['output'])/'terminal.json').read_text())
    assert terminal['exit_code']==int(bool(fail))
    if fail:assert not terminal['restart']


@pytest.mark.parametrize('fault',['dispatch','prefix_dispatch','paths','remote','interval','pin'])
def test_invalid_sequence_does_not_run(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    if fault=='dispatch':c['finite_training_dispatch_allowed']=False
    elif fault=='prefix_dispatch':c['prefix']['finite_training_dispatch_allowed']=False
    elif fault=='paths':c['recovery']['package']=c['prefix']['package']+'/child'
    elif fault=='remote':c['recovery']['remote_root']='/data/prefix/replay'
    elif fault=='interval':c['monitor_interval_seconds']=60
    else:
        p=tmp_path/'code';p.write_text('changed');c['pins']=[dict(path=str(p),sha256='0'*64)]
    with pytest.raises(ValueError):module.validate(c)
