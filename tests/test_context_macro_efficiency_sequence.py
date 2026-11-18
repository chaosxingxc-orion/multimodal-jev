import json
from pathlib import Path

import pytest

from tools.reports import run_context_macro_efficiency_sequence as module


def setup(tmp_path,monkeypatch,fault=None):
    root=tmp_path/'baseline';root.mkdir();(root/'finalization-driver.json').write_text('{}')
    (root/'finalization-terminal.json').write_text(json.dumps(dict(status='finite_training_collected_audited',audit_status='finite_dp2_efficiency_audited')))
    arms={'micro8':({'operations_root':str(root)}, {})}
    for name in ('micro16','micro32'):
        ops=tmp_path/name;ops.mkdir();arms[name]=({'operations_root':str(ops)}, {})
    c=dict(output=str(tmp_path/'sequence'),pins={},predecessor_finalizer=str(root/'finalization-driver.json'))
    comparison={'output':str(tmp_path/'comparison.json')};events=[]
    def validate(c):events.append('validate');return arms,comparison,{}
    monkeypatch.setattr(module,'validate_sequence',validate)
    monkeypatch.setattr(module,'check_paths',lambda p:None)
    monkeypatch.setattr(module,'wait_driver',lambda *a:events.append('wait'))
    def audit(ops):
        events.append('baseline_audit')
        if fault=='baseline_audit':raise ValueError('bad baseline')
        return {'status':'finite_dp2_efficiency_audited'}
    monkeypatch.setattr(module,'audit_collected',audit)
    def train(ops,provenance):
        name=Path(ops['operations_root']).name;events.append('train_'+name)
        if fault=='train_'+name:raise ValueError('actual run failed')
    monkeypatch.setattr(module,'train',train)
    def collect(ops,**kw):
        name=Path(ops['operations_root']).name;events.append('collect_'+name)
        if fault=='collect_'+name:raise ValueError('actual audit failed')
        assert kw['result_dirs']==('run','journal','service-0','service-1')
        return {'status':'finite_dp2_efficiency_audited'}
    monkeypatch.setattr(module,'collect_and_audit',collect)
    monkeypatch.setattr(module,'summarize',lambda *a,**kw:{'summary':kw})
    def compare(c):events.append('compare');return {'matched':True}
    monkeypatch.setattr(module,'compare',compare)
    if fault=='finalizer':(root/'finalization-terminal.json').write_text('{"status":"failed"}')
    return c,events


def test_exact_wait_then_sequential_audit_and_compare_without_poll_or_restart(tmp_path,monkeypatch):
    c,events=setup(tmp_path,monkeypatch)
    assert module.run(c,{})=={'matched':True}
    assert events==['validate','wait','validate','baseline_audit','train_micro16','collect_micro16','train_micro32','collect_micro32','compare']
    terminal=json.loads((Path(c['output'])/'terminal.json').read_text())
    assert terminal['exit_code']==0 and not terminal['formal_training_dispatched']
    with pytest.raises(FileExistsError):module.run(c,{})


@pytest.mark.parametrize('fault',['finalizer','baseline_audit','train_micro16','collect_micro16','train_micro32','collect_micro32'])
def test_failed_predecessor_or_successor_stops_sequence(tmp_path,monkeypatch,fault):
    c,events=setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):module.run(c,{})
    terminal=json.loads((Path(c['output'])/'terminal.json').read_text())
    assert terminal['exit_code']==1 and terminal['preserve_handles'] and not terminal['restart']
    assert 'compare' not in events
    if fault in ('finalizer','baseline_audit','train_micro16','collect_micro16'):assert 'train_micro32' not in events
