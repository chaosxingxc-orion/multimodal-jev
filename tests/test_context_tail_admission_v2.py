import json
from pathlib import Path
import pytest
from tools.reports import run_context_tail_admission_v2 as s


def write(root, name, value):
    p = root/(name+'.json'); p.write_text(json.dumps(value)); return str(p)


@pytest.mark.parametrize('failure', [None, 'predecessor', 'training', 'audit', 'audit_status'])
def test_tail_runs_once_only_after_successful_efficiency_queue(tmp_path, monkeypatch, failure):
    events = []
    c = dict(output=str(tmp_path/'out'), predecessor_driver=write(tmp_path,'driver',{'pid':1}),
        predecessor_terminal=write(tmp_path,'terminal',dict(exit_code=int(failure=='predecessor'),
                                                         status='kv_training100_compared')))
    monkeypatch.setattr(s,'validate',lambda c:dict(finite_training_dispatch_allowed=False,
                                                  tail_diagnostic_dispatch_allowed=False))
    monkeypatch.setattr(s,'check_paths',lambda c:None)
    monkeypatch.setattr(s,'wait_driver',lambda *a:events.append('wait'))
    def train(ops, provenance):
        events.append('training')
        assert ops['tail_diagnostic_dispatch_allowed'] is True
        assert ops['finite_training_dispatch_allowed'] is False
        if failure=='training':raise ValueError('training failed')
    def audit(*a, **kw):
        events.append('audit')
        assert kw['validator'] is s.tail.validate
        assert kw['command_builder'] is s.tail.command
        assert kw['output_auditor'] is s.tail_original.audit_outputs
        if failure=='audit':raise ValueError('audit failed')
        return dict(status='partial' if failure=='audit_status' else 'finite_dp2_tail_outputs_audited')
    monkeypatch.setattr(s.tail,'run',train)
    monkeypatch.setattr(s,'collect_and_audit',audit)
    if failure:
        with pytest.raises(ValueError):s.run(c,{})
    else:s.run(c,{})
    expected=['wait'] if failure=='predecessor' else ['wait','training'] if failure=='training' else ['wait','training','audit']
    assert events==expected
    result=json.loads((tmp_path/'out/terminal.json').read_text())
    assert result['exit_code']==int(failure is not None)
    if failure:assert result['restart'] is False and result['preserve_handles']
    else:assert result['formal_dispatched'] is False and result['acoustic_kv_tail_admitted'] is False


@pytest.mark.parametrize('fault',[None,'pin','parent','terminal','lease','attempt','dispatch'])
def test_validator_binds_queue_ancestry_and_rejects_reuse(tmp_path,monkeypatch,fault):
    root=tmp_path/'attempt';root.mkdir()
    ops=dict(operations_root=str(root),host='npu',lease='same',finite_training_dispatch_allowed=False,
             tail_diagnostic_dispatch_allowed=False)
    if fault=='lease':ops['lease']='other'
    if fault=='dispatch':ops['tail_diagnostic_dispatch_allowed']=True
    if fault=='attempt':(root/'attempt.json').write_text('{}')
    parent=dict(manifest_sha256='base')
    kv=dict(host='npu',lease='same',parent_operations=dict(parent_operations=parent))
    queue=dict(output=str(tmp_path/'queue'),operations=write(tmp_path,'kv',kv))
    c=dict(operations=write(tmp_path,'ops',ops),predecessor_driver=write(tmp_path,'driver',{'pid':1}),
           predecessor_config=write(tmp_path,'queueconfig',queue),
           predecessor_terminal=str(tmp_path/'queue/terminal.json'))
    if fault=='terminal':c['predecessor_terminal']=str(tmp_path/'other.json')
    c['pins']={c[k]:s.file_hash(c[k]) for k in ('operations','predecessor_driver','predecessor_config')}
    if fault=='pin':c['pins'][c['operations']]='bad'
    monkeypatch.setattr(s.tail,'validate',lambda c:dict(parent_manifest_sha256='wrong' if fault=='parent' else 'base'))
    if fault:
        with pytest.raises(ValueError):s.validate(c)
    else:assert s.validate(c)==ops
