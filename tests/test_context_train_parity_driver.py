import json
from pathlib import Path
import pytest

from tools.reports import context_train_parity_operations as module


@pytest.mark.parametrize('fails',[False,True])
def test_driver_requires_independent_audit_and_preserves_failure(tmp_path,monkeypatch,fails):
    from tools.reports import audit_context_train_parity as auditor
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'launch',lambda *a:dict(status='collected_independent_train_parity_audit_required'))
    seen=[]
    def audit(c,p):
        seen.append(c)
        if fails:raise ValueError('parity output differs')
        return {'status':'independent_real_auditor_fixture'}
    monkeypatch.setattr(auditor,'audit',audit)
    c=dict(operations_root=str(tmp_path/'operations'),tag='parity',parity_package={'output':'package'})
    if fails:
        with pytest.raises(ValueError,match='output differs'):module.run_and_audit(c,{})
    else:module.run_and_audit(c,{})
    root=Path(c['operations_root'])/'finalization'
    t=json.loads((root/'terminal.json').read_text())
    assert t['exit_code']==int(fails)
    assert seen[0]['deployment']==c
    assert seen[0]['archive']==str(tmp_path/'operations/parity-collected.tar')
    assert (root/'audit.json').exists() is not fails
    if fails:assert t['preserve_handles'] and not t['restart']
