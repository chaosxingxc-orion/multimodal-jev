import json
from pathlib import Path
import pytest
from test_context_transfer_parity_sequence import fixture
from tools.reports import run_context_transfer_parity_sequence_v2 as v2


@pytest.mark.parametrize('failure',[None,'prior','pin','tail','run','collect','old_terminal'])
def test_repaired_tail_binding_keeps_serial_fail_closed_sequence(tmp_path,monkeypatch,failure):
    c,calls=fixture(tmp_path,monkeypatch,failure)
    # Reuse the established lifecycle fixture, but bind the real v2 adapter.
    monkeypatch.setattr(v2.tail,'validate',v2.original.tail.validate)
    monkeypatch.setattr(v2.tail,'audit_collected',v2.original.tail.audit_collected)
    terminal=Path(c['predecessor_terminal']);value=json.loads(terminal.read_text())
    if failure!='old_terminal':value['status']='tail_v2_collected_audited'
    terminal.write_text(json.dumps(value))
    validate,run=v2.sequence()
    if failure:
        with pytest.raises(ValueError):run(c,{})
    else:run(c,{})
    expected=['wait']
    if failure not in ('prior','pin','old_terminal'):expected.append('tail-audit')
    if failure not in ('prior','pin','old_terminal','tail'):
        for d in v2.original.DATASETS:
            expected.append('run:'+d)
            if failure=='run' and d==v2.original.DATASETS[1]:break
            expected.append('collect:'+d)
            if failure=='collect':break
    assert calls==expected
    result=json.loads((Path(c['output'])/'terminal.json').read_text())
    assert result['exit_code']==int(failure is not None)
    if not failure:assert result['target_scored'] is False
    assert validate.__globals__['tail'] is v2.tail
    assert v2.original.validate.__globals__['tail'] is not v2.tail
