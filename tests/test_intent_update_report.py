from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import report_intent_update_metrics as module


def test_independent_acceptance_failure_precedes_metrics(monkeypatch):
    def fail(*args):raise ValueError('backend mismatch')
    monkeypatch.setattr(module,'compare',fail)
    monkeypatch.setattr(module,'read',lambda *a:pytest.fail('read training before acceptance'))
    monkeypatch.setattr(module,'summarize',lambda *a:pytest.fail('reported before acceptance'))
    with pytest.raises(ValueError,match='backend mismatch'):module.run('/training','/suite')
