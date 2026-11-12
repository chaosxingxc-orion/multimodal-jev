import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import audit_intent_saved_supervision as module


def test_failed_acceptance_prevents_supervision_read(monkeypatch):
    def fail(*args):raise ValueError('actual gate failed')
    monkeypatch.setattr(module,'compare',fail)
    monkeypatch.setattr(module,'read',lambda *a:pytest.fail('read before acceptance'))
    with pytest.raises(ValueError,match='actual gate failed'):module.run('/train','/suite')


def test_different_objective_not_silently_treated_as_correctness(monkeypatch):
    monkeypatch.setattr(module,'compare',lambda *a:{})
    monkeypatch.setattr(module,'read',lambda *a:{'arm':'D','objective':{'name':'smoothed-utility-v1'}})
    monkeypatch.setattr(module,'load_pilot',lambda *a:pytest.fail('wrong objective'))
    with pytest.raises(ValueError,match='original correctness'):module.run('/train','/suite')
