from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_prepared_acceptance as module


def test_waits_ten_minutes_on_resource_before_creation(tmp_path,monkeypatch):
    calls=[]
    def check(image):
        calls.append(image)
        if len(calls)==1:raise ValueError('physical cards reserved by container: exact')
    waits=[];monkeypatch.setattr(module,'check_devices',check);monkeypatch.setattr(module.time,'sleep',waits.append)
    module.wait_resources(dict(image='fixed',pins={}),tmp_path)
    assert calls==['fixed','fixed'] and waits==[600]


def test_does_not_hide_unknown_resource_errors(tmp_path,monkeypatch):
    def fail(image):raise ValueError('host descriptors inaccessible')
    monkeypatch.setattr(module,'check_devices',fail)
    with pytest.raises(ValueError,match='inaccessible'):module.wait_resources(dict(image='fixed',pins={}),tmp_path)
