import json
from types import SimpleNamespace

import pytest

from test_context_macro_training_operations import fixture
from tools.reports import context_acoustic_service_candidate as service
from tools.reports import context_acoustic_training_operations as operations
from tools.reports import context_macro_training_operations as shared


@pytest.mark.parametrize('role', ['client', 'preflight', 'service-0', 'service-1'])
def test_only_service_module_changes(tmp_path, role):
    c, _ = fixture(tmp_path)
    before, after = shared.command(c, role), operations.command(c, role)
    if role.startswith('service-'):
        index = before.index('tools.reports.context_dual_runtime_candidate')
        assert after[index] == 'tools.reports.context_acoustic_service_candidate'
        after[index] = before[index]
    assert after == before


def test_authenticated_service_retains_run_and_transport(monkeypatch):
    config = {'policy': {'physical_devices': [6]}}
    namespace = {}; seen = []
    exec('def run(c): return owner_class("sealed-source")(c["policy"])', namespace)
    sentinel = object(); namespace['serve_connection'] = sentinel
    def original(c):
        assert c is config
        seen.append('verify_runtime')
        return namespace
    def owner(root, *, acoustic):
        assert root == 'sealed-source' and acoustic is True
        return lambda policy: seen.append(policy) or 'engine'
    monkeypatch.setattr(service.parent, 'service_namespace', original)
    monkeypatch.setattr(service, 'owner_class', owner)
    bound = service.service_namespace(config)
    assert bound['serve_connection'] is sentinel
    assert bound['run'](config) == 'engine'
    assert seen == ['verify_runtime', config['policy']]


def test_shared_scope_rejected_before_engine_construction(monkeypatch):
    monkeypatch.setattr(service.parent,'service_namespace',lambda c:pytest.fail('must not initialize'))
    with pytest.raises(ValueError):
        service.service_namespace({'policy': {'process_device_scope': [4,5]}})


def test_prepared_arm_requires_explicit_finite_release(monkeypatch):
    monkeypatch.setattr(shared,'run_services',lambda *a,**k:pytest.fail('must not dispatch'))
    with pytest.raises(ValueError): operations.run({'finite_training_dispatch_allowed':False},{})
