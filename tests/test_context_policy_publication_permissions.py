"""Published execution artifacts must survive collection by another user."""
import os
import stat

import pytest

from test_context_policy_artifacts import fixture
from multimodal_jev.context_policy_artifacts import read_execution_bundle, export_native_adapter, verify_native_adapter
from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
from multimodal_jev.context_policy_snapshot import export_policy, _parameters


def test_new_training_packages_include_publication_fixes():
    from tools.reports.prepare_context_macro_training import ADDITIONS
    assert {
        'src/multimodal_jev/context_policy_snapshot.py',
        'src/multimodal_jev/context_checkpoint_policy.py',
        'src/multimodal_jev/context_policy_artifacts.py',
    } <= set(ADDITIONS)


@pytest.mark.parametrize('kind', ['snapshot', 'checkpoint', 'adapter'])
def test_shared_publication_with_restrictive_umask(tmp_path, monkeypatch, kind):
    model, source, policy_id, contract = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr('multimodal_jev.local_runtime_storage.check_paths', lambda *a: [])
    monkeypatch.setattr('multimodal_jev.context_checkpoint_policy.check_paths', lambda *a: [])
    bundle = read_execution_bundle(source, expected_policy_id=policy_id,
        expected_contract=contract, expected_base_id='a'*64)
    # Processor directories copied with copytree can also inherit owner-only access.
    (source/'processor').chmod(0o700)
    out = tmp_path/'published'
    old_umask = os.umask(0o077)
    try:
        if kind == 'snapshot':
            identity = export_policy(out, model, source/'processor', contract)
        elif kind == 'checkpoint':
            execution = {k: p.detach().clone() for k, p in _parameters(model).items()}
            identity = export_checkpoint_policy(source, out, execution, policy_id=policy_id,
                contract=contract, base_id='a'*64)['policy_id']
        else:
            export_native_adapter(out, bundle)
    finally:
        os.umask(old_umask)
    for path in [out, *out.rglob('*')]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o755 if path.is_dir() else 0o644)
    if kind == 'adapter':
        verify_native_adapter(out, bundle)
    else:
        read_execution_bundle(out, expected_policy_id=identity,
            expected_contract=contract, expected_base_id='a'*64)
