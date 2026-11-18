import json
from pathlib import Path

import pytest

from tools.reports import context_replica_transfer_auxiliary as m


def test_exact_allowlist_preserves_manifest_and_refuses_reuse(tmp_path):
    manifest = tmp_path/'RUNTIME-SOURCES.json'
    manifest.write_text(json.dumps({'src/model.py':'sealed'}))
    before = manifest.read_bytes()
    proof = m.prepare(tmp_path, manifest)
    assert manifest.read_bytes() == before
    assert proof['model_sources_modified'] is False and proof['npu_executed'] is False
    assert set(proof['files']) == {n+'.py' for n in m.MODULES}
    for name in proof['files']:
        assert (tmp_path/'transfer-auxiliary'/name).read_bytes() == (Path(m.__file__).parent/name).read_bytes()
    compile((tmp_path/'transfer-client.py').read_text(), 'transfer-client.py', 'exec')
    with pytest.raises(ValueError, match='fresh'): m.prepare(tmp_path, manifest)


def test_cannot_shadow_sealed_model_modules(tmp_path):
    manifest = tmp_path/'RUNTIME-SOURCES.json'
    manifest.write_text(json.dumps({'tools/reports/'+m.MODULES[0]+'.py':'sealed'}))
    with pytest.raises(ValueError, match='sealed'): m.prepare(tmp_path, manifest)
    assert not (tmp_path/'transfer-auxiliary').exists()


def test_entrypoint_refuses_repository_imports_and_tampered_inventory(tmp_path):
    manifest = tmp_path/'RUNTIME-SOURCES.json'
    manifest.write_text('{}')
    m.prepare(tmp_path, manifest)
    code = (tmp_path/'transfer-client.py').read_text()
    namespace = {'__file__':str(tmp_path/'transfer-client.py'), '__name__':'probe'}
    # Running against the mutable repository must never masquerade as isolated
    # sealed-runtime execution, even when all auxiliary bytes are correct.
    with pytest.raises(ValueError, match='shadow'): exec(code, namespace)
    (tmp_path/'transfer-auxiliary/extra.py').write_text('raise AssertionError()')
    with pytest.raises(ValueError, match='inventory'): exec(code, namespace)
