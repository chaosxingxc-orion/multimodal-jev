import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_builder import build
from multimodal_jev.context_reference_resume import ResumePool, archive_prefix, seal
from multimodal_jev.context_replica_macro_rollout import MacroRolloutPool
from test_context_reference_builder import setup


def pool(tmp_path, kw, name):
    engine = kw['engine']
    engine.bundle = dict(engine.bundle, contract=kw['identity']['contract'])
    engine.phase = 'ready'
    return MacroRolloutPool([engine], tmp_path/name, runtime_identity=engine.runtime_identity,
                            rollout_queries=1, max_queries=2)


@pytest.mark.parametrize('scenario', ['interrupted', 'replay-last', 'corrupt', 'wrong-input'])
def test_actual_macro_journal_resume(tmp_path, monkeypatch, scenario):
    from multimodal_jev import context_replica_macro_rollout as rollout
    monkeypatch.setattr(rollout, 'check_paths', lambda _: None)
    kw = setup(monkeypatch)
    original = pool(tmp_path, kw, 'original')
    calls = kw['engine'].calls
    report = build(tmp_path/'baseline', **dict(kw, engine=original))
    prefix = original.root/'batch-00000000'
    old = (prefix/'COMPLETE.json').read_bytes()
    if scenario != 'replay-last':
        (original.root/'batch-00000001/COMPLETE.json').unlink()
    lineage = {'test_endpoint': '318'}
    manifest = tmp_path/'prefix.json'
    payload = tmp_path/'prefix.zip'
    index = seal(original.root, lineage)
    manifest.write_text(json.dumps(index))
    if scenario == 'corrupt':
        with (prefix/'rollout-0000.json').open('ab') as stream:
            stream.write(b' ')
        with pytest.raises(ValueError, match='changed during packaging'):
            with payload.open('wb') as stream:
                archive_prefix(original.root, index, stream)
        with pytest.raises(ValueError, match='macro content changed'):
            build(tmp_path/'bad', **dict(kw, engine=ResumePool(
                pool(tmp_path, kw, 'fresh'), journal=original.root, manifest=manifest,
                manifest_sha256=file_hash(manifest), lineage=lineage)))
        return
    with payload.open('wb') as stream:
        archive_prefix(original.root, index, stream)
    restored = pool(tmp_path, kw, 'restored')
    calls.clear()
    engine = ResumePool(restored, archive=payload, manifest=manifest, manifest_sha256=file_hash(manifest),
                        lineage=lineage, replay_last=scenario == 'replay-last')
    if scenario == 'wrong-input':
        kw['assets']['s0']['audio_sha256'] = '9'*64
        with pytest.raises(ValueError, match='macro input'):
            build(tmp_path/'wrong', **dict(kw, engine=engine))
        assert not calls
        return
    actual = build(tmp_path/'resumed', **dict(kw, engine=engine))
    assert actual['identity'] == report['identity']
    assert json.loads((tmp_path/'resumed/records.json').read_text()) == json.loads(
        (tmp_path/'baseline/records.json').read_text())
    assert calls == [1]  # Replay at most the final one-query macro.
    assert (restored.root/'batch-00000000/COMPLETE.json').read_bytes() == old
    assert (prefix/'COMPLETE.json').read_bytes() == old
    assert engine.finish() == dict(manifest_sha256=file_hash(manifest), reused_macros=1,
        replayed_macros=int(scenario == 'replay-last'), new_macros=int(scenario == 'interrupted'))


def test_reference_resume_package_pins_overlay_and_rejects_wrong_endpoint(tmp_path, monkeypatch):
    from test_prepare_context_reference_job import setup as package_setup
    from tools.reports import prepare_context_reference_job as ref
    from multimodal_jev.context_reference_resume import lineage
    c = package_setup(tmp_path, monkeypatch)
    package = Path(c['parent']['output'])
    build_cfg = ref.build_config(json.loads((package/'config.json').read_text()), c['remote_root'],
                                 c['input_evidence'])
    prefix = tmp_path/'prefix'
    prefix.mkdir()
    manifest = prefix/'reference-prefix.json'
    archive = prefix/'reference-prefix.zip'
    archive.write_bytes(b'test payload')
    index = dict(commits=[{'COMPLETE.json': 'a'*64}], lineage=lineage(build_cfg,
        endpoint_sha256=file_hash(package/'initial/endpoint-commit.json')))
    manifest.write_text(json.dumps(index))
    receipt = prefix/'receipt.json'

    def pin():
        receipt.write_text(json.dumps(dict(schema='jev-reference-prefix-transport-v1', commits=1,
            prefix_sha256=file_hash(manifest), archive_sha256=file_hash(archive))))
        c['resume'] = dict(prefix_root=str(prefix), receipt_sha256=file_hash(receipt), replay_last=False)

    pin()
    c.update(legacy_engine_timeout_seconds=120, shutdown_timeout_seconds=180)
    ready = ref.prepare(c,{})
    c['manifest_sha256'] = ready['manifest_sha256']
    out = Path(c['output'])
    ref.audit(c,{})
    assert (out/'RUNTIME-SOURCES.json').read_bytes() == (package/'RUNTIME-SOURCES.json').read_bytes()
    manifest_data = json.loads((out/'PREPARED.json').read_text())
    assert 'reference-prefix.json' in manifest_data['layout']['configs']
    assert 'resume_reference.py' in (out/'run.py').read_text()
    runner = (out/'run.py').read_text()
    assert "'resume-service.py'" in runner and "'--engine-timeout', '120'" in runner
    assert 'shutdown_timeout=180' in runner
    assert ready['manifest_sha256'] == file_hash(out/'PREPARED.json')
    assert 'resume-service.py' in json.loads((out/'PREPARED.json').read_text())['files']
    for name in ('resume_reference.py','reference_resume_builder.py','reference_resume_pool.py'):
        compile((out/name).read_text(), name, 'exec')
    index['lineage']['endpoint_sha256'] = '0'*64
    manifest.write_text(json.dumps(index))
    pin()
    with pytest.raises(ValueError,match='exact same endpoint'):
        ref.audit(c,{})
