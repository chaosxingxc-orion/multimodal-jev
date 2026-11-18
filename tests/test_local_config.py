import json

import pytest

from multimodal_jev.local_config import load_config, resolve
from tools import configure


def test_nested_macros_and_json_types():
    config, used = resolve({'path': '${DATA}/train', 'count': '${COUNT}', 'cards': '${CARDS}'},
                          {'ROOT': '/shared', 'DATA': '${ROOT}/datasets', 'COUNT': 128, 'CARDS': [4, 5]})
    assert config == {'path': '/shared/datasets/train', 'count': 128, 'cards': [4, 5]}
    assert used == ['CARDS', 'COUNT', 'DATA', 'ROOT']


@pytest.mark.parametrize('config,macros', [
    ('${MISSING}', {}), ('${A}', {'A': '${B}', 'B': '${A}'}),
    ('x/${COUNT}', {'COUNT': 128}), ('${bad}', {}),
    ({'${KEY}': 1}, {'KEY': 'key'}), ('${ROOT', {'ROOT': '/tmp'}),
])
def test_invalid_macros_fail_closed(config, macros):
    with pytest.raises(ValueError):
        resolve(config, macros)


def test_no_environment_or_shell_interpolation(monkeypatch):
    monkeypatch.setenv('ROOT', '/unexpected')
    with pytest.raises(ValueError, match='Missing'):
        resolve('${ROOT}', {})
    assert resolve('${TEXT}', {'TEXT': '$(touch /never-run) `id` $HOME'})[0] == '$(touch /never-run) `id` $HOME'


def test_materialization_identity_and_nonoverwrite(tmp_path, monkeypatch):
    template, local, output = [tmp_path / name for name in ('template.json', 'local.json', 'out/config.json')]
    template.write_text(json.dumps({'path': '${ROOT}', 'batch': '${BATCH}'}))
    local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {'ROOT': '/a', 'BATCH': 128}}))
    monkeypatch.setattr(configure, 'check_paths', lambda *a: [])
    assert configure.materialize(template, local, output)['status'] == 'resolved'
    assert configure.materialize(template, local, output)['status'] == 'already_resolved'
    _, proof = load_config(template, local)
    assert '/a' not in json.dumps(proof)
    assert json.loads(output.read_text()) == {'path': '/a', 'batch': 128}
    local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {'ROOT': '/b', 'BATCH': 128}}))
    with pytest.raises(ValueError, match='Different or partial'):
        configure.materialize(template, local, output)
    assert json.loads(output.read_text())['path'] == '/a'


def test_partial_materialization_is_preserved(tmp_path, monkeypatch):
    template, local, output = [tmp_path / name for name in ('template.json', 'local.json', 'config.json')]
    template.write_text('{}')
    local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {}}))
    output.write_text('{}')
    monkeypatch.setattr(configure, 'check_paths', lambda *a: [])
    with pytest.raises(ValueError, match='partial'):
        configure.materialize(template, local, output)


def test_unrelated_macro_edits_reuse_without_rewriting_receipt(tmp_path, monkeypatch):
    template, local, output = [tmp_path / name for name in ('template.json','local.json','output.json')]
    template.write_text('{"model":"${MODEL}"}')
    bindings = dict(schema='jev-local-macros-v1', macros={'ROOT':'/models','MODEL':'${ROOT}/asr','CLEANUP':'/old'})
    local.write_text(json.dumps(bindings))
    monkeypatch.setattr(configure,'check_paths',lambda *a:[])
    configure.materialize(template,local,output)
    receipt=output.with_name(output.name+'.provenance.json');before=receipt.read_bytes()
    bindings['macros']['CLEANUP']='/new';bindings['macros']['UNUSED']=128
    local.write_text(json.dumps(bindings,indent=4))
    assert configure.materialize(template,local,output)['status']=='already_resolved'
    assert receipt.read_bytes()==before
    bindings['macros']['ROOT']='/other-models';local.write_text(json.dumps(bindings))
    with pytest.raises(ValueError,match='Different or partial'):
        configure.materialize(template,local,output)


def test_v1_receipts_remain_strict_and_readable(tmp_path,monkeypatch):
    template,local,output=[tmp_path/name for name in ('template.json','local.json','output.json')]
    template.write_text('{"value":"${VALUE}"}')
    bindings=dict(schema='jev-local-macros-v1',macros={'VALUE':128})
    local.write_text(json.dumps(bindings))
    value,provenance=load_config(template,local)
    output.write_text(json.dumps(value))
    receipt=output.with_name(output.name+'.provenance.json')
    receipt.write_text(json.dumps(dict(provenance,output_sha256=configure.sha(output))))
    monkeypatch.setattr(configure,'check_paths',lambda *a:[])
    assert configure.materialize(template,local,output)['status']=='already_resolved'
    bindings['macros']['UNUSED']='new';local.write_text(json.dumps(bindings))
    with pytest.raises(ValueError,match='Different or partial'):
        configure.materialize(template,local,output)
