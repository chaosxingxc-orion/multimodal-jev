import hashlib
import json

import pytest

from tools import verify_portable_configs as verifier


def fixture(tmp_path, expected=None, actual=None):
    expected = {'flag':False,'steps':[1,2]} if expected is None else expected
    actual = expected if actual is None else actual
    source=tmp_path/'source.json';source.write_text(json.dumps(expected))
    template=tmp_path/'template.json';template.write_text('{"value":"${VALUE}"}')
    # Wrap both sides to allow arbitrary JSON value cases.
    source.write_text(json.dumps(dict(value=expected)))
    local=tmp_path/'local.json'
    entry=dict(id='sample',template=str(template),source=str(source),
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros=dict(VALUE=actual,CONFIG_VERIFICATION_ENTRIES=[entry]))))
    return source,template,local


CONFIG='configs/portable/operations/verify-configs.json'


def test_exact_match_read_only(tmp_path):
    source,template,local=fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    result=verifier.verify(CONFIG,local)
    assert result['passed'] and len(result['entries'])==1
    assert before=={p:p.read_bytes() for p in tmp_path.iterdir()}
    assert verifier.verify(CONFIG,local)==result


@pytest.mark.parametrize('expected,actual',[(False,0),(1,1.0),([1,2],[2,1]),({'x':1},{'x':1,'y':2})])
def test_rejects_semantic_and_type_changes(tmp_path,expected,actual):
    _,_,local=fixture(tmp_path,expected,actual)
    with pytest.raises(ValueError,match='differs'):verifier.verify(CONFIG,local)


def test_source_hash_and_duplicate_identity(tmp_path):
    source,_,local=fixture(tmp_path);source.write_text('{}')
    with pytest.raises(ValueError,match='Frozen source changed'):verifier.verify(CONFIG,local)
    data=json.loads(local.read_text());entry=data['macros']['CONFIG_VERIFICATION_ENTRIES'][0]
    entry['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text('{"value":{"flag":false,"steps":[1,2]}}')
    entry['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    data['macros']['CONFIG_VERIFICATION_ENTRIES'].append(entry.copy());local.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Distinct'):verifier.verify(CONFIG,local)


def test_local_edit_between_reads_rejected(tmp_path,monkeypatch):
    _,_,local=fixture(tmp_path);original=verifier.load_config;calls=[]
    def changing(template,path):
        result=original(template,path);calls.append(template)
        if len(calls)==1:local.write_text(local.read_text()+'\n')
        return result
    monkeypatch.setattr(verifier,'load_config',changing)
    with pytest.raises(ValueError,match='bindings changed'):verifier.verify(CONFIG,local)
