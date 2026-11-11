import json
from copy import deepcopy
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_rule_population import select_rule_rows,validate_selection_protocol,SELECTION_FULL


def release(tmp_path):
    rows=[dict(id='x',split='selection',optimization_allowed=False)]
    path=tmp_path/'selection.jsonl';path.write_text(json.dumps(rows[0])+'\n')
    manifest=dict(schema='jev-span-selection-rules-v1',training_admitted=False,records=1,
                  files={'selection.jsonl':file_hash(path)})
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    return rows


def test_selection_requires_explicit_population_and_stays_nontraining(tmp_path):
    rows=release(tmp_path)
    assert select_rule_rows(tmp_path,SELECTION_FULL)==rows
    with pytest.raises(ValueError,match='schema'):select_rule_rows(tmp_path)
    with pytest.raises(ValueError,match='schema'):select_rule_rows(tmp_path,'unknown')
    p=tmp_path/'selection.jsonl';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):select_rule_rows(tmp_path,SELECTION_FULL)


@pytest.mark.parametrize('fault',['policy','population','request','updates','training','pin'])
def test_protocol_rejects_changed_model_or_population(tmp_path,fault):
    rows=release(tmp_path);p=tmp_path/'protocol.json'
    protocol=dict(schema='jev-selection-frozen-evaluation-v1',release_manifest_sha256=file_hash(tmp_path/'MANIFEST.json'),
        population=SELECTION_FULL,request_ids=['x'],policy_ids=['base:adapter'],optimizer_updates=0,training_admitted=False)
    p.write_text(json.dumps(protocol))
    config=dict(base_id='base',adapter_id='adapter',release=str(tmp_path),population=SELECTION_FULL,
                evaluation_protocol=str(p),pins={str(p):file_hash(p)})
    assert validate_selection_protocol(config,rows)==protocol
    if fault=='policy':config['adapter_id']='other'
    elif fault=='population':config['population']='learning-probe-216-v1'
    elif fault=='request':rows[0]['id']='other'
    elif fault=='pin':config['pins']={}
    else:
        protocol['optimizer_updates' if fault=='updates' else 'training_admitted']=1 if fault=='updates' else True
        p.write_text(json.dumps(protocol));config['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):validate_selection_protocol(config,rows)
