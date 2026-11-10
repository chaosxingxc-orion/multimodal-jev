import hashlib
import json
import pytest
from multimodal_jev.consumption import canonical, request_from_example, inference_messages
from multimodal_jev.task_schema import Candidate, DecisionSchema, JevExample
from multimodal_jev.policy_evaluation import prediction_record
from multimodal_jev.frozen_contract import FrozenContract


def test_jev_existing_prompt_and_prediction_contract_unchanged():
    ex=JevExample('r','train','/audio',DecisionSchema('s','choice','choose',
        (Candidate('a','on'),Candidate('b','off'))),'a',{'source_group':'g','audio_sha256':'hash'})
    row=request_from_example(ex);contract=FrozenContract({'route':'asr-text'})
    assert contract.validate(row)==row
    assert contract.messages(row,'audio evidence','instruction')==inference_messages(row,'audio evidence','instruction')
    for raw in ['{"value":"a"}','{"value":true}','invalid','{"value":null}']:
        record=prediction_record(row['id'],raw)
        record['schema_valid']=record['format_valid'] and ex.schema.accepts(record.get('value'))
        assert contract.prediction(row,raw)==record


def setup(tmp_path):
    taxonomy={'scenario_action_pairs':[['email','send']], 'entity_types':['person']}
    path=tmp_path/'taxonomy.json';path.write_bytes(canonical(taxonomy))
    digest=hashlib.sha256(canonical(taxonomy)).hexdigest()
    cfg={'task_format':'slurp-native','route':'asr-text','taxonomy':str(path),'taxonomy_sha256':digest}
    row={'id':'private-id','split':'train','source_group':'private-group','audio_sha256':'hash',
         'input':{'audio':'/audio','schema_sha256':digest}}
    return cfg,row


def test_native_messages_hide_ids_and_preserve_duplicate_entities(tmp_path):
    cfg,row=setup(tmp_path);contract=FrozenContract(cfg);contract.validate(row)
    assert 'private' not in json.dumps(contract.messages(row,'actual audio','decide'))
    value={'scenario':'email','action':'send','entities':[{'type':'person','filler':'amy'}]*2}
    record=contract.prediction(row,json.dumps(value))
    assert record['format_valid'] and record['schema_valid'] and len(record['prediction']['entities'])==2
    assert not contract.prediction(row,'{"value":"a"}')['format_valid']
    value['scenario']='wrong'
    assert not contract.prediction(row,json.dumps(value))['schema_valid']


def test_native_refuses_label_fields_changed_taxonomy_and_unsupported_route(tmp_path):
    cfg,row=setup(tmp_path);contract=FrozenContract(cfg)
    with pytest.raises(ValueError):contract.validate({**row,'target':'leak'})
    row['input']['schema_sha256']='changed'
    with pytest.raises(ValueError):contract.validate(row)
    cfg['route']='audio-direct'
    with pytest.raises(ValueError):FrozenContract(cfg)
