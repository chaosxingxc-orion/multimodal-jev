import copy
import hashlib
import json
import pytest
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_score import prepare_predictions, INVALID


def fixture():
    taxonomy={'scenario_action_pairs':[['email','send']], 'entity_types':['person']}
    target={'scenario':'email','action':'send','entities':[{'type':'person','filler':'amy'}]*2}
    request={'id':'r1','split':'train','source_group':'g1','audio_sha256':'audio',
             'input':{'audio':'/audio','schema_sha256':hashlib.sha256(canonical(taxonomy)).hexdigest()}}
    digest=hashlib.sha256(canonical(request)).hexdigest()
    gold={'id':'r1','target':target,'metadata':{'file':'r1.flac'},'request_sha256':digest}
    pred={'example_id':'r1','replicate':0,'request_sha256':digest,'raw_text':json.dumps(target)}
    return [request],[gold],[pred],taxonomy


def test_recording_and_entity_multiplicity_preserved():
    args=fixture();rows,report=prepare_predictions(*args)
    assert rows[0]['entities']==args[1][0]['target']['entities']
    assert report['format_valid']==1 and report['schema_valid']==1


def test_malformed_retained_and_out_of_taxonomy_not_repaired():
    args=fixture();args[2][0]['raw_text']='malformed'
    rows,report=prepare_predictions(*args)
    assert rows==[{'file':'r1.flac','scenario':INVALID,'action':INVALID,'entities':[]}]
    assert report['malformed']==1 and report['examples']==1
    value=copy.deepcopy(args[1][0]['target']);value['scenario']='wrong'
    value['entities'][0]={'type':'unknown','filler':'Amy'}
    args[2][0]['raw_text']=json.dumps(value)
    rows,report=prepare_predictions(*args)
    assert rows[0]['entities'][0]==value['entities'][0]
    assert report['format_valid']==1 and report['schema_valid']==0


@pytest.mark.parametrize('change',['missing','duplicate','unknown','binding','gold','taxonomy','replicate'])
def test_coverage_and_custody_fail_closed(change):
    args=fixture()
    if change=='missing':args[2].clear()
    if change=='duplicate':args[2].append(copy.deepcopy(args[2][0]))
    if change=='unknown':args[2][0]['example_id']='unknown'
    if change=='binding':args[2][0]['request_sha256']='changed'
    if change=='gold':args[1][0]['request_sha256']='changed'
    if change=='taxonomy':args[3]['entity_types'].append('unknown')
    if change=='replicate':args[2][0]['replicate']=True
    with pytest.raises(ValueError):prepare_predictions(*args)
