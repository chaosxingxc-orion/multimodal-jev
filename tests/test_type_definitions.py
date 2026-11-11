import copy
import json
from pathlib import Path

import pytest

from multimodal_jev.type_definitions import review_contract, validate_catalog

PATH=Path(__file__).parents[1]/'configs/tasks/slurp/type-definition-candidates-v1.json'


def test_catalog_complete_and_review_renderer_contains_no_examples():
    catalog=json.loads(PATH.read_text())
    names=[r['entity_type'] for r in catalog['types']]
    assert len(names)==55
    validate_catalog(catalog,names)
    for row in catalog['types']:
        result=review_contract(catalog,row['entity_type'])
        assert result['optimization_allowed'] is False
        assert row['definition'] in result['schema'].question
        assert result['thinking'].output=='decision_only'
        assert 'evidence_source_ids' not in result['schema'].question
        assert 'text_reference' not in json.dumps(catalog)


@pytest.mark.parametrize('error',['duplicate','missing','unknown_neighbor','empty_definition','approval','empty_requirements'])
def test_invalid_or_prematurely_approved_catalog_rejected(error):
    catalog=json.loads(PATH.read_text());names=[r['entity_type'] for r in catalog['types']]
    if error=='duplicate':catalog['types'].append(copy.deepcopy(catalog['types'][0]))
    if error=='missing':catalog['types'].pop()
    if error=='unknown_neighbor':catalog['types'][0]['confusable_types']=['invented']
    if error=='empty_definition':catalog['types'][0]['definition']=''
    if error=='approval':catalog['types'][0]['optimization_allowed']=True
    if error=='empty_requirements':catalog['types'][0]['thinking_requirements']=[]
    with pytest.raises(ValueError):validate_catalog(catalog,names)
