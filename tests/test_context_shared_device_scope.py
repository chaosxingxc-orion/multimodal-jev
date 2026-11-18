import copy
import pytest
from tools.jev.context_shared_device_scope import validate_mapping


def valid():
    return dict(config=dict(physical_devices=[7],max_model_len=4096,
        process_device_scope=dict(kind='same_container_runtime_visibility_v1',
                                  mounted_physical=[4,5,6,7],visible_ordinal='3')),
        mounted=['davinci4','davinci5','davinci6','davinci7'],
        visible='3',count=1,physical=7)


def test_actual_scope_required():
    assert validate_mapping(**valid())==7


@pytest.mark.parametrize('key,value',[
    ('visible',None),('visible','7'),('count',4),('physical',3),
    ('mounted',['davinci7']),('mounted',[f'davinci{i}' for i in range(8)])])
def test_environment_or_files_cannot_replace_actual_mapping(key,value):
    args=valid();args[key]=value
    with pytest.raises(ValueError):validate_mapping(**args)


@pytest.mark.parametrize('key,value',[
    ('physical_devices',[6]),('max_model_len',8192),('process_device_scope',None)])
def test_unversioned_or_changed_configuration_rejected(key,value):
    args=copy.deepcopy(valid());args['config'][key]=value
    with pytest.raises(ValueError):validate_mapping(**args)
