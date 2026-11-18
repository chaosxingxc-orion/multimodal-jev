import copy

import pytest

from tools.jev.context_configurable_device_scope import validate_mapping


def valid(mounted_set=(0, 1, 2, 3, 4, 5, 6, 7), target=6):
    return dict(config=dict(physical_devices=[target], max_model_len=4096,
                            process_device_scope=dict(
                                kind='same_container_runtime_visibility_v1',
                                mounted_physical=list(mounted_set),
                                visible_ordinal=str(list(mounted_set).index(target)))),
                mounted=[f'davinci{d}' for d in mounted_set],
                visible=str(list(mounted_set).index(target)), count=1, physical=target)


@pytest.mark.parametrize('mounted_set,target', [
    ((0, 1, 2, 3, 4, 5, 6, 7), 6), ((0, 1, 2, 3, 4, 5, 6, 7), 7),
    ((0, 1, 2, 3), 2), ((5, 6), 5), ((7,), 7)])
def test_declared_mounted_set_derives_visible_ordinal(mounted_set, target):
    assert validate_mapping(**valid(mounted_set, target)) == target


@pytest.mark.parametrize('key,value', [
    ('visible', '0'), ('visible', None), ('count', 2), ('physical', 0),
    ('mounted', [f'davinci{i}' for i in range(6)]),
    ('mounted', [f'davinci{i}' for i in range(8)] + ['davinci8'])])
def test_actual_process_visibility_cannot_be_replaced(key, value):
    args = valid()
    args[key] = value
    with pytest.raises(ValueError):
        validate_mapping(**args)


@pytest.mark.parametrize('key,value', [
    ('physical_devices', [6, 7]), ('max_model_len', 8192), ('process_device_scope', None)])
def test_single_device_context4096_binding_required(key, value):
    args = copy.deepcopy(valid())
    args['config'][key] = value
    with pytest.raises(ValueError):
        validate_mapping(**args)


@pytest.mark.parametrize('scope', [
    dict(kind='other', mounted_physical=[0, 1, 2, 3, 4, 5, 6, 7], visible_ordinal='6'),
    dict(kind='same_container_runtime_visibility_v1', mounted_physical=[0, 1, 2, 3, 4, 5, 6, 7],
         visible_ordinal='6', extra=True),
    dict(kind='same_container_runtime_visibility_v1', mounted_physical=[], visible_ordinal='0'),
    dict(kind='same_container_runtime_visibility_v1', mounted_physical=[0, 0, 6],
         visible_ordinal='2'),
    dict(kind='same_container_runtime_visibility_v1', mounted_physical=[0, 1, 6],
         visible_ordinal='1'),
    dict(kind='same_container_runtime_visibility_v1', mounted_physical=[0, 1, 6],
         visible_ordinal=2)])
def test_versioned_scope_shape_and_ordinal_consistency(scope):
    args = copy.deepcopy(valid())
    args['config']['process_device_scope'] = scope
    with pytest.raises(ValueError):
        validate_mapping(**args)


def test_service_device_outside_declared_set_rejected():
    args = valid()
    args['config']['process_device_scope']['mounted_physical'] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match='outside the declared mounted set'):
        validate_mapping(**args)
