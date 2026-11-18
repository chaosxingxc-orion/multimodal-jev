import pytest

from tools.reports.context_device_visibility import visibility
from tools.jev.context_configurable_device_scope import validate_mapping


@pytest.mark.parametrize('training,inference', [([4,5],[6,7]),([7,2],[5,0]),([0,1],[2,3])])
def test_visible_ids_satisfy_real_process_scope_validator(training,inference):
    manifest=dict(training_devices=training,
                  inference_services=[dict(name=str(i),device=d) for i,d in enumerate(inference)])
    actor,services=visibility(manifest);mounted=sorted(training+inference)
    assert [mounted[int(n)] for n in actor.split(',')]==training
    for (_,visible),physical in zip(services,inference):
        policy=dict(physical_devices=[physical],max_model_len=4096,
            process_device_scope=dict(kind='same_container_runtime_visibility_v1',
                mounted_physical=mounted,visible_ordinal=visible))
        assert validate_mapping(policy,mounted=['davinci'+str(d) for d in mounted],
            visible=visible,count=1,physical=physical)==physical


def test_shared_card_rejected():
    with pytest.raises(ValueError):
        visibility(dict(training_devices=[4,5],inference_services=[dict(name='a',device=5)]))
