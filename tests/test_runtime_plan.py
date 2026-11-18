import pytest
from tools.plan_runtime import plan, IMAGE


def config():
    return dict(schema='ascend-context-runtime-plan-v1',host='training-host',image=IMAGE,
        mount_source_container='a'*64,remote_root='/local/run/context-rl-grpo-team-v1',
        encoded_audio_cache='/local/cache/run/cache',audio_mirror_root='/local/audio/run',
        shared_root='/shared',driver_root='/driver',physical_devices=[4,5,6,7],
        resources=dict(cpus=24,memory='128g',shm_size='8g'))


def test_plan_is_offline_and_preserves_local_cache_and_readonly_assets():
    result=plan(config());args=result['argv']
    assert result['training_dispatch_allowed'] is False
    assert '--network=none' in args and '--pull=never' in args
    assert '/shared:/shared:ro' in args
    assert '/driver:/usr/local/Ascend/driver:ro' in args
    assert 'TMPDIR=/jev-tmp' in args
    assert '/local/run/context-rl-grpo-team-v1/cache:/jev-tmp' in args


@pytest.mark.parametrize('key,value',[
    ('image','latest'),('host','-oProxyCommand=id'),('mount_source_container','short'),
    ('remote_root','/shared/context-rl-grpo-team-v1'),('encoded_audio_cache','/shared/cache'),
    ('shared_root','/shared:rw'),('driver_root','/a/../driver'),
    ('physical_devices',[4,4,6,7]),('physical_devices',[True,5,6,7])])
def test_invalid_or_unvalidated_bindings_rejected(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):plan(c)
