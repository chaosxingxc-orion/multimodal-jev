import json
from pathlib import Path
import pytest
from tools.reports import context_stage2_runtime_runner as runner


@pytest.fixture(autouse=True)
def local_runtime_environment(tmp_path,monkeypatch):
    for key in ('HOME','TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):
        monkeypatch.setenv(key,str(tmp_path))


def setup(tmp_path):
    config=dict(experiment_kind='engineering_prefix',engineering_prefix=True,training_dispatch_allowed=False,
                start_update=0,stop_update=2,checkpoint_interval=1,physical_devices=[4,5],
                objective_extension={'root':str(tmp_path/'objective-extension')})
    manifest=dict(training_devices=[4,5],inference_services=[dict(name='a',device=6),dict(name='b',device=7)],
                  layout={'services':{'a':'service-a.json','b':'service-b.json'}})
    for name,value in [('config.json',config),('PREPARED.json',manifest),
                       ('service-a.json',{'socket_path':str(tmp_path/'a')}),
                       ('service-b.json',{'socket_path':str(tmp_path/'b')})]:
        (tmp_path/name).write_text(json.dumps(value))
    return config,manifest


@pytest.mark.parametrize('stage',['prefix','recovery','efficiency'])
def test_configuration_drives_same_actor_entry(tmp_path,stage):
    config,manifest=setup(tmp_path)
    if stage=='recovery':config.update(start_update=1,resume_root=str(tmp_path/'resume'))
    if stage=='efficiency':config.update(experiment_kind='efficiency',stop_update=100)
    result=runner.plan(tmp_path,config,manifest)
    assert result['train_visible']=='0,1' and result['services']==[('a','2'),('b','3')]
    assert str(tmp_path/'objective-extension/bootstrap.py') in result['actor_command']


@pytest.mark.parametrize('kind',['formal_source','cached_reference_tail_admission'])
def test_full_and_tail_windows_use_registered_contracts(tmp_path,kind):
    config,manifest=setup(tmp_path)
    design=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/context-two-stage-v0.0.11.json').read_text())
    config.update(batch_config=design['stage2']['batch_config'],stage='task',experiment_kind=kind)
    if kind=='formal_source':
        config.update(engineering_prefix=False,training_dispatch_allowed=True,stop_update=634,
                      evaluation_steps=[0,*range(64,634,64),634],seed=317,
                      algorithm='dual_credit_grpo_floor_clipped_v1',dual_arm='joint')
    result=runner.plan(tmp_path,config,manifest)
    entry='objective-tail-entry.py' if kind=='cached_reference_tail_admission' else 'objective-extension/bootstrap.py'
    assert str(tmp_path/entry) in result['actor_command']


def test_preflight_failure_never_loads_service(tmp_path):
    setup(tmp_path);cache=tmp_path/'cache';cache.mkdir();calls=[]
    def reject(*args,**kwargs):
        assert kwargs['env']['PYTHONSAFEPATH']=='1'
        assert kwargs['env']['ASCEND_RT_VISIBLE_DEVICES']==''
        raise ValueError('packaged import failed')
    with pytest.raises(ValueError,match='packaged import'):
        runner.run(tmp_path,cache_root=cache,command_runner=reject,supervisor=lambda **kw:calls.append(kw))
    assert calls==[]
    result=json.loads((tmp_path/'startup-preflight.json').read_text())
    assert result['status']=='failed' and result['services_started'] is False


def test_same_safe_actor_environment_after_preflight(tmp_path):
    setup(tmp_path);cache=tmp_path/'cache';cache.mkdir();calls=[]
    def preflight(*args,**kwargs):calls.append('preflight')
    def supervise(**kw):
        assert calls==['preflight']
        assert kw['actor_env']['PYTHONSAFEPATH']=='1'
        assert kw['actor_env']['ASCEND_RT_VISIBLE_DEVICES']=='0,1'
        assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in kw['services']]==['2','3']
        return {'passed':True}
    assert runner.run(tmp_path,cache_root=cache,command_runner=preflight,supervisor=supervise)['passed']


@pytest.mark.parametrize('change',[{'resume_root':'/unexpected'}, {'start_update':1},
                                 {'training_dispatch_allowed':True}, {'stop_update':634}])
def test_invalid_window_rejected(tmp_path,change):
    config,manifest=setup(tmp_path);config.update(change)
    with pytest.raises(ValueError):runner.plan(tmp_path,config,manifest)
