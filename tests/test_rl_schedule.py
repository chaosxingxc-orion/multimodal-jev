import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path=Path(__file__).parents[1]/'tools/jev/run_audio_rl.py'
    spec=importlib.util.spec_from_file_location('schedule',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_stage_completion_requires_terminal_and_result(tmp_path):
    m=module()
    assert not m.completed(tmp_path)
    (tmp_path/'terminal.json').write_text('{"exit_code":0}')
    with pytest.raises(RuntimeError,match='without result'):m.completed(tmp_path)
    (tmp_path/'result.json').write_text('{}')
    assert m.completed(tmp_path)
    (tmp_path/'terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(RuntimeError,match='failed stage'):m.completed(tmp_path)


def test_existing_incomplete_stage_is_never_restarted(tmp_path,monkeypatch):
    m=module();stage=tmp_path/'actor';stage.mkdir()
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError,match='process inspection'):m.execute('update_audio_actor.py',{'x':1},stage)
    assert not calls


def test_completed_stage_does_not_rerun_or_accept_new_config(tmp_path,monkeypatch):
    m=module();stage=tmp_path/'rollout';stage.mkdir()
    (stage/'terminal.json').write_text('{"exit_code":0}')
    (stage/'result.json').write_text('{}')
    (tmp_path/'rollout-config.json').write_text('{"seed":17}')
    calls=[];monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append(a))
    m.execute('sample_audio_policy.py',{'seed':17},stage)
    assert not calls
    with pytest.raises(ValueError,match='config changed'):m.execute('sample_audio_policy.py',{'seed':29},stage)


def test_schedule_requires_explicit_existing_actor_base(tmp_path):
    m=module()
    config={'initial_checkpoint':str(tmp_path/'missing-rollout'),
            'manifest':str(tmp_path/'manifest'), 'actor':{}}
    with pytest.raises(ValueError,match='actor.model_path'):
        m.run(config,tmp_path/'out')


@pytest.mark.parametrize('adapted',[False,True])
def test_two_iterations_preserve_actor_base_and_use_fresh_rollout(tmp_path,monkeypatch,adapted):
    from test_rl_preflight import fixture
    m=module();config=fixture(tmp_path,adapted=adapted)
    config['audio_input_mode']='actor_features'
    original=json.loads(json.dumps(config));root=tmp_path/'schedule';root.mkdir()
    events=[];actors=[];rollouts=[];exports=[]
    def execute(tool,child,output):
        output.mkdir();events.append(tool)
        if tool=='sample_audio_policy.py':
            rollouts.append(child)
        else:
            actors.append(child)
            # Test doubles exercise scheduling only; not actual RL training.
            (output/'result.json').write_text(json.dumps({'updates':len(actors)-1}))
    def export(actor,export_root,stage):
        events.append('export');exports.append((str(actor),str(export_root)))
        return str(export_root/'merged')
    monkeypatch.setattr(m,'execute',execute);monkeypatch.setattr(m,'export_actor',export)
    m.run(config,root)
    assert events==['sample_audio_policy.py','update_audio_actor.py','export']*2
    assert rollouts[0]['checkpoint']==config['initial_checkpoint']
    assert all(child['audio_input_mode']=='actor_features' for child in rollouts)
    assert rollouts[1]['checkpoint']==str(Path(exports[0][1])/'merged')
    assert all(child['model_path']==config['actor']['model_path'] for child in actors)
    assert actors[0]['initial_adapter']==config['initial_adapter']
    assert actors[1]['initial_adapter'] is None
    assert actors[0]['parent_run'] is None
    assert actors[1]['parent_run']==exports[0][0]
    assert [child['sampling']['seed'] for child in rollouts]==[17,18]
    assert [child['seed'] for child in actors]==[17,18]
    assert exports[0][1]!=exports[1][1]
    result=json.loads((root/'result.json').read_text())
    assert result['iterations']==2 and result['optimizer_updates']==1
    assert config==original


def test_preflight_failure_never_invokes_rollout(tmp_path,monkeypatch):
    from test_rl_preflight import fixture
    m=module();config=fixture(tmp_path,adapted=True);calls=[]
    (Path(config['initial_checkpoint'])/'model.safetensors').write_bytes(b'changed')
    monkeypatch.setattr(m,'execute',lambda *a:calls.append(a))
    with pytest.raises(ValueError,match='published export file changed'):
        m.run(config,tmp_path/'schedule')
    assert not calls and not (tmp_path/'schedule').exists()


@pytest.mark.parametrize('limit,expected_iterations', [(0,0),(104,1),(208,2)])
def test_budget_stops_before_whole_batch_and_replays_completed_prefix(tmp_path,monkeypatch,limit,expected_iterations):
    from test_budget_control import profiled_fixture
    m=module();config=profiled_fixture(tmp_path)
    config['budget_control']['limits']={'training_tokens':limit,'generated_tokens':64}
    root=tmp_path/'run';root.mkdir();events=[]
    def execute(tool,child,output):
        if output.exists():return
        output.mkdir();events.append(tool)
        if tool=='sample_audio_policy.py':
            assert child['budget_prompt_counts']['ex']['prompt_tokens']==10
            result={'prompts':1,'rollouts':2,'generation_budget':{
                'requests':1,'responses':2,'request_prompt_tokens':10,
                'replicated_prompt_tokens':20,'generated_tokens':32,
                'length_terminated_responses':2,'engine_generate_seconds':1.,
                'prompt_tokens_per_request':[10],'response_tokens_per_record':[16,16]}}
        else:
            result={'updates':int(output.parent.name.split('-')[-1])+1,
                    'metrics':{'training_budget':child['training_reservation']}}
        (output/'result.json').write_text(json.dumps(result))
    def export(actor,export_root,stage):return str(export_root/'merged')
    monkeypatch.setattr(m,'execute',execute);monkeypatch.setattr(m,'export_actor',export)
    m.run(config,root)
    result=json.loads((root/'result.json').read_text())
    assert result['iterations']==expected_iterations
    assert len(events)==expected_iterations*2
    assert result['stop_reason']==('schedule_completed' if expected_iterations==2 else 'budget_exhausted')
    assert result['budget_control']['used']['training_tokens']==expected_iterations*104
    if expected_iterations==0:assert result['last_actor_run'] is None
    m.run(config,root)
    assert json.loads((root/'result.json').read_text())==result
    assert len(events)==expected_iterations*2
