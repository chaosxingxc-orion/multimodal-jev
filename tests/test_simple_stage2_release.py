"""Formal entry refuses unreleased or unrelated evidence before model loading."""
import importlib.util
import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.resident_service_admission import IMAGE
from multimodal_jev.simple_stage2_release import authenticate, validate_experiment
from multimodal_jev.simple_resident_selection import SELECTION_UPDATES


def fixture(tmp_path):
    experiment = dict(schema='slurp-simple-sampled-rl-v1',training_dispatch_allowed=True,
        gates=dict(npu_audio_lifecycle=True),task='direct_audio_native_Final60',group_size=16,
        temperature=1.,reward='correctness_0_or_1',process_reward=0,optimizer_epochs_per_behavior_batch=1,
        ppo_clip_ratio=.2,dual_clip=3.,sampling='all_G_draws_with_replacement_keep_constant_groups',
        padding='explicit_zero_weight_excluded_from_Nreal',class_weight='same_frozen_full_learning_population_as_stage1',
        ce=dict(start=1.,end=.1,decay_fraction=.8,schedule='cosine'),reference_kl=dict(coefficient=.01,
            type='exact_categorical_policy_to_fixed_stage1',reference_refresh=False),
        budget_candidate=dict(learning_population=40513,epochs=2,global_batch=128,microbatch_per_rank=1,
            dp_world_size=4,updates_per_epoch=317,total_updates=634,new_stage_cursor=0,
            optimizer_reset_at_stage_boundary=True,selection_population=9226,
            selection_updates=list(SELECTION_UPDATES),schedule_sha256='schedule'))
    pins = {}
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
        pins[str(path)] = file_hash(path)
        return str(path)
    plan = dict(algorithm='grpo_v1', rollout_physical_devices=[4], lifecycle={})
    plan['experiment'] = put(tmp_path / 'experiment.json', experiment)
    continuous = None
    for role in ('continuous', 'restore', 'resume_update'):
        training = dict(sampled_probe=dict(algorithm='grpo_v1'))
        if role in ('restore','resume_update'): training['continuous_resume'] = dict(training_root=str(tmp_path / 'continuous-train'))
        train = put(tmp_path / (role + '-train/config.json'), training)
        files = {train: file_hash(train)}
        if continuous: files[continuous] = file_hash(continuous)
        report = dict(passed=True, updates=4 if role=='resume_update' else 3, ranks=4, files=files)
        if role == 'continuous': report.update(new_updates=2, exact_master_export=True,
            exact_peft_reload=True, exact_pre_update_recovery=True,
            checks=[dict(passed=True, requests=128)]*2)
        elif role=='resume_update': report.update(resumed_from_update=3,new_updates=1,recovery_probability_requests=128,
            exact_master_adam=True,exact_master_export=True,exact_peft_reload=True,exact_pre_update_recovery=True,
            checks=[dict(passed=True,requests=128)])
        else: report.update(requests=128, exact_master_adam=True, full_candidate_parity=True)
        audit = tmp_path / (role + '-audit')
        entry = dict(training_config=train,
            report=put(audit / 'run/state-verified.json', report),
            launch=put(audit / 'launch.json', dict(container_id=role, image=IMAGE, remote_root=str(audit))),
            observed=put(audit / 'observed.json', dict(container_id=role, image=IMAGE,
                state=dict(Status='exited', ExitCode=0))))
        plan['lifecycle'][role] = entry
        if role == 'continuous': continuous = entry['report']
    config = dict(stage2_initialization=plan, pins=pins, physical_devices=[0, 1, 2, 3],
        schedule_sha256=experiment['budget_candidate']['schedule_sha256'], global_batch=128,
        microbatch_per_rank=1, lora_execution_dtype='bfloat16', lora_rank=4, lora_alpha=8,
        lr=1e-4, seed=317, context_limit=4096, task_mode='direct_final', loss_reduction='real_request_mean')
    return config, put, experiment


def test_bound_algorithm_lifecycle_and_frozen_runtime(tmp_path):
    config, _, _ = fixture(tmp_path)
    report = authenticate(config)
    assert report['passed'] and report['algorithm'] == 'grpo_v1'


@pytest.mark.parametrize('fault',['missing','restore_only','no_update','wrong_source','partial'])
def test_formal_release_requires_real_resumed_optimizer_step(tmp_path,fault):
    config,put,_=fixture(tmp_path);lifecycle=config['stage2_initialization']['lifecycle']
    if fault=='missing':
        lifecycle.pop('resume_update')
    else:
        entry=lifecycle['resume_update'];path=Path(entry['report']);value=json.loads(path.read_text())
        if fault=='restore_only':value['updates']=3
        elif fault=='no_update':value['new_updates']=0
        elif fault=='partial':value['checks'][0]['requests']=32
        else:
            train=Path(entry['training_config']);source=json.loads(train.read_text())
            source['continuous_resume']['training_root']='/different';put(train,source)
            value['files'][str(train)]=file_hash(train)
        put(path,value)
    with pytest.raises((ValueError,KeyError)):authenticate(config)


@pytest.mark.parametrize('fault', ['dispatch', 'lifecycle_gate', 'group', 'kl', 'budget',
    'partial', 'live', 'image', 'algorithm', 'changed_file', 'missing_pin', 'other_restore', 'wrong_topology'])
def test_no_launch_with_unaccepted_changed_or_unrelated_evidence(tmp_path, fault):
    config, put, experiment = fixture(tmp_path); plan = config['stage2_initialization']
    if fault in ('dispatch', 'lifecycle_gate', 'group', 'kl', 'budget'):
        if fault == 'dispatch': experiment['training_dispatch_allowed'] = False
        elif fault == 'lifecycle_gate': experiment['gates']['npu_audio_lifecycle'] = False
        elif fault == 'group': experiment['group_size'] = 8
        elif fault == 'kl': experiment['reference_kl']['reference_refresh'] = True
        else: experiment['budget_candidate']['total_updates'] = 3
        put(Path(plan['experiment']), experiment)
    elif fault == 'wrong_topology': config['physical_devices'] = [4, 5, 6, 7]
    elif fault == 'missing_pin': config['pins'].pop(plan['lifecycle']['continuous']['report'])
    else:
        role = 'restore' if fault == 'other_restore' else 'continuous'
        key = 'observed' if fault in ('live', 'image') else 'training_config' if fault in ('algorithm', 'changed_file', 'other_restore') else 'report'
        path = Path(plan['lifecycle'][role][key]); value = json.loads(path.read_text())
        if fault == 'partial': value['checks'][0]['requests'] = 32
        elif fault == 'live': value['state']['Status'] = 'running'
        elif fault == 'image': value['image'] = 'other'
        elif fault == 'algorithm': value['sampled_probe']['algorithm'] = 'rloo_v1'
        elif fault == 'other_restore': value['continuous_resume']['training_root'] = '/unrelated'
        else: value['extra'] = 'changed'
        put(path, value)
        if fault in ('algorithm', 'other_restore'):
            report_path = Path(plan['lifecycle'][role]['report'])
            value = json.loads(report_path.read_text()); value['files'][str(path)] = file_hash(path)
            put(report_path, value)
    with pytest.raises(ValueError): authenticate(config)


def test_actual_formal_entry_refuses_before_any_model_import(tmp_path, monkeypatch):
    config, put, experiment = fixture(tmp_path)
    experiment['training_dispatch_allowed'] = False
    put(Path(config['stage2_initialization']['experiment']), experiment)
    source = Path(__file__).parents[1] / 'tools/jev/train_simple_stage2_npu.py'
    spec = importlib.util.spec_from_file_location('stage2_entry', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import sys
    monkeypatch.setitem(sys.modules, 'probe_simple_sampled_npu', None)
    monkeypatch.setitem(sys.modules, 'probe_simple_resident_train', None)
    with pytest.raises(ValueError, match='not been released'):
        module.run(config, tmp_path / 'output', 0)
    assert not (tmp_path / 'output').exists()


def test_candidate_still_refuses_formal_dispatch(tmp_path):
    _, _, experiment = fixture(tmp_path)
    experiment['training_dispatch_allowed'] = False
    with pytest.raises(ValueError, match='not been released'):
        validate_experiment(experiment, 'grpo_v1')


@pytest.mark.parametrize('resuming',[False,True])
def test_formal_entry_routes_shared_loop_through_correct_loader(tmp_path,monkeypatch,resuming):
    import sys,types
    import multimodal_jev.local_runtime_storage as storage
    config,_,_=fixture(tmp_path)
    if resuming:config['continuous_resume']={'source':'audited'}
    monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('LOCAL_RANK','0')
    monkeypatch.setattr(storage,'require_training_storage',lambda *a:None)
    source=Path(__file__).parents[1]/'tools/jev/train_simple_stage2_npu.py'
    spec=importlib.util.spec_from_file_location('stage2_entry_test',source)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    events=[];loop=object()
    def initialize(c,root,rank,*,after_initialize):
        assert c is config and rank==0 and after_initialize is loop;events.append('initialize')
    def resume(c,root,rank,*,after_resume):
        assert c is config and rank==0 and after_resume is loop;events.append('resume')
    monkeypatch.setitem(sys.modules,'probe_simple_sampled_npu',types.SimpleNamespace(
        initialize_stage2_actor=initialize,resume_stage2_actor=resume))
    monkeypatch.setitem(sys.modules,'probe_simple_resident_train',types.SimpleNamespace(continuous=loop))
    output=tmp_path/'output';entry.run(config,output,0)
    assert events==['resume' if resuming else 'initialize']
    assert json.loads((output/'rank-0/terminal.json').read_text())['exit_code']==0
