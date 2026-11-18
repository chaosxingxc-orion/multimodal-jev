"""Formal launch admission over a frozen simple-RL experiment and real audits.

No boolean grants acceptance by itself: lifecycle reports, exact process exits,
algorithm identity and complete evidence file graphs must agree.
"""
import json
from pathlib import Path

from .baseline import file_hash
from .resident_service_admission import IMAGE
from .simple_resident_selection import SELECTION_UPDATES


def validate_experiment(experiment, algorithm):
    if (experiment.get('schema') != 'slurp-simple-sampled-rl-v1'
            or experiment.get('training_dispatch_allowed') is not True
            or experiment.get('gates', {}).get('npu_audio_lifecycle') is not True):
        raise ValueError('formal experiment has not been released for training')
    expected = dict(task='direct_audio_native_Final60', group_size=16, temperature=1.,
        reward='correctness_0_or_1', process_reward=0, optimizer_epochs_per_behavior_batch=1,
        ppo_clip_ratio=.2, dual_clip=3., sampling='all_G_draws_with_replacement_keep_constant_groups',
        padding='explicit_zero_weight_excluded_from_Nreal',
        class_weight='same_frozen_full_learning_population_as_stage1')
    if (algorithm not in ('grpo_v1', 'rloo_v1')
            or any(experiment.get(k) != v for k, v in expected.items())
            or experiment.get('ce') != dict(start=1., end=.1, decay_fraction=.8, schedule='cosine')
            or experiment.get('reference_kl') != dict(coefficient=.01,
                type='exact_categorical_policy_to_fixed_stage1', reference_refresh=False)):
        raise ValueError('released simple-RL objective differs from implemented baseline')
    budget = experiment['budget_candidate']
    fixed = dict(learning_population=40513, epochs=2, global_batch=128, microbatch_per_rank=1,
        dp_world_size=4, updates_per_epoch=317, total_updates=634, new_stage_cursor=0,
        optimizer_reset_at_stage_boundary=True, selection_population=9226,
        selection_updates=list(SELECTION_UPDATES))
    if any(budget.get(k) != v for k, v in fixed.items()):
        raise ValueError('released full-data budget differs')
    return budget


def authenticate(config):
    """Called inside the immutable trainer before constructing any NPU model."""
    plan = config['stage2_initialization']; pins = config['pins']; files = {}
    def read(path):
        path = Path(path); digest = file_hash(path)
        if pins.get(str(path)) != digest:
            raise ValueError('pinned formal release evidence required: ' + str(path))
        files[str(path)] = digest
        return json.loads(path.read_text())
    experiment = read(plan['experiment'])
    budget = validate_experiment(experiment, plan['algorithm'])
    if (budget['schedule_sha256'] != config['schedule_sha256']
            or config['physical_devices'] != [0, 1, 2, 3]
            or plan['rollout_physical_devices'] != [4]
            or config['global_batch'] != 128 or config['microbatch_per_rank'] != 1
            or config['lora_execution_dtype'] != 'bfloat16'
            or config['lora_rank'] != 4 or config['lora_alpha'] != 8
            or config['lr'] != 1e-4 or config['seed'] != 317
            or config['context_limit'] != 4096 or config['task_mode'] != 'direct_final'
            or config['loss_reduction'] != 'real_request_mean'):
        raise ValueError('released runtime and training factors differ')
    continuous_source = None
    for role in ('continuous', 'restore', 'resume_update'):
        evidence = plan['lifecycle'][role]
        launch = read(evidence['launch']); observed = read(evidence['observed']); report = read(evidence['report'])
        expected_path = Path(launch['remote_root']) / 'run/state-verified.json'
        if (Path(evidence['report']) != expected_path or launch['image'] != IMAGE
                or observed['image'] != IMAGE or launch['container_id'] != observed['container_id']
                or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0
                or report.get('passed') is not True or report.get('updates') != (4 if role == 'resume_update' else 3) or report.get('ranks') != 4
                or not report.get('files')):
            raise ValueError('successful independent lifecycle audit required: ' + role)
        for path, digest in report['files'].items():
            if file_hash(path) != digest:
                raise ValueError('lifecycle evidence changed: ' + path)
            files[path] = digest
        training = read(evidence['training_config'])
        if report['files'].get(evidence['training_config']) != file_hash(evidence['training_config']):
            raise ValueError('audited algorithm source required')
        if training['sampled_probe']['algorithm'] != plan['algorithm']:
            raise ValueError('lifecycle accepted a different algorithm')
        if role == 'continuous':
            continuous_source = str(Path(evidence['training_config']).parent)
            if (report.get('new_updates') != 2 or any(report.get(k) is not True for k in
                    ('exact_master_export', 'exact_peft_reload', 'exact_pre_update_recovery'))
                    or len(report.get('checks', [])) != 2
                    or any(c.get('passed') is not True or c.get('requests') != 128 for c in report['checks'])):
                raise ValueError('continuous two-update lifecycle acceptance required')
        elif role == 'resume_update':
            if (report.get('resumed_from_update') != 3 or report.get('new_updates') != 1
                    or report.get('recovery_probability_requests') != 128
                    or any(report.get(k) is not True for k in
                        ('exact_master_adam','exact_pre_update_recovery','exact_master_export','exact_peft_reload'))
                    or len(report.get('checks', [])) != 1
                    or report['checks'][0].get('passed') is not True
                    or report['checks'][0].get('requests') != 128):
                raise ValueError('fresh-process resumed optimizer update acceptance required')
        elif (report.get('requests') != 128 or report.get('exact_master_adam') is not True
                or report.get('full_candidate_parity') is not True):
            raise ValueError('fresh-process continuous checkpoint recovery acceptance required')
        if role in ('restore', 'resume_update'):
            original = plan['lifecycle']['continuous']['report']
            if (training.get('continuous_resume', {}).get('training_root') != continuous_source
                    or report['files'].get(original) != file_hash(original)):
                raise ValueError('recovery must bind the same accepted continuous run')
    return dict(passed=True, algorithm=plan['algorithm'], files=files,
        scope='Frozen experiment and real lifecycle admission; final Stage1 independently authenticated by loader.')
