"""Read-only, compact runtime projection of an independently reproduced TRAIN release.

No data mining or model calls on the training path. The package pins this file and
the exact selection/schedule/consumer bytes from the producer's manifest.
"""
import hashlib
import json
from pathlib import Path


def pinned(path, digest):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('ordinary pinned file required: '+str(path))
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('view content changed: '+str(path))
    return json.loads(raw)


def layout(c):
    validate_execution(c['view_execution'])
    from multimodal_jev.context_topology_batch_plan import derive
    plan=derive(c['batch_config']);binding=c['view_binding'];steps=c['evaluation_steps']
    if (plan['dp_replicas']!=c['batch_config']['training_devices']
            or c['global_batch']!=plan['global_batch'] or c['microbatch']!=plan['configuration']['microbatch']
            or c['batch_config']['epochs']<2 or c['batch_config']['inference_instances']<1
            or c['stage']!='task' or c['algorithm']!='dual_credit_grpo_floor_clipped_v1' or c['dual_arm']!='joint'
            or type(c['seed']) is not int or c['seed']<0 or c['total_updates']!=plan['total_updates']
            or type(steps) is not list or len(steps)<2 or any(type(i) is not int for i in steps)
            or steps!=sorted(set(steps)) or steps[0]!=0 or steps[-1]!=plan['total_updates']
            or binding['component']!='stage2_difficulty_release_v2'
            or binding['manifest_sha256']!=c['schedule_sha256']
            or binding['source_policy_id']!=c['source_policy_id'] or c['source_policy_id']!=c['reference_policy_id']
            or binding['reference_manifest_sha256']!=c['reference_dataset']['manifest_sha256']):
        raise ValueError('dynamic Stage2 training, topology or endpoint differs')
    return plan


def indices(c):
    plan=layout(c);total=plan['total_updates'];per=plan['updates_per_epoch']
    kind=c['view_window']
    windows={'prefix':(0,2,[0,1]),'recovery':(1,2,[0,1]),'tail':(0,2,[per-1,per]),
             'formal':(0,total,list(range(total)))}
    start,stop,selected=windows[kind]
    if (type(c['start_update']) is not int or type(c['stop_update']) is not int
            or (c['start_update'],c['stop_update'])!=(start,stop)
            or bool(c.get('resume_root'))!=(kind=='recovery')
            or c['engineering_prefix'] is not (kind!='formal')):
        raise ValueError('window and optimizer cursor differ')
    return selected


def checkpoint_contract(c):
    # Identical prefix/recovery identities; excludes filesystem locations and
    # diagnostic cursors, but binds the actual TRAIN IDs, budget and objective.
    layout(c)
    return dict(c['view_binding'],lineage=c['view_lineage'],evaluation_steps=c['evaluation_steps'],batch_config=c['batch_config'],
                prompt_layout=c.get('prompt_layout'),schedule_mode='epoch_boundary' if c['view_window']=='tail' else 'ordered')


def consume(c,rows,task,*,directory=None):
    from tools.reports.context_replica_formal_candidate import validate_schedule
    root=Path(directory or c['schedule']);plan=layout(c);binding=c['view_binding']
    manifest=pinned(root/'MANIFEST.json',binding['manifest_sha256'])
    if manifest['schema']!='jev-stage2-dataset-release-v2':raise ValueError('wrong training dataset version')
    selection=pinned(root/'selection.json',manifest['files']['selection.json'])
    batches=pinned(root/'batches.json',manifest['files']['batches.json'])
    consumer=pinned(root/'consumer.json',manifest['files']['consumer.json'])
    full={r['id']:r for r in rows};ids=selection['selected_ids']
    from multimodal_jev.context_reference_dataset import digest
    if (digest(ids)!=binding['selected_ids_sha256'] or len(full)!=len(rows) or set(full)!=set(c['reference_dataset']['population'])
            or any(r.get('role')!='learning' or r.get('official_split')!='train'
                   or r.get('known_pending_source') is not False for r in rows)
            or len(set(ids))!=len(ids) or not set(ids)<=full.keys()
            or [item['id'] for item in task['schema']['candidates']]!=c['reference_dataset']['identity']['candidate_ids']
            or c['release_sha256']!=consumer['release_sha256']
            or manifest['contract']['endpoint']['prompt_layout']!=c.get('prompt_layout')
            or consumer['selected_ids_sha256']!=binding['selected_ids_sha256']
            or consumer['source_policy_id']!=c['source_policy_id']
            or consumer['reference_manifest_sha256']!=binding['reference_manifest_sha256']
            or consumer['stop_update']!=plan['total_updates'] or consumer['evaluation_steps']!=c['evaluation_steps']
            or consumer['seed']!=c['seed'] or consumer['batch_config']!=plan['configuration']):
        raise ValueError('release, full reference and selected training population differ')
    population={uid:full[uid] for uid in ids}
    counts=validate_schedule(batches,population.keys(),plan)
    if counts!=[b['real_queries'] for b in batches] or sum(counts)!=consumer['real_exposures']:
        raise ValueError('incorrect real exposure or tail denominator')
    return population,[batches[i] for i in indices(c)],consumer


def validate_execution(profile):
    import math
    def number(value, minimum=0):
        return type(value) in (int, float) and math.isfinite(value) and value >= minimum

    def keys(obj, names):
        return isinstance(obj, dict) and set(obj) == set(names.split())

    integer = ('actor_threads', 'collective_timeout_seconds', 'master_port',
               'max_input_tokens', 'objective_group_size', 'actor_seed')
    if (not isinstance(profile, dict)
            or not keys({k: v for k, v in profile.items() if k != 'activation_checkpointing'},
                        ' '.join(integer) + ' optimizer grad_clip policy_clip dual_coefficients container')
            or any(type(profile[k]) is not int or profile[k] < 1 for k in integer)
            or profile['master_port'] > 65535 or profile['objective_group_size']<2):
        raise ValueError('invalid Stage2 execution profile')
    if 'activation_checkpointing' in profile:
        setting = profile['activation_checkpointing']
        paths = setting.get('module_lists') if isinstance(setting, dict) else None
        if (not keys(setting, 'enabled module_lists')
                or type(setting['enabled']) is not bool or not isinstance(paths, list)
                or setting['enabled'] != bool(paths)
                or any(type(path) is not str or not all(part.isidentifier() for part in path.split('.'))
                       for path in paths)
                or len(paths) != len(set(paths))
                or any(a.startswith(b + '.') for a in paths for b in paths if a != b)):
            raise ValueError('invalid Stage2 activation_checkpointing settings')
    opt = profile['optimizer']; clip = profile['policy_clip']; dual = profile['dual_coefficients']
    resources = profile['container']
    if (not keys(opt, 'lr betas eps weight_decay')
            or not all(number(opt[k]) for k in ('lr','eps','weight_decay'))
            or opt['lr'] == 0 or opt['eps'] == 0
            or type(opt['betas']) is not list or len(opt['betas']) != 2
            or not all(number(beta) and beta < 1 for beta in opt['betas'])
            or not number(profile['grad_clip']) or profile['grad_clip'] == 0
            or not keys(clip, 'clip_ratio clip_ratio_low clip_ratio_high clip_ratio_c')
            or not all(number(clip[k]) for k in ('clip_ratio', 'clip_ratio_c'))
            or clip['clip_ratio_c'] <= 1
            or any(clip[k] is not None and not number(clip[k]) for k in ('clip_ratio_low','clip_ratio_high'))
            or not keys(dual, 'ranking_coefficient probability_coefficient kl_coefficient')
            or not all(number(value) for value in dual.values())
            or not keys(resources, 'cpus memory_gib shm_gib log_max_size_mib log_max_files')
            or any(type(value) is not int or value < 1 for value in resources.values())):
        raise ValueError('invalid Stage2 optimizer, objective or container settings')
    return profile


def actor_contract(config):
    profile = validate_execution(config['view_execution'])
    coefficients = profile['dual_coefficients']
    sampling=(dict(group_size=profile['objective_group_size'],sampling_seed=profile['actor_seed'],
                   clip_ratio=profile['policy_clip']['clip_ratio']) if 'objective_config' in config else {})
    return dict(
        optimizer=profile['optimizer'], grad_clip=profile['grad_clip'],
        dual_objective=dict(algorithm=config['algorithm'], arm=config['dual_arm'],
                            reference_policy_id=config['source_policy_id'], **coefficients,**sampling),
        objective=dict(group_size=profile['objective_group_size'], reward='component_binary_correctness',
                       kl=coefficients['kl_coefficient'], auxiliary='editable_stage2_objective'),
        view_execution=profile, seed=config['seed'])


def validate_objective_design(c):
    """CPU bind the training protocol, Python objective and execution profile before publishing a parent."""
    from multimodal_jev.source_snapshot import regular
    root=Path(__file__).resolve().parents[2]
    ref=c['objective_config']
    design=pinned(regular(root,ref['path']),ref['sha256'])
    source=regular(root,design['objective_source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=design['objective_sha256']:
        raise ValueError('objective source differs from pinned design')
    execution=design['execution']
    profile=pinned(regular(root,execution['source']),execution['sha256'])
    objective=c['stage2_objective']
    return editable_objective(dict(c,view_execution=profile,view_lineage={'objective_design_sha256':ref['sha256']},
        objective_plugin=dict(sha256=design['objective_sha256'],options=design['options'],
                              numerics=dict(policy_clip=profile['policy_clip'],advantage_std_floor=objective['advantage_std_floor']))))


def editable_objective(c):
    profile=validate_execution(c['view_execution'])
    objective=c['stage2_objective']
    if (set(objective)!={'algorithm','arm','group_size','clip_ratio','advantage_std_floor','learning_rate'}
            or objective['algorithm']!=c['algorithm'] or objective['arm']!=c['dual_arm']
            or objective['group_size']!=profile['objective_group_size']
            or objective['clip_ratio']!=profile['policy_clip']['clip_ratio']
            or objective['learning_rate']!=profile['optimizer']['lr']
            or c['objective_config']['sha256']!=c['view_lineage']['objective_design_sha256']):
        raise ValueError('editable objective and admitted execution differ')
    from multimodal_jev.grpo_numerics import validate
    numerics=validate(dict(advantage_std_floor=objective['advantage_std_floor'],policy_clip=profile['policy_clip']))
    plugin=c['objective_plugin']
    if plugin.get('numerics')!=numerics:
        raise ValueError('actual plugin numerical settings differ from the admitted objective')
    return dict(objective,plugin={key:plugin[key] for key in ('sha256','options','numerics')})


def validate_inherited_objective(config):
    """Reject tuning the sealed rollout protocol through an unrelated actor knob."""
    from tools.reports.context_cached_reference_trainer import validate_objective
    inherited = validate_objective(config)
    profile = validate_execution(config['view_execution'])
    expected = dict(ranking_coefficient=inherited['ranking_auxiliary_coefficient'],
                    probability_coefficient=inherited['binary_nll_coefficient'],
                    kl_coefficient=inherited['reference_bernoulli_kl_coefficient'])
    if (profile['dual_coefficients'] != expected
            or profile['objective_group_size'] != inherited['group_size']):
        raise ValueError('execution profile differs from the sealed rollout protocol')


class ReadOnlyAudioMirror:
    """Resolve published audio by content hash without writing to its storage."""
    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError('existing published audio directory required')

    def resolve(self, digest):
        from multimodal_jev.baseline import file_hash
        if (type(digest) is not str or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('explicit SHA256 audio identity required')
        path = self.root / (digest + '.audio')
        if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
            raise ValueError('mirrored audio missing or changed: ' + digest)
        return path
