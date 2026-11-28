"""Bind predeclared policy endpoints to independently verified test inputs."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from multimodal_jev.model_provider import require_same_provider
from tools.reports.prepare_tpo_windows import macros
from tools.reports.native_evaluation_cache import catalog_match, shared_publication, read_verified


def read(p):return json.loads(Path(p).read_bytes())


def evaluation_cache(source, dataset, parent, initial, *, defer_audit=False):
    if 'catalog_cache' in source:
        cache = dict(source['catalog_cache'])
        if (not {'seal', 'processor', 'encoding'} <= set(cache)
                or set(cache) - {'dataset', 'seal', 'processor', 'encoding', 'model_provider', 'panel_size',
                                 'output', 'cache_identity'}):
            raise ValueError('catalog cache must declare only the sealed input identity')
        cache.setdefault('dataset', dataset)
        if ('output' in cache) != ('cache_identity' in cache):
            raise ValueError('explicit cache location requires pinned audit evidence')
    elif 'cache_config' in source:
        ref = source['cache_config']
        path = Path(ref['path'])
        if file_hash(path) != ref['sha256']:
            raise ValueError('evaluation cache config changed')
        cache = read(path)
    elif 'scored_cache' in source:
        ref = source['scored_cache']
        if file_hash(ref['path']) != ref['sha256']:
            raise ValueError('previously scored cache configuration changed')
        cache = read(ref['path'])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['endpoints'][dataset]['cache']
    else:
        old = Path(source['package'])
        endpoint = read(old/'entry/full-102.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['endpoint']
        cache = dict(endpoint['cache'])
        old_ops = read(old/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
        cache['output'], = [p.split(':')[0] for p in old_ops['read_only_mounts']
                            if p.endswith(':/jev-inputs/full-cache')]
    require_same_provider(parent, cache)
    if 'cache_config' in source and not defer_audit:
        read_verified(cache, strict=False)
    if initial is not None and any(k in source for k in ('catalog_cache', 'cache_config', 'scored_cache')):
        from multimodal_jev.cached_context_preparation import processor_cache_identity
        if cache['processor']['sha256'] != processor_cache_identity(initial):
            raise ValueError('evaluation processor differs from the initialized model')
    if cache.get('panel_size') is not None or cache['dataset'] != dataset:
        raise ValueError('full frozen test required')
    return cache


def block_queue(config, common, datasets):
    queue = config.get('sample_blocks')
    if queue is None:
        return None
    allowed = {'devices', 'batches_per_block', 'prefetch', 'resume_from', 'parity', 'replicas_per_device'}
    if (not isinstance(queue, dict) or queue.keys() - allowed
            or not {'devices', 'batches_per_block', 'prefetch'} <= queue.keys()):
        raise ValueError('invalid sample block scheduling configuration')
    devices = queue['devices']
    from multimodal_jev.card_device_config import physical_devices
    physical_devices(devices)
    if any(type(queue[k]) is not int or queue[k] < 1 for k in ('batches_per_block', 'prefetch')):
        raise ValueError('positive block size and prefetch required')
    from tools.reports.card_evaluation_replicas import replica_count
    replicas = replica_count(queue.get('replicas_per_device', 1))
    if replicas > 1 and ('single_card' not in common or 'choice' in common or config.get('single_card_probe')):
        raise ValueError('replicas_per_device applies to formal single-card scoring')
    if 'choice' in common and (not common.get('full_ontology_only') or common.get('choice_only')
            or 'candidate_lengths' in common['choice'] or 'paired_lengths' in common):
        raise ValueError('variable Choice records depend on legacy shard/batch indices')
    for dataset, source in datasets.items():
        if type(source.get('logical_partitions')) is not int or source['logical_partitions'] < 1:
            raise ValueError(f'positive logical_partitions required for {dataset}')
    return queue


def queue_jobs(config, entry, common, datasets, endpoints, inferred, mounts, image, *, reuse_policy=None,
               finite_parity_shards=None):
    settings = block_queue(config, common, datasets)
    parent = dict(common, phase='tpo_eval_block_worker', endpoints=endpoints, inferred_arms=inferred)
    queue = dict(root='/jev-work/sample-blocks', batches_per_block=settings['batches_per_block'],
                 prefetch=settings['prefetch'],
                 partitions={dataset: source['logical_partitions'] for dataset, source in datasets.items()})
    if 'parity' in settings:
        parity = settings['parity']
        if (set(parity) != {'blocks_per_dataset', 'legacy'}
                or type(parity['blocks_per_dataset']) is not int or parity['blocks_per_dataset'] < 1
                or set(parity['legacy']) != set(datasets)):
            raise ValueError('limited parity requires pinned legacy shards for every dataset')
        for dataset, arms in parity['legacy'].items():
            if set(arms) != set(inferred[dataset]):
                raise ValueError('limited parity requires references for every inferred policy')
            for pins in arms.values():
                if len(pins) != queue['partitions'][dataset]:
                    raise ValueError('limited parity requires the legacy logical partition layout')
                for pin in pins:
                    path = Path(pin['path'])
                    if not path.is_absolute():
                        raise ValueError('limited parity references require absolute remote paths')
                    mounts.add(str(path.parent))
        queue['parity'] = parity
    probe = config.get('single_card_probe')
    if probe:
        parent['single_card_probe'] = probe
    queue['namespace'] = content_key(parent, queue, image)
    parent['queue'] = queue
    plan = dict(parent, phase='tpo_eval_block_plan', devices=[], result_path='/jev-work/blocks-planned.json')
    if finite_parity_shards is not None:
        plan['finite_policy_parity_shards'] = finite_parity_shards
    if 'resume_from' in settings:
        resume = Path(settings['resume_from'])
        target = Path(config['deployment']['remote_root'])
        if (not resume.is_absolute() or '..' in resume.parts or resume.name != 'sample-blocks'
                or resume.is_relative_to(target) or target.is_relative_to(resume)):
            raise ValueError('resume requires a distinct, absolute past sample-blocks directory')
        mounts.add(str(resume))
        plan['queue'] = dict(queue, resume_from=str(resume))
    if reuse_policy:
        if 'resume_from' in settings:
            raise ValueError('choose either exact resume or completed policy reuse')
        path = Path(reuse_policy['plan']['path'])
        if not path.is_absolute() or path.name != 'plan.json' or '..' in path.parts:
            raise ValueError('a pinned remote evaluation plan is required')
        mounts.add(str(path.parent))
        plan['queue'] = dict(queue, reuse_policy=reuse_policy)
    commit_json(entry/'block-plan.json', macros('NATIVE_SEAL_CONTAINER_CONFIG', plan))
    prep = dict(name='block-plan', devices=[], collect='blocks-planned.json', **config['cache_container'])
    workers = []
    for index, device in enumerate(settings['devices']):
        name = f'block-worker-{index}'
        receipt = name + '.json'
        job = dict(parent, devices=[device], output='/jev-work/'+name, result_path='/jev-work/'+receipt)
        if 'replicas_per_device' in settings:
            job['replicas_per_device'] = settings['replicas_per_device']
        if probe:
            job['single_card_probe'] = dict(probe, shards=len(settings['devices']), shard=index)
        commit_json(entry/(name+'.json'), macros('NATIVE_SEAL_CONTAINER_CONFIG', job))
        workers.append(dict(name=name, devices=[device], collect=receipt, **config['eval_container']))
    return prep, workers, queue


def prepare(config):
    pkg,repo = Path(config['package']),Path(config['repository'])
    ops=Path(config['training_ops'])
    prior=read(config['training_operations_local'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    if 'expected_host' in config and prior['host'] != config['expected_host']:
        raise ValueError('evaluation host differs from the registered deployment')
    job=read(Path(config['training_operations_local']).parent/'entry/finite.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    training_entry = Path(config['training_operations_local']).parent/'entry'
    joint = None
    if 'joint_sha' in job['training']:
        plan_path = training_entry/'joint.json'
        if file_hash(plan_path) != job['training']['joint_sha']:
            raise ValueError('joint training plan changed')
        joint = read(plan_path)
        binding = dict(joint_sha=job['training']['joint_sha'], target_sha=joint['target_sha'],
                       parent_sha=joint['parent_sha'], world_size=len(joint['training_devices']),
                       initial_policy=joint['source_policy'])
    else:
        binding={k:job['training'][k] for k in ('target_sha','parent_sha','world_size')}
    if 'next_macro_sha' in job['training']:
        binding['next_macro_sha'] = job['training']['next_macro_sha']
    binding['source']=job['source']['sha256']
    if (ops/'finite-collected.json').exists() and read(ops/'finite-collected.json')['binding']!=binding:
        raise ValueError('finite receipt differs from the sealed training package')
    orig=Path(config['parent_config'])
    parent=read(orig)
    full = joint is not None and joint.get('initialization') == 'pre_task_policy'
    if joint:
        steps = joint['updates']
    else:
        targets_path = training_entry/'targets/MANIFEST.json'
        if file_hash(targets_path) != binding['target_sha']:
            raise ValueError('frozen training target manifest changed')
        steps = read(targets_path)['updates']
    if config.get('bind_collected_formal'):
        receipt, terminal = ops/'tpo-collected.json', ops/'terminal.json'
        proof = read(receipt)
        if (proof['binding'] != binding or proof['updates'] != steps or proof['passed'] is not True
                or proof['window'] != 'formal' or proof['arm'] != 'tpo'
                or read(terminal) != {'exit_code': 0, 'status': 'declared_stages_finished'}):
            raise ValueError('formal endpoint must match the collected training window')
        config = dict(config, collected_predecessor=dict(receipt_sha256=file_hash(receipt),
                                                         terminal_sha256=file_hash(terminal)))
        calibration = config['calibration']
        config['calibration'] = dict(calibration, policy_ids=dict(calibration['policy_ids'], tpo=proof['endpoint']))

    remote_parent = Path(prior['remote_root'])/'entry' if full else Path(parent['source_policy']).parents[1]
    parent_file = remote_parent/('parent.json' if full else 'config.json')
    if 'model_provider' in config:
        require_same_provider(config,parent)
    if file_hash(orig)!=binding['parent_sha']:
        raise ValueError('parent identity changed')
    entry=pkg/'entry'
    entry.mkdir(parents=True,exist_ok=False)
    frozen=freeze(repo,entry/'source',patterns=config['source_patterns'],io_workers=config['io_workers'])
    reuse = config.get('evaluation_cache_reuse')
    catalog = []
    cache_paths = {}
    if reuse:
        catalog_path = repo / reuse['catalog']
        catalog = read(catalog_path)
        if catalog['schema'] != 'jev-native-evaluation-cache-catalog-v1':
            raise ValueError('invalid native evaluation cache catalog')
        catalog = catalog['entries'] if reuse.get('catalog_host') == prior['host'] else []
        source_identity = read(entry/'source/FILES.json')
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    common=dict(source=dict(root='/jev-entry/source',sha256=frozen['manifest_sha256']),io_workers=config['io_workers'])
    if full:
        if parent['source_policy'] != '/jev-entry/initial/policy':
            raise ValueError('unexpected single-stage policy layout')
        initial = training_entry/'initial/policy'
        if file_hash(initial/'MANIFEST.json') != joint['source_manifest_sha256']:
            raise ValueError('step-zero policy changed')
        # Recreate the sealed parent's relative container layout for evaluation.
        shutil.copytree(initial, entry/'initial/policy')
    if full and 'model_provider' in parent and config.get('full_ontology_only'):
        if ('choice_protocol' in config or 'choice_baseline' in config or 'paired_lengths' in config
                or set(config.get('arms', ())) != {'stage1', 'tpo', *config.get('checkpoint_steps', {})}):
            raise ValueError('use the bound Joint TPO objective and its policy endpoints')
        common.update(choice={'temperature':joint['objective']['temperature']},full_ontology_only=True)
    elif joint and config.get('full_ontology_only') and 'choice_protocol' not in config:
        if 'paired_lengths' in config:
            raise ValueError('full ontology and variable Choice must be scored separately')
        common.update(choice={'temperature':joint['objective']['temperature']},full_ontology_only=True)
        if config.get('external_comparisons') is True:
            if 'choice_baseline' in config:
                raise ValueError('external comparisons cannot also rerun a baseline')
            common['external_comparisons'] = True
        elif 'choice_baseline' in config:
            common['choice_baseline'] = config['choice_baseline']
        elif 'stage1' not in config.get('arms', ()):
            raise ValueError('register an audited Choice comparison endpoint')
    elif 'choice_protocol' in config:
        choice_baseline = config['choice_baseline']
        if config.get('choice_only'):
            if (not joint or config.get('full_ontology_only') or 'paired_lengths' in config
                    or file_hash(config['choice_protocol']) != config['choice_protocol_sha256']):
                raise ValueError('Choice-only comparison requires a pinned external variable protocol')
            if file_hash(choice_baseline['operations']) != choice_baseline['operations_sha256']:
                raise ValueError('Choice comparison deployment changed')
            baseline_ops = read(choice_baseline['operations'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
            if baseline_ops['host'] != prior['host']:
                raise ValueError('Choice baseline deployment must share the evaluation host')
            choice_baseline = dict(root=str(Path(baseline_ops['remote_root'])/'tpo'),
                                   receipt_sha256=choice_baseline['receipt_sha256'])
            common['choice_only'] = True
        spec=read(config['choice_protocol'])
        if common.get('choice_only') and not spec['choice'].get('candidate_lengths'):
            raise ValueError('Choice-only comparison requires variable candidate lengths')
        common.update(choice=spec['choice'],choice_baseline=choice_baseline)
        if config.get('full_ontology_only'):
            if 'paired_lengths' in config or file_hash(config['choice_protocol'])!=binding['target_sha']:
                raise ValueError('expected bound full-ontology protocol')
            common['full_ontology_only']=True
        if 'paired_lengths' in config:
            if file_hash(config['choice_protocol'])!=binding['target_sha']:raise ValueError('Choice identity differs')
            common['paired_lengths']=config['paired_lengths']
    if full and ('arms' not in config or 'direct' in config['arms']):
        raise ValueError('single-stage run has only initial and TPO endpoints')
    if 'arms' in config:
        common['arms']=config['arms']
    endpoint_bindings = config.get('audited_endpoints', {})
    if endpoint_bindings:
        common['audited_endpoints'] = endpoint_bindings
    if 'probability_panels' in config:
        common['probability_panels'] = config['probability_panels']
    checkpoint_steps = config.get('checkpoint_steps', {})
    if checkpoint_steps:
        if (not full or not config.get('training_custody')
                or set(checkpoint_steps.values()) - set(joint['intermediate_export_updates'])
                or any(arm != f'step-{step}' for arm, step in checkpoint_steps.items())
                or set(checkpoint_steps) - set(common['arms'])):
            raise ValueError('only audited pre-registered intermediate checkpoints may be evaluated')
        proof = read(Path(config['training_custody'])/'formal-audit.json')
        if (proof['status'] != 'formal_training_collected_and_audited' or proof['updates'] != steps
                or proof['endpoint'] != read(ops/'tpo-collected.json')['endpoint']):
            raise ValueError('formal checkpoint audit required')
        common['checkpoints'] = {arm: proof['checkpoints'][f'policy-{arm}'] for arm in checkpoint_steps}
    jobs=[];mounts={str(remote_parent),prior['remote_root']};sets={};pieces={}
    mounts.update(str(Path(endpoint['root']).parent) for endpoint in endpoint_bindings.values())
    encode_jobs, verify_jobs, cache_jobs, semantic = [], [], [], {}
    calibration_groups = []
    calibration = config.get('calibration')
    if calibration:
        if (full or 'probability_panels' not in common or 'choice' in common
                or set(calibration['policy_ids']) != set(common['arms'])
                or not calibration['devices'] or len(set(calibration['devices'])) != len(calibration['devices'])):
            raise ValueError('source calibration requires paired raw probabilities and distinct devices')
        roster = Path(calibration['roster']['path'])
        if (file_hash(roster) != calibration['roster']['sha256']
                or file_hash(training_entry/'targets'/roster.name) != calibration['roster']['sha256']):
            raise ValueError('calibration roster differs from the frozen training design')
        shutil.copyfile(roster, entry/'calibration-roster.json')
        rule = dict(calibration, roster=dict(path='/jev-entry/calibration-roster.json',
                                             sha256=calibration['roster']['sha256']))
        parent_args = dict(parent=str(parent_file), parent_sha=binding['parent_sha'], calibration=rule)
        mounts.add(parent['encoded_audio_cache'])
        prior_predictions = config.get('reused_calibration_predictions', {})
        if (set(prior_predictions) - set(calibration['policy_ids']) or any(
                not parts or len({p['path'] for p in parts}) != len(parts)
                for parts in prior_predictions.values())):
            raise ValueError('each reused calibration arm requires distinct pinned shards')
        paths = [Path(p['path']) for parts in prior_predictions.values() for p in parts]
        if any(not path.is_absolute() for path in paths):
            raise ValueError('reused calibration predictions require absolute paths')
        mounts.update(str(path.parent) for path in paths)
        remaining = [arm for arm in common['arms'] if arm not in prior_predictions]
        calibration_jobs, shards = [], []
        if remaining:
            for i, device in enumerate(calibration['devices']):
                name = f'calibration-{i}'
                output = '/jev-work/'+name
                opts = dict(common, **parent_args, phase='tpo_eval', arms=remaining,
                            endpoint=dict(cache=dict(processor={}), audio_token_id=parent['audio_token_id']),
                            devices=[device], shard=i, shards=len(calibration['devices']), output=output,
                            trained=prior['remote_root'], batch=config['batch'], binding=binding,
                            updates=steps, prefetch=calibration['prefetch'])
                commit_json(entry/(name+'.json'), macros('NATIVE_SEAL_CONTAINER_CONFIG',opts))
                calibration_jobs.append(dict(name=name, devices=[device], **config['eval_container']))
                shards.append(output)
        threshold_file = '/jev-work/thresholds.json'
        fit = dict(common, **parent_args, phase='tpo_score', calibration_fit=True,
                   shards=shards, output=threshold_file,
                   reused_calibration_predictions=prior_predictions)
        commit_json(entry/'calibration-fit.json', macros('NATIVE_SEAL_CONTAINER_CONFIG',fit))
        calibration_groups = [calibration_jobs] if calibration_jobs else []
        calibration_groups.append([dict(name='calibration-fit', devices=[],
                                        collect='thresholds.json', **config['cache_container'])])
    cache_store_used = False
    if 'choice_baseline' in common:mounts.add(common['choice_baseline']['root'])
    datasets=config['datasets']
    reused_predictions = {}
    inferred_arms = {}
    if 'formal_datasets' in config:
        selected=config['formal_datasets']
        if not selected or len(set(selected))!=len(selected):
            raise ValueError('unique formal dataset IDs required')
        datasets={key:datasets[key] for key in selected}
    pooled = block_queue(config, common, datasets)
    for dataset,source in datasets.items():
        if 'encode' in source:
            if full or not config.get('full_ontology_only') or not config.get('publish_seals'):
                raise ValueError('semantic evaluation requires a pinned Stage1 parent and full panels')
            encode = source['encode']
            cache = dict(encode['cache'], dataset=dataset, output='/jev-work/'+source['name']+'-cache',
                         processor=dict(policy=parent['source_policy'], sha256=encode['processor_sha256']))
            require_same_provider(parent, cache)
            if cache.get('panel_size') is not None:
                raise ValueError('full frozen test required')
            semantic[dataset] = encode['scoring']
            if 'reuse_verified_cache' in source:
                legacy = source['reuse_verified_cache']
                if file_hash(legacy['operations']) != legacy['operations_sha256']:
                    raise ValueError('previous semantic cache deployment changed')
                predecessor = read(legacy['operations'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
                if predecessor['host'] != prior['host']:
                    raise ValueError('verified cache must be on the registered host')
                cache['output'] = predecessor['remote_root'] + '/' + source['name'] + '-cache'
                cache['cache_identity'] = dict(manifest_sha256=legacy['manifest_sha256'],
                                               verification_sha256=legacy['verification_sha256'])
                mounts.add(cache['output'])
            elif not reuse:
                for phase, group in (('encode', encode_jobs), ('verify', verify_jobs)):
                    name = source['name']+'-'+phase
                    commit_json(entry/(name+'.json'), macros('NATIVE_SEAL_CONTAINER_CONFIG',
                                                              dict(common, phase=phase, cache=cache)))
                    group.append(dict(name=name, devices=[], **config['cache_container']))
            mounts.add(cache['seal']['root'])
        else:
            cache = evaluation_cache(source, dataset, parent, initial if full else None, defer_audit=bool(reuse))
        if reuse:
            cache['processor'] = dict(cache['processor'], policy=parent['source_policy'])
            mode = 'verified'
            if 'cache_identity' not in cache:
                match = catalog_match(cache, reuse['catalog_root'], reuse['catalog_seal_root'], catalog)
                if match:
                    cache, mode = match, 'catalog'
                elif 'catalog_cache' in source or 'encode' in source or 'publication_key' in cache:
                    # Use the store on this host even if the source came from a prior scored package.
                    cache, mode = shared_publication(
                        cache, reuse['store'], image=prior['image'], source=source_identity)
            cache_paths[dataset] = dict(mode=mode, output=cache['output'])
            publish = mode == 'shared'
            if publish:
                if 'catalog_cache' in source:
                    cache.update(reuse['build'], io_workers=config['io_workers'])
                cache_store_used = True
                cache_paths[dataset]['publication_key'] = cache['publication_key']
            else:
                mounts.add(cache['output'])
            role = source['name'] + ('-cache' if publish else '-cache-check')
            receipt = role+'-receipt.json'
            phase = 'publish_eval_cache' if publish else 'preflight_eval_cache'
            commit_json(entry/(role+'.json'), macros('NATIVE_SEAL_CONTAINER_CONFIG',
                        dict(common, phase=phase, cache=cache, result_path='/jev-work/'+receipt)))
            cache_jobs.append(dict(name=role, devices=[], publish_eval_cache=publish,
                                   collect=receipt, **config['cache_container']))
        elif 'encode' not in source:
            if 'output' not in cache:
                raise ValueError('catalog cache requires evaluation_cache_reuse')
            mounts.add(cache['output'])
        if 'seal_sha256' in source and cache['seal']['sha256'] != source['seal_sha256']:
            raise ValueError('comparison and evaluated population must share the registered seal')
        if 'cache_config' in source or 'scored_cache' in source or 'catalog_cache' in source or reuse:
            mounts.add(cache['seal']['root'])
        c=dict(cache=cache,cache_verification=cache['output']+'/VERIFIED.json',audio_token_id=cache['encoding']['audio_token_id'] if full else parent['audio_token_id'])
        sets[dataset]=c;pieces[dataset]=[]
        historical = source.get('reused_predictions', {})
        if historical:
            if 'probability_panels' not in common or 'choice' in common:
                raise ValueError('raw audited predictions can only be reused for probability panels')
            expected_policies = {'stage1': parent['source_policy_id'],
                                 **{arm: proof['policy_id'] for arm, proof in endpoint_bindings.items()}}
            for arm, proof in historical.items():
                if (arm not in common['arms'] or arm not in expected_policies
                        or proof['policy_id'] != expected_policies[arm]
                        or proof['manifest_sha256'] != cache['cache_identity']['manifest_sha256']
                        or not proof['shards']):
                    raise ValueError('historical policy or input cache identity differs')
                mounts.update(str(Path(part['path']).parent) for part in proof['shards'])
            reused_predictions[dataset] = historical
        arms = [arm for arm in common.get('arms', ('stage1', 'direct', 'tpo')) if arm not in historical]
        if historical or pooled:
            inferred_arms[dataset] = arms
        if pooled:
            continue
        for i,device in enumerate(source['devices']):
            name=f"{source['name']}-{i}"
            opts=dict(common,phase='tpo_eval',endpoint=c,devices=[device],shard=i,shards=len(source['devices']),
                trained=prior['remote_root'],output='/jev-work/'+name,batch=config['batch'],binding=binding,
                updates=steps,parent=str(parent_file),parent_sha=binding['parent_sha'])
            if historical:
                opts['arms'] = arms
            commit_json(entry/(name+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',opts))
            pieces[dataset].append(opts['output'])
            jobs.append(dict(name=name,devices=[device],**config['eval_container']))
    scoring=dict(common,phase='tpo_score',output='/jev-work/metrics.json',endpoints=sets,shards=pieces,
                 steps=dict(stage1=config['stage1_step'],direct=steps,tpo=steps,**checkpoint_steps))
    if calibration:
        scoring.update(thresholds=threshold_file, calibration_roster_sha256=calibration['roster']['sha256'])
    if reused_predictions:
        scoring.update(reused_predictions=reused_predictions, inferred_arms=inferred_arms)
    if semantic:
        scoring['semantic_scoring'] = semantic
    if config.get('publish_seals'):
        scoring['publish_seals'] = True
    block_groups = []
    if pooled:
        opts = dict(common, trained=prior['remote_root'], batch=config['batch'], binding=binding,
                    updates=steps, parent=str(parent_file), parent_sha=binding['parent_sha'])
        prep, jobs, queue = queue_jobs(config, entry, opts, datasets, sets, inferred_arms, mounts, prior['image'])
        block_groups = [[prep]]
        scoring['queue'] = queue
        scoring['inferred_arms'] = inferred_arms
        if 'parity' in queue:
            if 'choice' in common:
                raise ValueError('limited probability parity does not accept Choice panels')
            scoring['phase'] = 'tpo_eval_block_parity'
    commit_json(entry/'score.json',macros('NATIVE_SEAL_CONTAINER_CONFIG',scoring))
    archive=pkg/'entry.tar'
    with tarfile.open(archive,'x') as tar:tar.add(entry,arcname='entry')
    ctl={k:prior[k] for k in ('host','lease','entrypoint','image','driver_root','shared_root','transfer_timeout')}
    if 'training_custody' in config:
        custody = Path(config['training_custody'])
        if read(custody/'operations.json') != prior:
            raise ValueError('training custody belongs to a different sealed operation')
        terminal = custody/'terminal.json'
        predecessor = dict(terminal=str(terminal), required_status='formal_training_collected_and_audited')
        if terminal.is_file():
            proof = read(terminal)
            if (proof['exit_code'] != 0 or proof['status'] != predecessor['required_status']
                    or proof['endpoint'] != read(ops/'tpo-collected.json')['endpoint']):
                raise ValueError('matching audited training terminal required')
            predecessor['terminal_sha256'] = file_hash(terminal)
        else:
            predecessor['driver'] = read(custody/'started.json')
    elif 'collected_predecessor' in config:
        terminal = ops/'terminal.json'
        required = config['collected_predecessor']
        receipt = ops/'tpo-collected.json'
        if (file_hash(terminal) != required['terminal_sha256']
                or read(terminal) != {'exit_code': 0, 'status': 'declared_stages_finished'}
                or file_hash(receipt) != required['receipt_sha256']):
            raise ValueError('pinned completed training evidence required')
        proof = read(receipt)
        if (proof['binding'] != binding or proof['updates'] != steps or proof['passed'] is not True
                or proof['window'] != 'formal' or proof['arm'] != 'tpo'):
            raise ValueError('collected training does not match the frozen plan')
        predecessor = dict(terminal=str(terminal), terminal_sha256=required['terminal_sha256'])
    else:
        predecessor = dict(driver=read(ops.with_suffix('.started.json')), terminal=str(ops/'terminal.json'))
    groups = [*([encode_jobs, verify_jobs] if encode_jobs else []),
              *([cache_jobs] if cache_jobs else []), *calibration_groups, *block_groups, jobs,
              [dict(name='score', devices=[], collect='metrics.json', **config['eval_container'])]]
    ctl.update(config['deployment'],archive=str(archive),read_only_mounts=[p+':'+p for p in sorted(mounts)],
        device_group_lease=True,wait_for_lease=True,predecessor=predecessor,
        chains=[groups])
    if cache_store_used:
        ctl['evaluation_cache_store'] = reuse['store']
    if cache_paths:
        ctl['evaluation_cache_plan'] = cache_paths
    if config.get('publish_seals') or reuse or calibration or pooled:
        ctl['entry_archive_sha256'] = file_hash(archive)
        ctl['entry_pins'] = {f'{job["name"]}.json': file_hash(entry/f'{job["name"]}.json')
                             for group in groups for job in group}
    commit_json(pkg/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',ctl))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('local');a=p.parse_args()
    config = load_config(a.config,a.local)[0]
    if config.get('evaluation_cache_reuse'):
        config['repository'] = str(Path(__file__).resolve().parents[2])
    prepare(config)
