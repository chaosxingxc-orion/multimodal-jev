"""Freeze one paired evaluation from audited single-card training endpoints."""
import json
from pathlib import Path
import shutil
import sys
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.prepare_card_evaluation import pinned
from tools.reports.prepare_tpo_evaluation import queue_jobs
from tools.reports.prepare_tpo_windows import macros


def history_result_key(experiment):
    import re
    match = re.fullmatch(r'JEV-([0-9]+)', experiment)
    if not match:
        raise ValueError('expected a registered JEV experiment comparison')
    return 'versus_' + match[1]


def read_pinned(pin):
    return pinned(pin["path"], pin["sha256"])


def settings(pin):
    return read_pinned(pin)['macros']['NATIVE_SEAL_CONTAINER_CONFIG']


def extend_official(entry, inputs, clean):
    """Evaluate only the extra official rows; share the clean predictions."""
    name = inputs['dataset']
    prior_root = Path(clean['seal']['root'])
    prior = read_pinned(dict(path=prior_root/'MANIFEST.json', sha256=clean['seal']['sha256']))
    paths = {key: prior_root/f'{folder}/{clean["dataset"]}/{key}.json'
             for key, folder in [('panel', 'inference'), ('rows', 'inference'), ('targets', 'scoring')]}
    panel, clean_rows, clean_gold = (read_pinned(dict(path=p, sha256=prior['files'][str(p.relative_to(prior_root))]))
                                    for p in paths.values())
    official = {}
    for key in ('requests', 'targets'):
        path = Path(inputs[key]['path'])
        if file_hash(path) != inputs[key]['sha256']:
            raise ValueError('official SLURP input identity changed')
        with path.open() as stream:
            data = [json.loads(line) for line in stream]
        official[key] = {r['id']: r for r in data}
        if not len(data) == len(official[key]) == inputs['population']:
            raise ValueError('incomplete official SLURP population')
    if official['requests'].keys() != official['targets'].keys():
        raise ValueError('official requests and targets differ')
    clean_ids = {r['id'] for r in clean_rows}
    if len(clean_ids) != len(clean_rows) or not clean_ids < official['requests'].keys():
        raise ValueError('official TEST must contain the full clean subset')

    def audio_row(row):
        return dict(id=row['id'], audio=row['input']['audio'], audio_sha256=row['audio_sha256'])

    def gold(row):
        t = official['targets'][row['id']]['target']
        return t['scenario']+'_'+t['action']

    official_gold_mismatches = 0
    for row, target in zip(clean_rows, clean_gold, strict=True):
        native = official['requests'][row['id']]
        if audio_row(native) != row or target['id'] != row['id']:
            raise ValueError('official and clean input identity differs')
        official_gold_mismatches += gold(native) != target['gold']
    if official_gold_mismatches:
        raise ValueError('clean SLURP scores cannot be reused for the official metric')
    extra = [r for uid, r in official['requests'].items() if uid not in clean_ids]
    score = [dict(id=r['id'], audio_sha256=r['audio_sha256'], gold=gold(r), group=r['source_group'])
             for r in extra]
    if any(r['gold'] not in panel['candidate_ids'] for r in score):
        raise ValueError('official gold outside the frozen decision menu')
    seal = entry/'official-supplement'
    members = {f'inference/{name}/panel.json': panel, f'inference/{name}/rows.json': list(map(audio_row, extra)),
               f'scoring/{name}/targets.json': score}
    for path, payload in members.items():
        location = seal/path
        location.parent.mkdir(parents=True, exist_ok=True)
        commit_json(location, payload)
    manifest = dict(schema=prior['schema'], prompt_layout=prior['prompt_layout'], target_tuning_allowed=False,
                    datasets={name:dict(recordings=len(extra), candidates=len(panel['candidate_ids']),
                                        split='official_test_outside_clean_subset')},
                    files={path:file_hash(seal/path) for path in members})
    commit_json(seal/'MANIFEST.json', manifest)
    cache = dict(clean, dataset=name, output='/jev-work/official-extra-cache',
                 seal=dict(root='/jev-entry/official-supplement', sha256=file_hash(seal/'MANIFEST.json')))
    return cache, dict(clean=clean['dataset'], extra=name, dataset=inputs['report_dataset'],
                       population=inputs['population'])



def bind_completed_endpoints(c):
    inputs = c['completed_endpoints']
    proof_path = Path(inputs['stage1_receipt'])
    proof_pin = dict(path=str(proof_path), sha256=file_hash(proof_path))
    proof = read_pinned(proof_pin)
    training_dir = Path(inputs['stage2_package'])
    staged = training_dir/'entry'
    target = staged/'targets/MANIFEST.json'
    training = staged/'tpo.json'
    train_pin = dict(path=str(training), sha256=file_hash(training))
    bound = settings(train_pin)
    if file_hash(target) != bound['training']['target_sha']:
        raise ValueError('audited TPO target schedule changed')
    manifest = read_pinned(dict(path=str(target), sha256=bound['training']['target_sha']))
    if (manifest['source_policy'] != proof['policy_id']
            or manifest['card_input']['checkpoint_sha256'] != proof['checkpoint_sha256']):
        raise ValueError('evaluation endpoints do not belong to the same training run')
    card = proof_path.with_name('cards.json')
    cards = dict(path=str(card), sha256=manifest['card_input']['cards_sha256'])
    deployment = training_dir/'operations-local.json'
    result = dict(c, stage1_snapshot=proof['policy'], stage1_audit=proof_pin,
                  stage1_manifest=dict(path=str(proof_path.with_name('endpoint-manifest.json')), sha256=proof['manifest_sha256']),
                  cards=cards, stage2_updates=manifest['updates'], training_config=train_pin,
                  training_deployment=dict(path=str(deployment), sha256=file_hash(deployment)))
    result['completed_endpoints'] = dict(inputs, stage1_training=dict(
        path=str(proof_path.with_name('training.json')), sha256=proof['config_sha256']))
    return result

def single_card_modes(config):
    modes = {}
    projections = config.get('combined_tile_projections')
    for dataset, entry in config.get('single_card_modes', {}).items():
        mode = entry
        if isinstance(entry, dict):
            if set(entry) != {'mode', 'combined_tile_projections'} or entry['mode'] != 'reuse_audio_tiles':
                raise ValueError('invalid combined-tile mode configuration')
            mode = entry['mode']
            selected = entry['combined_tile_projections']
            if projections is not None and projections != selected:
                raise ValueError('combined-tile modes must share the frozen projection allowlist')
            projections = selected
        if mode not in ('bounded_batch', 'matched_shape', 'reuse_audio_length', 'reuse_audio_batch',
                        'reuse_audio_attention', 'reuse_audio_tiles', 'hf_singleton'):
            raise ValueError('unsupported single-card benchmark mode')
        modes[dataset] = mode
    if 'reuse_audio_tiles' in modes.values():
        if (not isinstance(projections, list) or not projections
                or not all(isinstance(name, str) and name for name in projections)
                or len(set(projections)) != len(projections)):
            raise ValueError('combined tiles require a nonempty list of distinct projection names')
    return modes, projections


def run(c):
    base = settings(c['eval_plan'])
    score_base = settings(c['eval_score'])
    cal_previous = settings(c['eval_calibration'])
    c = bind_completed_endpoints(c) if 'completed_endpoints' in c else c
    parent = read_pinned(c['old_parent'])
    policy_proof = read_pinned(c['stage1_audit'])
    endpoint = read_pinned(c['stage1_manifest'])
    terminal = read_pinned(c['training_config'])
    trained_deploy = read_pinned(c['training_deployment'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    train = terminal['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    binding = {k:train['training'][k] for k in ('target_sha','parent_sha','world_size')}
    binding['source'] = train['source']['sha256']
    receipt = json.loads((Path(trained_deploy['operations'])/'tpo-collected.json').read_bytes())
    train_status = json.loads((Path(trained_deploy['operations'])/'terminal.json').read_bytes())
    if (train_status != dict(exit_code=0, status='declared_stages_finished')
            or receipt['binding'] != binding or receipt['passed'] is not True
            or receipt['updates'] != c['stage2_updates'] or receipt['window'] != 'formal'
            or receipt['arm'] != 'tpo' or receipt['endpoint'] == endpoint['policy_id']
            or not policy_proof['passed'] or policy_proof['container_exited'] is not True
            or policy_proof['updates'] != c['stage1_updates']
            or policy_proof['manifest_sha256'] != c['stage1_manifest']['sha256']
            or policy_proof['policy_id'] != endpoint['policy_id']
            or trained_deploy['host'] != c['deployment']['host']
            or trained_deploy['lease'] != c['deployment']['lease']):
        raise ValueError('paired evaluation requires the audited training endpoints and exact host')
    arms = ['stage1', 'tpo']
    policies = dict(stage1=endpoint['policy_id'], tpo=receipt['endpoint'])
    parent.update(source_policy=c['stage1_snapshot'], source_policy_id=endpoint['policy_id'],
                  source_contract=endpoint['contract'])
    if 'completed_endpoints' in c:
        stage1 = read_pinned(c['completed_endpoints']['stage1_training'])
        parent.update(model=stage1['model'], base_id=stage1['base_id'])
    root = Path(c['output'])
    entry = root/'entry'
    entry.mkdir(parents=True)
    cards = read_pinned(c['cards'])
    if len(cards['labels']) != c['training_labels']:
        raise ValueError('wrong trained card catalog')
    for name, pin in [('cards.json', c['cards']), ('calibration-roster.json', c['calibration_roster'])]:
        if file_hash(pin['path']) != pin['sha256']:
            raise ValueError('changed calibration or card input')
        shutil.copyfile(pin['path'], entry/name)
    view = read_pinned(c['old_cards_view'])
    single = dict(cards=dict(path='/jev-entry/cards.json', sha256=c['cards']['sha256']),
                  encoding=view['encoding'], witness=view['witness'], pair_batch=c['pair_batch'],
                  token_cache_entries=c['token_cache_entries'])
    if 'max_batch_audio_chunks' in c:
        single.update(max_batch_audio_chunks=c['max_batch_audio_chunks'], parity_pairs=c['parity_pairs'])
    modes, projections = single_card_modes(c)
    if 'reuse_audio_tiles' in modes.values():
        single['combined_tile_projections'] = projections
    from multimodal_jev.card_experiment_config import reference_settings
    parent['reference_inference'] = reference_settings(c)
    commit_json(entry/'parent.json', parent)
    repo = Path(__file__).resolve().parents[2]
    frozen = freeze(repo, entry/'source', patterns=c['source_patterns'], io_workers=c['io_workers'])
    source = dict(root='/jev-entry/source', sha256=frozen['manifest_sha256'])
    common = dict(source=source, parent='/jev-entry/parent.json', parent_sha=file_hash(entry/'parent.json'),
                  reference_inference=parent['reference_inference'], arms=arms, batch=c['batch'], io_workers=c['io_workers'], single_card=single,
                  trained=trained_deploy['remote_root'], updates=c['stage2_updates'], binding=binding)
    benchmark = c.get('performance_comparison')
    if benchmark is not None:
        if (set(benchmark) != {'role', 'archive_plan'} or benchmark['role'] not in {'standard', 'candidate'}
                or not c.get('single_card_reference') or 'single_card_probe' in c
                or benchmark['role'] == 'standard' and set(modes.values()) != {'bounded_batch'}
                or set(benchmark['archive_plan']) != {'path', 'sha256'}
                or Path(benchmark['archive_plan']['path']).name != 'plan.json'
                or not Path(benchmark['archive_plan']['path']).is_absolute()):
            raise ValueError('invalid full performance comparison')
        common['performance_comparison'] = benchmark
    ports = base['endpoints']
    for name, port in ports.items():
        port.pop('card_view', None)
        port['use_source_cards'] = name in c['source_card_datasets']
    panels = {k:dict(v, comparisons=['stage1']) for k, v in score_base['probability_panels'].items()}

    def write(name, opts):
        commit_json(entry/(name+'.json'), macros('NATIVE_SEAL_CONTAINER_CONFIG', opts))

    def cpu(name, path=None):
        return dict(name=name, devices=[], **c['cache_container'], **(dict(collect=path) if path else {}))

    pre = []
    official = None
    partitions = dict(base['queue']['partitions'])
    if c.get('official_test'):
        official_spec = c['official_test']
        clean = ports[official_spec['clean_dataset']]
        cache, official = extend_official(entry, official_spec, clean['cache'])
        cache.pop('cache_identity', None)
        cache.pop('publication_key', None)
        cache['processor'] = dict(cache['processor'], policy=c['stage1_snapshot'])
        cache.update(io_workers=c['io_workers'], prefetch=c['sample_blocks']['prefetch'],
                     progress_interval=c['progress_interval'])
        name = official_spec['dataset']
        partitions[name] = 1
        panels[name] = dict(panels[official_spec['clean_dataset']], primary='top1_accuracy', cluster='id',
                            comparisons=[])
        prior = c.get('official_cache_reuse', c.get('reuse'))
        if prior:
            proof = read_pinned(prior['official_receipt'])
            if (proof['status'] != 'full_stable_prefix_cache_verified' or proof['model_calls'] != 0
                    or proof['npu_executed'] is not False
                    or proof['recordings'] != read_pinned(dict(path=entry/'official-supplement/MANIFEST.json',
                                                               sha256=cache['seal']['sha256']))['datasets'][name]['recordings']):
                raise ValueError('full official supplement verification required')
            cache.update(output=str(Path(prior['remote_root'])/'official-extra-cache'),
                         cache_identity=dict(manifest_sha256=proof['manifest_sha256'],
                                             verification_sha256=prior['official_receipt']['sha256']))
            preflight_result = 'official-cache-preflight.json'
            write('official-preflight', dict(common, phase='preflight_eval_cache', cache=cache,
                                             result_path='/jev-work/'+preflight_result))
            pre.append([cpu('official-preflight', preflight_result)])
        else:
            for phase in ('encode','verify'):
                role = 'official-'+phase
                write(role, dict(common, phase=phase, cache=cache))
                pre.append([cpu(role, 'official-extra-cache/VERIFIED.json' if phase == 'verify' else None)])

        ports[name] = dict(cache=cache, cache_verification=cache['output']+'/VERIFIED.json',
                           audio_token_id=parent['audio_token_id'], use_source_cards=True)

    if modes.keys() - ports.keys():
        raise ValueError('unknown dataset in single-card modes')
    for name, mode in modes.items():
        ports[name]['single_card_mode'] = mode

    deploy = dict(c['deployment'])
    datasets = {name:dict(logical_partitions=n) for name,n in partitions.items()}
    inferred = {name:arms for name in datasets}
    inputs = {c['stage1_snapshot'], str(Path(trained_deploy['remote_root'])/'tpo')}
    if benchmark is not None:
        inputs.add(str(Path(benchmark['archive_plan']['path']).parent))
    probe = c.get('single_card_probe')
    if probe:
        if (set(probe) - {'compare_reuse', 'trace_padding', 'trace_blocks', 'latency_rounds', 'latency_reference'} != {'plan', 'first_blocks', 'batches_per_block', 'edge_and_middle'}
                or any(type(probe[k]) is not int or probe[k] < 1
                       for k in ('first_blocks', 'batches_per_block'))
                or type(probe['edge_and_middle']) is not bool
                or set(probe['plan']) != {'path', 'sha256'}
                or not Path(probe['plan']['path']).is_absolute()):
            raise ValueError('finite probe requires a pinned failed plan and a declared sample budget')
        inputs.add(str(Path(probe['plan']['path']).parent))
    admission = c.get('single_card_reference')
    if admission:
        if probe or len(modes) != len(ports):
            raise ValueError('full single-card evaluation requires modes for every referenced dataset')
        reference = read_pinned(admission)
        if (reference['schema'] != 'jev-single-card-hf-panel-v1'
                or reference['policies'] != policies or set(reference['inputs']) != set(ports)):
            raise ValueError('independent HF evidence differs from evaluation scope')
        commit_json(entry/'hf-reference.json', reference)
        if file_hash(entry/'hf-reference.json') != admission['sha256']:
            raise ValueError('cannot reproduce the independent HF evidence')
        c = dict(c, single_card_reference=dict(path='/jev-entry/hf-reference.json', sha256=admission['sha256']))
    prep, workers, queue = queue_jobs(c, entry, common, datasets, ports, inferred, inputs, deploy['image'])
    checks = []
    finite = c.get('finite_policy_parity')
    if finite:
        if admission or probe or len(modes) != len(ports) or type(finite['batches_per_block']) is not int or finite['batches_per_block'] < 1:
            raise ValueError('explicit finite parity specification for every evaluation dataset required')
        for i, job in enumerate(workers):
            name = f'hf-check-{i}'
            opts = settings(dict(path=entry/(job['name']+'.json'),
                                 sha256=file_hash(entry/(job['name']+'.json'))))
            write(name, dict(opts, output='/jev-work/'+name, result_path='/jev-work/'+name+'.json',
                             finite_policy_parity=dict(finite, shard=i, shards=len(workers))))
            checks.append(dict(job, name=name, collect=name+'.json', visible_ordinal='0'))
    if admission:
        for i, job in enumerate(workers):
            name = f'hf-check-{i}'
            opts = settings(dict(path=entry/(job['name']+'.json'),
                                 sha256=file_hash(entry/(job['name']+'.json'))))
            write(name, dict(opts, output='/jev-work/'+name, result_path='/jev-work/'+name+'.json',
                             single_card_reference=c['single_card_reference'],
                             single_card_shard=i, single_card_shards=len(workers)))
            checks.append(dict(job, name=name, collect=name+'.json', visible_ordinal='0'))

    cal = dict(cal_previous['calibration'], policy_ids=policies, prefetch=c['sample_blocks']['prefetch'])
    if cal['roster']['sha256'] != c['calibration_roster']['sha256']:
        raise ValueError('calibration population differs')
    calibration, shards = [], []
    for i, device in enumerate(c['sample_blocks']['devices']):
        name = f'calibration-{i}'
        shards.append('/jev-work/'+name)
        write(name, dict(common, phase='tpo_eval', endpoint=cal_previous['endpoint'], calibration=cal,
                         devices=[device], shard=i, shards=len(c['sample_blocks']['devices']),
                         output=shards[-1], prefetch=cal['prefetch']))
        calibration.append(dict(name=name, devices=[device], **c['eval_container']))
    for job in calibration+workers:
        job['visible_ordinal'] = '0'
    write('calibration-fit', dict(common, phase='tpo_score', calibration=cal, calibration_fit=True,
                                 shards=shards, output='/jev-work/thresholds.json'))
    cal_groups = [calibration, [cpu('calibration-fit','thresholds.json')]]
    score = dict(common, phase='tpo_score', queue=queue, endpoints=ports, inferred_arms=inferred,
                 probability_panels=panels, thresholds='/jev-work/thresholds.json',
                 calibration_roster_sha256=cal['roster']['sha256'], output='/jev-work/metrics.json',
                 official_combination=official)
    if 'reuse' in c:
        reuse = c['reuse']
        old_deploy = read_pinned(reuse['deployment'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
        failure = read_pinned(reuse['terminal'])
        threshold = read_pinned(reuse['thresholds'])
        if (old_deploy['remote_root'] != reuse['remote_root'] or deploy['host'] != old_deploy['host']
                or deploy['image'] != old_deploy['image'] or failure['exit_code'] != 1 or failure['restart'] is not False
                or threshold['schema'] != 'jev-source-train-harness-thresholds-v1'
                or threshold['roster_sha256'] != cal['roster']['sha256']
                or {arm: fit['policy_id'] for arm, fit in threshold['arms'].items()} != policies
                or any(fit['queries'] != cal['recordings'] or not 0 <= fit['threshold'] <= 1
                       for fit in threshold['arms'].values())):
            raise ValueError('calibration reuse must match the exact evaluated policies, roster and runtime')
        commit_json(entry/'thresholds.json', threshold)
        if file_hash(entry/'thresholds.json') != reuse['thresholds']['sha256']:
            raise ValueError('collected calibration did not reproduce the committed receipt')
        score.update(thresholds='/jev-entry/thresholds.json', thresholds_sha256=reuse['thresholds']['sha256'])
        cal_groups = []
    if 'historical_comparison' in c:
        history = c['historical_comparison']
        metrics = read_pinned(history['metrics'])
        path = Path(history['archive_root'])
        if not path.is_absolute():
            raise ValueError('historical evaluation requires the absolute archived block directory')
        inputs.add(str(path))
        commit_json(entry/'historical-metrics.json',metrics)
        if file_hash(entry/'historical-metrics.json') != history['metrics']['sha256']:
            raise ValueError('historical metrics serialization differs')
        score['historical_comparison'] = dict(history, result_key=history_result_key(history['experiment']), metrics=dict(history['metrics'],path='/jev-entry/historical-metrics.json'))
    write('score', score)
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',
                    entry/'native-seal-container.json')
    if probe and cal_groups:
        raise ValueError('numerical-only replay requires reusable calibration')
    score_jobs = [] if probe else [[cpu('score','metrics.json')]]
    if checks:
        cal_groups = [checks, *cal_groups] if finite else [*cal_groups, checks]
    with tarfile.open(root/'entry.tar', 'x') as archive:
        archive.add(entry, arcname='entry')
    for port in ports.values():
        inputs.update((port['cache']['output'], port['cache']['seal']['root']))
    entry_pins = {str(p.relative_to(entry)):file_hash(p) for p in
                  [*entry.glob('*.json'), *entry.glob('official-supplement/**/*.json'), entry/'source/FILES.json']}
    deploy.update(archive=str(root/'entry.tar'), entry_archive_sha256=file_hash(root/'entry.tar'), entry_pins=entry_pins,
                  chains=[[*pre, [prep], *cal_groups, workers, *score_jobs]],
                  read_only_mounts=[p+':'+p for p in sorted(inputs)
                                    if not any(Path(p).is_relative_to(r) for r in
                                               (deploy['shared_root'], '/jev-entry','/jev-work'))],
                  wait_for_lease=True, device_group_lease=True)
    commit_json(root/'operations-local.json', macros('NATIVE_SEAL_OPERATIONS_CONFIG',deploy))


if __name__ == '__main__':
    run(load_config(*sys.argv[1:])[0])
