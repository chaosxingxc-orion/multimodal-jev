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


def run(c):
    base = settings(c['eval_plan'])
    score_base = settings(c['eval_score'])
    cal_previous = settings(c['eval_calibration'])
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
    commit_json(entry/'parent.json', parent)
    repo = Path(__file__).resolve().parents[2]
    frozen = freeze(repo, entry/'source', patterns=c['source_patterns'], io_workers=c['io_workers'])
    source = dict(root='/jev-entry/source', sha256=frozen['manifest_sha256'])
    common = dict(source=source, parent='/jev-entry/parent.json', parent_sha=file_hash(entry/'parent.json'),
                  arms=arms, batch=c['batch'], io_workers=c['io_workers'], single_card=single,
                  trained=trained_deploy['remote_root'], updates=c['stage2_updates'], binding=binding)
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
        ports[name] = dict(cache=cache, audio_token_id=parent['audio_token_id'], use_source_cards=True)
        partitions[name] = 1
        panels[name] = dict(panels[official_spec['clean_dataset']], primary='top1_accuracy', cluster='id',
                            comparisons=[])
        for phase in ('encode','verify'):
            role = 'official-'+phase
            write(role, dict(common, phase=phase, cache=cache))
            pre.append([cpu(role, 'official-extra-cache/VERIFIED.json' if phase == 'verify' else None)])

    deploy = dict(c['deployment'])
    datasets = {name:dict(logical_partitions=n) for name,n in partitions.items()}
    inferred = {name:arms for name in datasets}
    inputs = {c['stage1_snapshot'], str(Path(trained_deploy['remote_root'])/'tpo')}
    prep, workers, queue = queue_jobs(c, entry, common, datasets, ports, inferred, inputs, deploy['image'])
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
    write('score', dict(common, phase='tpo_score', queue=queue, endpoints=ports, inferred_arms=inferred,
                        probability_panels=panels, thresholds='/jev-work/thresholds.json',
                        calibration_roster_sha256=cal['roster']['sha256'], output='/jev-work/metrics.json',
                        official_combination=official))
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',
                    entry/'native-seal-container.json')
    with tarfile.open(root/'entry.tar', 'x') as archive:
        archive.add(entry, arcname='entry')
    for port in ports.values():
        inputs.update((port['cache']['output'], port['cache']['seal']['root']))
    entry_pins = {str(p.relative_to(entry)):file_hash(p) for p in
                  [*entry.glob('*.json'), *entry.glob('official-supplement/**/*.json'), entry/'source/FILES.json']}
    deploy.update(archive=str(root/'entry.tar'), entry_archive_sha256=file_hash(root/'entry.tar'), entry_pins=entry_pins,
                  chains=[[*pre, [prep], calibration, [cpu('calibration-fit','thresholds.json')], workers,
                           [cpu('score','metrics.json')]]],
                  read_only_mounts=[p+':'+p for p in sorted(inputs)
                                    if not any(Path(p).is_relative_to(r) for r in
                                               (deploy['shared_root'], '/jev-entry','/jev-work'))],
                  wait_for_lease=True, device_group_lease=True)
    commit_json(root/'operations-local.json', macros('NATIVE_SEAL_OPERATIONS_CONFIG',deploy))


if __name__ == '__main__':
    run(load_config(*sys.argv[1:])[0])
