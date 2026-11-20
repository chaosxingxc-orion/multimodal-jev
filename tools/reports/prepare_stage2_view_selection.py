"""Prepare native evaluation for a sealed Stage2 curve or fixed held-out endpoint."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze, regular, verify
from multimodal_jev.stage2_view_runtime import pinned
from tools.reports import run_native_seal_endpoint
from tools.reports.seal_stage2_view_selection import DATASET, read

HELDOUT_SCHEMA = 'jev-stage2-view-heldout-endpoint-v1'


def heldout_plan(c, selection):
    heldout = c['heldout']
    if set(heldout) != {'dataset', 'seal', 'cache_root'}:
        raise ValueError('held-out dataset, seal and cache root are required')
    name, seal = heldout['dataset'], heldout['seal']
    manifest = pinned(Path(seal['root'])/'MANIFEST.json', seal['sha256'])
    cache_root = Path(heldout['cache_root'])
    shared = Path(c['operations']['shared_root'])
    if (manifest['schema'] != 'jev-native-evaluation-seal-v1' or manifest['target_tuning_allowed'] is not False
            or name == DATASET or name not in manifest['datasets'] or not Path(seal['root']).is_relative_to(shared)
            or not cache_root.is_absolute() or '..' in cache_root.parts or ':' in str(cache_root)
            or cache_root.is_relative_to(shared) or 'read_only_mounts' in c['operations']
            or selection['target_tuning_allowed'] is not False or selection['model_selection_performed'] is not False):
        raise ValueError('independent sealed held-out inputs and previously frozen endpoint required')
    steps = [p['step'] for p in selection['checkpoints']]
    if len(steps) < 2 or steps[0] != 0 or steps != sorted(set(steps)):
        raise ValueError('complete preregistered checkpoint roster required')
    return dict(selection, schema=HELDOUT_SCHEMA, evaluation_dataset=name,
                checkpoints=selection['checkpoints'][-1:], seal_sha256=seal['sha256'],
                fixed_terminal_step=steps[-1], source_selection_sha256=c['seal']['sha256'])


def bind_forward(snapshot, request_pin, *, image, base_id):
    """Install the training request's exact forward sources into a fresh evaluation tree."""
    from multimodal_jev.context_execution import map_io
    from multimodal_jev.source_snapshot import inventory
    request = pinned(request_pin['path'],request_pin['sha256'])['macros']['STAGE2_ENDPOINT_SUCCESSOR']
    rt = request['runtime']
    package = Path(rt['runtime']['package'])
    receipt = pinned(package/'PREPARED.json',rt['runtime']['manifest_sha256'])
    source = package/'source'
    members = pinned(package/'RUNTIME-SOURCES.json',receipt['files']['RUNTIME-SOURCES.json'])
    workers = rt['model_binding']['io_workers']
    design = request['source']
    spec = pinned(design['design'],design['design_sha256'])
    if (not members or any(not k.endswith('.py') for k in members)
            or receipt['image'] != image or
            set(members) != {str(p.relative_to(source)) for p in source.rglob('*.py')}
            or inventory(source,members,io_workers=workers) != members):
        raise ValueError('forward image or complete source inventory differs')
    prior = request['predecessor']['config']
    final = pinned(prior['path'],prior['sha256'])
    ops,_ = load_config(final['operations_config'],final['operations_local'])
    parent = Path(ops['package'])
    prepared = pinned(parent/'PREPARED.json',ops['manifest_sha256'])
    train = pinned(parent/'config.json',prepared['files']['config.json'])
    if (request['view_pipeline']['trace_replay']['image'] != image or train['base_id'] != base_id
            or spec['stage2']['reuse_stage1_adapter'] is not True):
        raise ValueError('shared endpoint adapter and admitted image required')
    def install(name):
        target = snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(regular(source,name),target)
    map_io(install,sorted(members),workers=workers)
    if inventory(snapshot,members,io_workers=workers) != members:
        raise ValueError('evaluation copy differs from frozen forward sources')
    return dict(attention_execution=rt['attention_execution'],lora_arithmetic=rt['lora_arithmetic']), spec


def verify_point(root, point, plan, spec):
    from multimodal_jev.context_stage_lora import from_policy, normalize
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    policy = root/f"policy-{point['step']}"
    adapter = root/f"adapter-{point['step']}"
    if from_policy(plan['source_contract']) != normalize(spec['stage2']['lora']):
        raise ValueError('evaluation adapter geometry differs from training recipe')
    bundle = read_execution_bundle(policy,expected_contract=plan['source_contract'],
        expected_base_id=plan['base_id'],expected_policy_id=point['policy_id'])
    verify_native_adapter(adapter,bundle)


def prepare_hf_successor(c, provenance):
    """Build an isolated HF-only package from a sealed endpoint and its HF panel."""
    prior = Path(c['prior_entry'])
    pins = {k: pinned(pin['path'], pin['sha256']) for k, pin in c['pins'].items()}
    panel = pins['panel_role']['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    full = pins['full_role']['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    endpoint, full_endpoint = panel['endpoint'], full['endpoint']
    if (panel['phase'] != 'hf' or full['phase'] != 'full' or panel['source'] != full['source']
            or any(endpoint[k] != full_endpoint[k] for k in (
                'point', 'source_plan', 'snapshot', 'model', 'base_id', 'contract', 'audio_token_id'))):
        raise ValueError('same frozen finite and full endpoint configurations required')
    workers = panel['io_workers']
    source_files = verify(prior/'source', manifest_sha256=panel['source']['sha256'], io_workers=workers)
    plan = pinned(prior/'source-plan.json', endpoint['source_plan']['sha256'])
    run_native_seal_endpoint.validate_hf_endpoint(plan, endpoint['point'], endpoint['cache'])
    dataset = endpoint['cache']['dataset']
    seal = endpoint['cache']['seal']
    policy = endpoint['point']['policy_id']
    reference_panel, reference = (pins[key] for key in ('reference_panel', 'reference'))
    if (full_endpoint['cache']['dataset'] != dataset or full_endpoint['cache']['seal'] != seal
            or full_endpoint['cache'].get('panel_size') is not None
            or reference.get('schema') != 'jev-target-width-hf-reference-v1'
            or reference_panel['dataset'] != dataset or reference['policy_id'] != policy
            or not len(reference['records']) == len(reference_panel['assets']) == endpoint['cache']['panel_size']
            or reference['panel_sha256'] != digest(reference_panel)):
        raise ValueError('original same-policy HF reference and matching finite and full input seals required')
    devices = c['devices']
    if (not devices or any(type(i) is not int or i < 0 for i in devices)
            or devices != sorted(set(devices)) or len(devices) > len(reference_panel['assets'])):
        raise ValueError('distinct sorted HF devices required')
    operations = c['operations']
    expected_image = 'sha256:'+full_endpoint['macro_client']['runtime_identity']['image_sha256']
    if operations['image'] != expected_image:
        raise ValueError('same frozen evaluation image required')
    comparison = pins.get('comparison')
    if comparison is not None and (comparison['exact'] is not False or comparison['dataset'] != dataset
            or comparison['seal_sha256'] != seal['sha256'] or comparison['policy_id'] != policy
            or comparison['queries'] != len(reference_panel['assets'])
            or comparison['panel_sha256'] != reference['panel_sha256']
            or comparison['runtime_identity'] != full_endpoint['macro_client']['runtime_identity']):
        raise ValueError('failed service parity must match the frozen HF evidence')

    root = Path(c['output'])
    check_paths({'HF successor': root})
    entry = root/'entry'
    entry.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(dir=root) as temp:
        source = Path(temp)
        for name in source_files:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(regular(prior/'source', name), path)
        if verify(prior/'source', manifest_sha256=panel['source']['sha256'], io_workers=workers) != source_files:
            raise ValueError('previous source snapshot differs')
        shutil.copyfile(run_native_seal_endpoint.__file__, source/'tools/reports/run_native_seal_endpoint.py')
        identity = freeze(source, entry/'source', patterns=['**/*'], io_workers=workers)
    for name in ('source-plan.json', 'native-seal-container.json'):
        shutil.copyfile(prior/name, entry/name)
    if file_hash(entry/'source-plan.json') != endpoint['source_plan']['sha256']:
        raise ValueError('copied native source plan differs')
    snapshot_path = Path(endpoint['snapshot'])
    if snapshot_path.parent != Path('/jev-entry'):
        raise ValueError('isolated native policy directory required')
    shutil.copytree(prior/snapshot_path.name, entry/snapshot_path.name)
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    snapshot = entry/snapshot_path.name
    if file_hash(snapshot/'MANIFEST.json') != endpoint['point']['manifest_sha256']:
        raise ValueError('copied native terminal policy differs')
    read_execution_bundle(snapshot, expected_contract=endpoint['contract'],
                          expected_base_id=endpoint['base_id'], expected_policy_id=policy)

    shards = dict(devices=devices)
    for key in ('reference_panel', 'reference'):
        path = entry/(key+'.json')
        pin = c['pins'][key]
        shutil.copyfile(pin['path'], path)
        if file_hash(path) != pin['sha256']:
            raise ValueError('copied HF reference evidence differs')
        shards[key] = dict(path='/jev-entry/'+path.name, sha256=pin['sha256'])

    def role(name, phase, prior_endpoint, scope, index=None):
        spec = dict(shards, index=index) if index is not None else shards
        cfg = dict(source=dict(panel['source'], sha256=identity['manifest_sha256']),
                   io_workers=workers, phase=phase, endpoint=dict(
                       prior_endpoint, hf_shards=spec, physical_devices=scope, output='/jev-work/'+name))
        commit_json(entry/(name+'.json'), dict(schema='jev-local-macros-v1',
                                               macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=cfg)))
        return dict(name=name, devices=scope, **c['resources'][phase])

    groups = [[role(f'{phase}-{i}', phase, config, [device], i) for i, device in enumerate(devices)]
              for phase, config in (('hf-shard-check', endpoint), ('hf-shard-full', full_endpoint))]
    audit = role('audit-hf-shards', 'audit-hf-shards', full_endpoint, [])
    groups.append([dict(audit, collect='audit-hf-shards/metrics.json')])
    return write_prepared(root, entry, dict(operations, device_group_lease=True, wait_for_lease=True),
                          groups, [], dict(schema='jev-native-hf-successor-prepared-v1', dataset=dataset,
                                           policy_id=policy, source=endpoint['source_plan'],
                                           evaluation_devices=devices, provenance=provenance))


def write_prepared(root, entry, operations, stages, other, receipt):
    archive_path = root/'entry.tar.gz'
    with tarfile.open(archive_path, 'x:gz') as target:
        target.add(entry, arcname='entry')
    operations_root = root/'run'
    operations = dict(operations, archive=str(archive_path), operations=str(operations_root),
                      chains=[stages, other] if other else [stages],
                      predecessor_terminal=str(operations_root/'terminal.json'))
    local = root/'operations-local.json'
    commit_json(local, dict(schema='jev-local-macros-v1', macros=dict(NATIVE_SEAL_OPERATIONS_CONFIG=operations)))
    result = dict(receipt, local=str(local), archive_sha256=file_hash(archive_path))
    commit_json(root/'PREPARED.json', result)
    return result


def prepare(c, provenance):
    if 'hf_successor' in c:
        return prepare_hf_successor(c['hf_successor'], provenance)
    root = Path(c['output'])
    check_paths({'evaluation operations':root})
    root.mkdir(parents=True, exist_ok=False)
    entry = root/'entry'
    entry.mkdir()
    workers = c['io_workers']
    seal_root = Path(c['seal']['root'])
    seal = pinned(seal_root/'MANIFEST.json', c['seal']['sha256'])
    if set(seal['datasets']) != {DATASET}:
        raise ValueError('source selection seal required')
    for rel, digest in seal['files'].items():
        path = seal_root/rel
        if (not path.resolve().is_relative_to(seal_root.resolve())
                or path.is_symlink() or file_hash(path) != digest):
            raise ValueError('selection seal changed: '+rel)
    plan = pinned(seal_root/'source-plan.json', seal['files']['source-plan.json'])
    if plan['schema'] != 'jev-stage2-view-selection-source-v1':
        raise ValueError('audited Stage2 source plan required')
    heldout = c.get('heldout')
    if heldout:
        plan = heldout_plan(c, plan)
        plan_path = entry/'source-plan.json'
        commit_json(plan_path, plan)
        source = dict(path='/jev-entry/source-plan.json', sha256=file_hash(plan_path))
        cache_seal = heldout['seal']
    else:
        shutil.copytree(seal_root, entry/'selection')
        source = dict(path='/jev-entry/selection/source-plan.json', sha256=seal['files']['source-plan.json'])
        cache_seal = dict(root='/jev-entry/selection',sha256=c['seal']['sha256'])
    old = c['runtime_archive']
    if file_hash(old['path']) != old['sha256']:
        raise ValueError('native runtime archive changed')
    with TemporaryDirectory(dir=root) as temp:
        staging = Path(temp)
        with tarfile.open(old['path']) as archive:
            members = [m for m in archive.getmembers() if m.name.startswith('entry/source/')]
            archive.extractall(staging, members=members, filter='data')
        snapshot = staging/'entry/source'
        verify(snapshot, manifest_sha256=c['runtime_source_sha256'], io_workers=workers)
        for p in [snapshot, *snapshot.rglob('*')]:
            p.chmod(0o755 if p.is_dir() else 0o644)
        (snapshot/'FILES.json').unlink()
        forward_options, forward_design = {}, None
        if 'training_request' in c:
            forward_options, forward_design = bind_forward(snapshot,c['training_request'],
                image=c['operations']['image'],base_id=plan['base_id'])
        shutil.copyfile(run_native_seal_endpoint.__file__, snapshot/'tools/reports/run_native_seal_endpoint.py')
        identity = freeze(snapshot, entry/'source', patterns=['**/*'], io_workers=workers)
    files = read(entry/'source/FILES.json')
    runtime_file = entry/'RUNTIME-SOURCES.json'
    commit_json(runtime_file, {k:v for k,v in files.items() if k.endswith('.py')})
    runtime = dict(c['runtime_identity'], source_manifest_sha256=file_hash(runtime_file))
    service_template = pinned(c['service_template']['path'], c['service_template']['sha256'])
    if (service_template['dual_runtime_identity']['processor_sha256'] != runtime['processor_sha256']
            or service_template['dual_runtime_identity']['image_sha256'] != runtime['image_sha256']
            or service_template['policy']['base_id'] != plan['base_id']):
        raise ValueError('same frozen native evaluator, image and processor required')
    if c['operations']['image'] != 'sha256:'+runtime['image_sha256']:
        raise ValueError('same frozen evaluation image required')
    devices = c['devices']
    if not devices or devices != sorted(set(devices)) or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError('ordered distinct evaluation devices required')
    for point in plan['checkpoints']:
        step = point['step']
        shutil.copytree(point['snapshot'], entry/f'policy-{step}')
        shutil.copytree(point['adapter'], entry/f'adapter-{step}')
        if file_hash(entry/f'policy-{step}'/'MANIFEST.json') != point['manifest_sha256']:
            raise ValueError('copied policy changed')

        if forward_design is not None:
            verify_point(entry,point,plan,forward_design)

    def macro(path, key, value):
        commit_json(path, dict(schema='jev-local-macros-v1', macros={key:value}))

    def role(name, phase, phase_devices=(), collect=None, **config):
        value = dict(source=dict(root='/jev-entry/source', sha256=identity['manifest_sha256']),
                     io_workers=workers, phase=phase, **config)
        macro(entry/(name+'.json'), 'NATIVE_SEAL_CONTAINER_CONFIG', value)
        job = dict(name=name, devices=list(phase_devices), **c['resources'][phase])
        return dict(job,collect=collect) if collect else job

    initial = plan['checkpoints'][0]['step']
    cache = dict(c['cache'], dataset=plan['evaluation_dataset'], seal=cache_seal,
                 processor=dict(policy=f'/jev-entry/policy-{initial}',sha256=runtime['processor_sha256']))
    cache_mount = '/jev-inputs' if heldout else '/jev-work'
    panel = dict(cache, output=cache_mount+'/panel-cache', panel_size=c['panel_size'])
    full = dict(cache, output=cache_mount+'/full-cache')
    stages, other = [], []
    hf_shards = c.get('hf_shards')
    if hf_shards is not None:
        if not heldout or set(hf_shards) != {'reference_panel', 'reference'}:
            raise ValueError('direct HF requires both pinned held-out reference files')
        point = plan['checkpoints'][0]
        spec = dict(devices=devices)
        for key, pin in hf_shards.items():
            pinned(pin['path'], pin['sha256'])
            path = entry/(key+'.json')
            shutil.copyfile(pin['path'], path)
            if file_hash(path) != pin['sha256']:
                raise ValueError('copied HF evidence differs')
            spec[key] = dict(path='/jev-entry/'+path.name, sha256=pin['sha256'])
        common = dict(source_plan=source, point=point, contract=plan['source_contract'], base_id=plan['base_id'],
                      model=service_template['policy']['model'], snapshot=f"/jev-entry/policy-{point['step']}",
                      audio_token_id=cache['encoding']['audio_token_id'], torch_threads=c['torch_threads'])
        for phase, input_cache in (('hf-shard-check', panel), ('hf-shard-full', full)):
            stages.append([role(f'{phase}-{idx}', phase, [device], endpoint=dict(
                common, hf_shards=dict(spec, index=idx), physical_devices=[device], cache=input_cache,
                cache_verification=input_cache['output']+'/VERIFIED.json', output=f'/jev-work/{phase}-{idx}'))
                for idx, device in enumerate(devices)])
        stages.append([role('audit-hf-shards', 'audit-hf-shards', collect='audit-hf-shards/metrics.json',
                            endpoint=dict(common, hf_shards=spec, physical_devices=[], cache=full,
                            cache_verification=full['output']+'/VERIFIED.json', output='/jev-work/audit-hf-shards'))])
    if not heldout:
        stages = [[role('panel-encode','encode',cache=panel)], [role('panel-verify','verify',cache=panel)]]
        other = [[role('full-encode','encode',cache=full)], [role('full-verify','verify',cache=full)]]
    for offset in range(0, len(plan['checkpoints']) if hf_shards is None else 0, len(devices)):
        group = []
        for idx, point in enumerate(plan['checkpoints'][offset:offset+len(devices)]):
            step = point['step']
            cfg = dict(source_plan=source, point=point, contract=plan['source_contract'], base_id=plan['base_id'],
                       model=service_template['policy']['model'], snapshot=f'/jev-entry/policy-{step}',
                       audio_token_id=cache['encoding']['audio_token_id'], torch_threads=c['torch_threads'])
            cfg.update(cache=panel, cache_verification=panel['output']+'/VERIFIED.json',
                       output=f'/jev-work/hf-{step}', physical_devices=[devices[idx]])
            group.append(role(f'hf-{step}', 'hf', [devices[idx]], endpoint=cfg))
        stages.append(group)

    def score_jobs(point, phase, cache_cfg):
        step = point['step']
        name = f'{phase}-{step}'
        snapshot = f'/jev-entry/policy-{step}'
        sockets, services = [], []
        scope = dict(source_root='/jev-entry/source', source_policy=snapshot,
                     dual_runtime_manifest='/jev-entry/RUNTIME-SOURCES.json', dual_runtime_identity=runtime)
        for idx, device in enumerate(devices):
            rid = f'r{idx}'
            ipc = f'/jev-work/{name}-ipc/{rid}'
            sockets.append(dict(socket=ipc+'/s', exchange=ipc+'/x'))
            item = copy.deepcopy(service_template)
            item.update(forward_options)
            item.update(scope, socket_path=ipc+'/s', exchange=ipc+'/x',
                        pins={'/jev-entry/source/'+k:v for k,v in files.items()})
            item['policy'].update(snapshot=snapshot, policy_id=point['policy_id'],
                adapter=f'/jev-entry/adapter-{step}', contract=plan['source_contract'],
                physical_devices=[device], process_device_scope=dict(
                    kind='same_container_runtime_visibility_v1', mounted_physical=devices, visible_ordinal=str(idx)))
            service_file = f'{name}-service-{idx}.json'
            commit_json(entry/service_file, item)
            services.append(dict(name=rid, config='/jev-entry/'+service_file, output=f'/jev-work/{name}-service-{idx}'))
        client = dict(c['client'], schema='jev-replica-macro-client-v1', attempt_id=name,
                      journal_root='/jev-work/journal', replicas=sockets, runtime_identity=runtime)
        out = '/jev-work/'+name
        cfg = dict(scope, source_plan=source, point=point, contract=plan['source_contract'], base_id=plan['base_id'],
                   model=service_template['policy']['model'], snapshot=snapshot, physical_devices=devices,
                   audio_token_id=cache['encoding']['audio_token_id'], torch_threads=c['torch_threads'],
                   cache=cache_cfg, cache_verification=cache_cfg['output']+'/VERIFIED.json',
                   macro_client=client, macro_queries=client['max_queries'], output=out)
        evidence = (dict(hf_reference=f'/jev-work/hf-{step}/reference.json') if phase == 'parity'
                    else dict(parity_receipt=f'/jev-work/audit-parity-{step}/comparison.json'))
        produce = role(name,phase,devices,endpoint=cfg,evidence=evidence,services=services,
                       supervision=out+'-supervision')
        audit = role('audit-'+name,'audit-'+phase,collect='audit-'+name+(
                     '/comparison.json' if phase=='parity' else '/metrics.json'),
                     endpoint=dict(cfg,inference=out+'/inference',output='/jev-work/audit-'+name),evidence=evidence)
        return [[produce],[audit]]

    # CPU cache and the finite lane overlap. Endpoint full jobs await the full cache.
    for point in plan['checkpoints'] if hf_shards is None else []:
        stages.extend(score_jobs(point,'parity',panel))
    operations_root = root/'run'
    for point in plan['checkpoints'] if hf_shards is None else []:
        groups = score_jobs(point,'full',full)
        if not heldout:
            groups[0][0]['wait_for'] = str(operations_root/'full-verify-wait.json')
        stages.extend(groups)
    (entry/'native-seal-container.json').write_text('"${NATIVE_SEAL_CONTAINER_CONFIG}"\n')
    operations = dict(c['operations'])
    if heldout:
        operations['read_only_mounts'] = [f"{heldout['cache_root']}/{name}:{cache_mount}/{name}"
                                         for name in ('panel-cache', 'full-cache')]
    return write_prepared(root, entry, operations, stages, other, dict(
        schema='jev-stage2-view-heldout-prepared-v1' if heldout else 'jev-stage2-view-selection-prepared-v1',
        source=source, evaluation_devices=devices, provenance=provenance))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    cfg, provenance = load_config(a.config, a.local)
    print(json.dumps(prepare(cfg, provenance)))
