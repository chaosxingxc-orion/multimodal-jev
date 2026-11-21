"""Release a shutdown-failed reference only after independently replaying its last committed macro."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import width
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import prepare_context_reference_job as ref
from tools.reports.context_device_visibility import visibility
from tools.reports.context_replica_training_operations import path
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.replay_context_reference_tail import read


def tree(root, workers):
    root = Path(root)
    entries = list(root.rglob('*'))
    if root.is_symlink() or any(x.is_symlink() for x in entries):
        raise ValueError('symlink in evidence')
    files = sorted(x for x in entries if x.is_file())
    with ThreadPoolExecutor(max_workers=width(workers)) as pool:
        return dict(zip((str(x.relative_to(root)) for x in files), pool.map(file_hash, files), strict=True))


def certify_previous(job, parity, provenance):
    from tools.reports.audit_context_train_parity import audit
    p = audit(parity, provenance)
    ps = parity['parity_package']
    if (not p['comparison']['exact'] or p['comparison']['queries'] != 256
            or p['comparison']['decisions'] != 60 or p['actor_updates'] != 0
            or ps['parent'] != job['parent'] or ps.get('scoring_devices') != job.get('scoring_devices')
            or job['deployment']['host'] != parity['deployment']['host']):
        raise ValueError('previous same-host, same-parent exact TRAIN parity required')
    proof = ref.audit_attempt_outputs(job, provenance, shutdown_failure=True)
    att = read(job['launch']); transport = read(job['transport']); state = read(job['inspect'])['State']
    if (read(job['wait']) != dict(container_id=att['container_id'], exit_code=1)
            or file_hash(job['archive']) != transport['archive_sha256'] or state['Error']):
        raise ValueError('previous shutdown-failed exact collection required')
    root = Path(job['collection'])
    clean = read(root/'supervision/terminal.json')
    if (clean['child_exit_codes'] != [0]*(1+len(clean['services']))
            or not any('force killing remaining process EngineCore' in
                (root/'supervision'/('service-'+s['name']+'.log')).read_text() for s in clean['services'])):
        raise ValueError('recovery is limited to verified engine teardown after successful builder')
    build = ref.collected_audit_config(job)
    result = read(root/'reference-build/result.json')
    keys = list(dict.fromkeys(result['population'][k] for k in sorted(result['population'])))
    macro = read(build['build_config'])['macro_client']['max_queries']
    index = (len(keys)-1)//macro
    name = f'batch-{index:08d}'
    last = Path(build['journal'])/name
    sample = {}
    for uid, key in sorted(result['population'].items()):
        sample.setdefault(key, uid)
    complete = read(last/'COMPLETE.json')
    ids = [sample[key] for key in keys[index*macro:]]
    if len(ids) != len(complete['completed_indices']) or complete['batch_id'] != name:
        raise ValueError('terminal macro coverage differs')
    return proof, dict(batch_id=name, sample_ids=ids), last


def candidate(job, spec, replay):
    root = Path(job['output']); m = read(root/'PREPARED.json')
    remote = path(spec['remote_root']); old = m['remote_root']
    if Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote):
        raise ValueError('separate replay root required')
    engine = spec['engine_shutdown_timeout_seconds']; supervisor = spec['shutdown_timeout_seconds']
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (engine, supervisor))
            or not 30 < engine < supervisor):
        raise ValueError('valid teardown deadline and supervisor margin required')
    service_devices = visibility(m)[1]
    runner = ref.RUNNER.split('services=[]', 1)[0]
    runner = runner[:runner.index("c=json.loads")] + runner[runner.index("def env("):]
    runner += f"""services=[]
for name,visible in {service_devices!r}:
 service=json.loads((r/('service-'+name+'.json')).read_text())
 services.append(dict(name=name,
  command=[sys.executable,str(r/'replay-service.py'),'--config',str(r/('service-'+name+'.json')),
   '--output',str(r/('service-'+name)),'--engine-timeout',{str(engine)!r}],
  ready_path=r/('service-'+name)/'ready.json',socket_path=Path(service['socket_path']),
  env=env('service-'+name,visible)))
report=supervise_services(services=services,
 actor_command=[sys.executable,str(r/'replay-tail.py'),'--build',str(r/'reference-build-config.json'),
                '--replay',str(r/'replay.json')],
 actor_env=env('reference-client',''),output=r/'supervision',shutdown_timeout={supervisor!r})
raise SystemExit(0 if report['passed'] else 1)
"""
    compile(runner, '<reference-tail-runner>', 'exec')
    layout = dict(m['layout'])
    layout['configs'] += ['replay.json', 'replay-service.py', 'replay-tail.py']
    immutable = dict(m, status='reference_shutdown_replay_prepared_only', layout=layout, remote_root=remote)
    immutable['original_manifest_sha256'] = job['manifest_sha256']
    fixed = {name: rebase(read(root/name), old, remote) for name in
        ['config.json', 'reference-build-config.json', *m['layout']['services'].values()]}
    return immutable, fixed, runner


def prepare(job, spec, replay):
    root = Path(job['output']); out = Path(spec['package']); workers = spec['io_workers']
    check_paths({'replay package':out})
    out.mkdir(parents=True, exist_ok=False)
    m, configs, runner = candidate(job, spec, replay)
    excluded = set(configs) | {'PREPARED.json', 'run.py', 'reference-local.json'}
    for name in read(root/'PREPARED.json')['files']:
        if name not in excluded:
            dest = out/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/name, dest)
    (out/'run.py').write_text(runner)
    for target, source in (('replay-service.py','context_reference_replay_service.py'),
                           ('replay-tail.py','replay_context_reference_tail.py')):
        shutil.copyfile(Path(__file__).with_name(source), out/target)
    commit_json(out/'replay.json', replay)
    fixed = configs['config.json']
    remote = m['remote_root']
    pins = {k:v for k,v in fixed['pins'].items() if not Path(k).is_relative_to(remote)}
    pins.update({remote+'/'+k:v for k,v in tree(out, workers).items()})
    for name, cfg in configs.items():
        commit_json(out/name, dict(cfg, pins=pins))
    commit_json(out/'reference-local.json', dict(schema='jev-local-macros-v1',
        macros={'REFERENCE_DATASET_BUILD_CONFIG':dict(configs['reference-build-config.json'], pins=pins)}))
    m['files'] = tree(out, workers)
    commit_json(out/'PREPARED.json', m)
    return file_hash(out/'PREPARED.json')


def validate_replay(job, spec, replay, *, after_commit_failure=False):
    pkg = Path(spec['package']); manifest = read(pkg/'PREPARED.json')
    expected, cfgs, runner = candidate(job, spec, replay)
    files = tree(pkg, spec['io_workers'])
    if (files.pop('PREPARED.json') != spec['manifest_sha256'] or files != manifest['files']
            or manifest != dict(expected, files=files) or (pkg/'run.py').read_text() != runner
            or read(pkg/'replay.json') != replay or spec['host'] != job['deployment']['host']
            or spec['lease'] != job['deployment']['lease']):
        raise ValueError('replay package and original host/lease must match')
    original = read(Path(job['output'])/'PREPARED.json')['files']
    for name in original.keys() - set(cfgs) - {'run.py', 'reference-local.json'}:
        if files.get(name) != original[name]:
            raise ValueError('original numerical runtime or input changed: ' + name)
    for target, source in (('replay-service.py','context_reference_replay_service.py'),
                           ('replay-tail.py','replay_context_reference_tail.py')):
        current = Path(__file__).with_name(source).read_text()
        if after_commit_failure and target == 'replay-tail.py':
            current = current.replace('from multimodal_jev.local_runtime_storage import check_paths\n','')
            current = current.replace("    output = Path(build['output'])\n    check_paths({'tail replay': output})\n    output.mkdir(exist_ok=False)\n", '')
            current = current.replace("commit_json(output/'terminal.json'", "commit_json(Path(build['output'])/'terminal.json'")
        import hashlib
        if files.get(target) != hashlib.sha256(current.encode()).hexdigest():
            raise ValueError('replay entrypoint differs')
    remote = manifest['remote_root']
    cfg = read(pkg/'config.json')
    pins = {k:v for k,v in cfgs['config.json']['pins'].items() if not Path(k).is_relative_to(remote)}
    pins.update({remote+'/'+k:v for k,v in files.items() if k not in cfgs and k!='reference-local.json'})
    for name, expected_cfg in cfgs.items():
        if read(pkg/name) != dict(expected_cfg, pins=pins):
            raise ValueError('replay scoring config differs')
    return manifest, cfg, {}


def compare_macro(prior, fresh):
    prior, fresh = Path(prior), Path(fresh)
    manifest = read(prior/'INPUT.json')
    if read(fresh/'INPUT.json') != manifest or (fresh/'FAILED.json').exists():
        raise ValueError('replay input hashes, order or replica assignment differs')
    hashes={}; ordered=[None]*len(manifest['expected_indices'])
    for job in manifest['jobs']:
        name=job['rollout_id']; old=read(prior/(name+'.json')); new=read(fresh/(name+'.json'))
        if ({k:v for k,v in new.items() if k!='seconds'} != {k:v for k,v in old.items() if k!='seconds'}
                or type(new['seconds']) not in (float,int) or not math.isfinite(new['seconds']) or new['seconds']<=0):
            raise ValueError('replayed logits or probabilities differ: '+name)
        hashes[name]=digest(new)
        for index, record in zip(job['indices'], new['records'], strict=True):
            if ordered[index] is not None:
                raise ValueError('duplicate replay index')
            ordered[index]=record
    complete=read(fresh/'COMPLETE.json'); old=read(prior/'COMPLETE.json')
    if (any(row is None for row in ordered) or complete['result_sha256']!=digest(ordered)
            or hashes!=complete['rollout_hashes']
            or any(complete[k]!=v for k,v in old.items() if k not in ('seconds','rollout_hashes'))):
        raise ValueError('replay journal commit differs')
    return complete['result_sha256']


def audit_recovery(job, provenance):
    from tools.reports.context_cached_storage_operations import command, audit_container
    recovery=job['shutdown_recovery']; spec=recovery['execution']; parity=recovery['parity']
    after_commit_failure = recovery.get('mode') == 'missing_terminal_directory_after_commit'
    if recovery.get('mode') not in (None, 'missing_terminal_directory_after_commit'):
        raise ValueError('unknown reference recovery evidence mode')
    original={k:v for k,v in job.items() if k!='shutdown_recovery'}
    proof, replay, end=certify_previous(original, parity, provenance)
    m,cfg,_=validate_replay(original,spec,replay,after_commit_failure=after_commit_failure)
    stem=Path(spec['operations_root'])/spec['tag']
    att=read(str(stem)+'-launch.json'); observed=read(str(stem)+'-inspect.json')
    transport=read(str(stem)+'-collected.transport.json'); root=Path(str(stem)+'-collected')
    argv=command(spec,m,cfg,producer=True)
    expected_exit = int(after_commit_failure)
    audit_container(argv,cfg,att,observed,producer=True,expected_exit=expected_exit)
    if (att['manifest_sha256']!=spec['manifest_sha256'] or att['layout']!=m['layout']
            or att['config_sha256']!=file_hash(Path(spec['package'])/'config.json')
            or att['services']!={k:file_hash(Path(spec['package'])/v) for k,v in m['layout']['services'].items()}
            or read(str(stem)+'-wait.json')!=dict(container_id=att['container_id'],exit_code=expected_exit)
            or transport['container_id']!=att['container_id'] or transport['image']!=att['image']
            or transport['remote_root']!=m['remote_root'] or transport['state']!=observed['State']
            or transport['status']!='transport_complete_independent_audit_required'
            or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))
            or file_hash(str(stem)+'-collected.tar')!=transport['archive_sha256']
            or tree(root,spec['io_workers'])!=transport['files']):
        raise ValueError('complete exact replay transport required')
    terminal=read(root/'supervision/terminal.json')
    if (terminal['passed'] is not (not after_commit_failure) or terminal['forced_cleanup'] is not False
            or terminal['actor_started'] is not True or terminal.get('error') is not None
            or terminal['actor_exit_code']!=expected_exit
            or terminal['child_exit_codes'] != [0]*len(m['inference_services'])+[expected_exit]
            or [x['name'] for x in terminal['services']] != list(m['layout']['services'])
            or any(x['exit_code']!=0 for x in terminal['services'])):
        raise ValueError('clean replay shutdown is required')
    for s in m['layout']['services']:
        log=(root/'supervision'/('service-'+s+'.log')).read_text()
        if (read(root/('service-'+s)/'terminal.json')['exit_code'] != 0
                or 'force killing remaining' in log or 'MPClient: complete' not in log):
            raise ValueError('clean EngineCore teardown is required: '+s)
    n=len(replay['sample_ids'])
    if after_commit_failure:
        log=(root/'supervision/actor.log').read_text()
        if ((root/'reference-build').exists() or log.count('Traceback (most recent call last):')!=1
                or "commit_json(Path(build['output'])/'terminal.json'" not in log
                or "FileNotFoundError: [Errno 2] No such file or directory: '"+m['remote_root']+"/reference-build/.terminal.json-" not in log):
            raise ValueError('expected only missing terminal directory after durable replay commit')
    elif read(root/'reference-build/terminal.json') != dict(exit_code=0,batch_id=replay['batch_id'],scored=n):
        raise ValueError('single complete replay macro required')
    build=read(root/'reference-build-config.json'); journal=root/'reference-journal'/build['macro_client']['attempt_id']
    if {x.name for x in journal.glob('batch-*')}!={replay['batch_id']}:
        raise ValueError('replay may score only one macro')
    result_sha=compare_macro(end,journal/replay['batch_id'])
    return dict(proof,status='reference_shutdown_recovered_and_audited',
        recovery=dict(container_id=att['container_id'], original_container_id=proof['container_id'],
                      replayed_queries=n, reused_queries=proof['output_audit']['distinct_inputs']-n,
                      batch_id=replay['batch_id'], exact_result_sha256=result_sha,
                      original_exit_code=1,replay_exit_code=expected_exit,
                      transport_sha256=file_hash(str(stem)+'-collected.transport.json')))


def run(c, provenance):
    from types import SimpleNamespace
    from tools.reports import context_composed_pipeline as pipeline
    from tools.reports.context_async_pipeline import ReadyQueue
    from tools.reports.context_cached_storage_operations import command
    job=c['original']; spec=dict(c['execution'])
    output=Path(c['output']); output.mkdir(parents=True, exist_ok=False)
    try:
        _, replay, _=certify_previous(job,c['parity'],provenance)
        spec['manifest_sha256']=prepare(job,spec,replay)
        bound=dict(job,shutdown_recovery=dict(execution=spec,parity=c['parity']))
        commit_json(output/'job.json',bound)
        effects=SimpleNamespace(**dict(vars(pipeline.EFFECTS),stage_command=command))
        pipeline.run_stage(spec,provenance,validate=lambda _:validate_replay(job,spec,replay),
            queue=ReadyQueue(output/'dispatch'),priority=0,producer=True,
            authorization='replay_dispatch_allowed',effects=effects)
        proof=audit_recovery(bound,provenance)
        commit_json(output/'audit.json',proof)
        commit_json(output/'terminal.json',dict(exit_code=0,status=proof['status'],**proof['recovery']))
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args()
    run(*load_config(a.config,a.local))
