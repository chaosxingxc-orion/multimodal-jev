"""Finite acceptance launch for any declared replica training topology.

Validates the sealed package, declared device set, storage, lease and pins
before creating one container that runs the declared inference services and the
data-parallel trainer. Never stops an existing job, retries a launch, or admits
full training.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.prepare_context_dual_throughput_pair import sha, write
from tools.reports.launch_context_batch_probe_candidate import wait_exact
from tools.reports.context_replica_trainer_candidate import ALGORITHM
from tools.jev.collect_context_replica_training import collect

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def path(value):
    if (type(value) != str or not re.fullmatch(r'/[a-zA-Z0-9_./-]+', value)
            or '..' in Path(value).parts or str(Path(value)) != value or value == '/'):
        raise ValueError('explicit canonical absolute runtime path required')
    return value


def release_evidence(c, config):
    """The frozen release verifies its own publication pins at the absolute host
    paths it recorded. Those paths live outside the candidate, so a declared
    relocation root supplies byte-identical copies inside the container."""
    value = c.get('release_evidence')
    if value is None:
        return None
    if set(value) != {'root', 'destination'}:
        raise ValueError('release evidence relocation requires root and destination')
    root = path(value['root']); destination = value['destination']
    if (path(destination) != destination or destination == '/'
            or not Path(root).is_dir()):
        raise ValueError('canonical release evidence root and destination required')
    publication = Path(path(config['release'])) / 'publication-config.json'
    pins = json.loads(publication.read_text())['pins']
    relocated = {name: digest for name, digest in pins.items()
                 if Path(name).is_relative_to(destination)}
    if not relocated:
        raise ValueError('release publishes no pins under the evidence destination')
    for name, digest in relocated.items():
        candidate = Path(root) / Path(name).relative_to(destination)
        if not candidate.is_file() or sha(candidate) != digest:
            raise ValueError('relocated release evidence differs: ' + name)
    return str(Path(root)), destination


def experiment_window(config, plan):
    """The admitted window is data: an engineering prefix or a declared formal
    source interval derived from the schedule, never a per-topology constant."""
    kind = config.get('experiment_kind', 'engineering_prefix')
    if kind == 'formal_source':
        from tools.reports.context_replica_formal_candidate import contract
        contract(config)
        return kind
    if (kind != 'engineering_prefix' or config['engineering_prefix'] is not True
            or config['training_dispatch_allowed'] is not False
            or config['start_update'] != 0 or config['stop_update'] != 2
            or config.get('resume_root') or config['checkpoint_interval'] != 1
            or config['stop_update'] > plan['total_updates']):
        raise ValueError('isolated matched declared initial prefix required')
    return kind


def validate(c):
    package = Path(c['package']); m = json.loads((package / 'PREPARED.json').read_text())
    if sha(package / 'PREPARED.json') != c['manifest_sha256']:
        raise ValueError('prepared manifest identity differs')
    actual = {str(p.relative_to(package)): sha(p) for p in package.rglob('*')
              if p.is_file() and p != package / 'PREPARED.json'}
    if any(p.is_symlink() for p in package.rglob('*')) or actual != m['files']:
        raise ValueError('complete immutable package required')
    remote = path(m['remote_root']); config = json.loads((package / 'config.json').read_text())
    layout = m['layout']; plan = derive(config['batch_config'])
    if plan != m['batch_plan']:
        raise ValueError('package batch plan differs from its configuration')
    kind = experiment_window(config, plan)
    expected_status = 'replica_formal_prepared_only' if kind == 'formal_source' else 'replica_integrated_prepared_only'
    world = plan['dp_replicas']
    service_names = list(layout['services'])
    services = {name: json.loads((package / layout['services'][name]).read_text())
                for name in service_names}
    identity = config['dual_runtime_identity']
    if (m['status'] != expected_status or m['image'] != IMAGE
            or config['global_batch'] != plan['global_batch']
            or config['microbatch'] != plan['configuration']['microbatch']
            or config['physical_devices'] != m['training_devices']
            or len(config['physical_devices']) != world
            or config['algorithm'] != ALGORITHM
            or config['dual_arm'] != 'joint'
            or config['source_root'] != remote + '/source'
            or config['schedule'] != remote + '/source/schedule'
            or config['schedule_sha256'] != m['schedule_sha256']
            or sha(package / 'source/schedule/MANIFEST.json') != m['schedule_sha256']
            or config['parent_schedule_sha256'] != sha(package / 'source/parent-schedule/MANIFEST.json')
            or config['reference_cache'] != remote + '/reference-cache'):
        raise ValueError('isolated matched declared initial prefix required')
    replicas = [dict(socket=remote + '/service-' + name + '/socket',
                     exchange=remote + '/service-' + name + '/exchange')
                for name in service_names]
    pool = dict(config['macro_client'])
    if (pool.get('schema') != 'jev-replica-macro-client-v1'
            or pool.get('journal_root') != remote + '/run/pool-journal'
            or pool.get('runtime_identity') != identity
            or pool.get('replicas') != replicas
            or pool.get('max_queries') != plan['macro_queries']
            or not 1 <= pool.get('rollout_queries', 0) <= plan['macro_queries']):
        raise ValueError('declared replica rollout pool differs')
    mounted = sorted([*m['training_devices'], *[s['device'] for s in m['inference_services']]])
    if len(m['inference_services']) != len(service_names):
        raise ValueError('inference service count differs')
    for binding in m['inference_services']:
        name = binding['name']; service = services[name]; policy = service['policy']
        expected_scope = dict(kind='same_container_runtime_visibility_v1',
                              mounted_physical=mounted,
                              visible_ordinal=str(mounted.index(binding['device'])))
        module = policy.get('device_scope_module')
        if (service['source_root'] != remote + '/source'
                or service['socket_path'] != remote + '/service-' + name + '/socket'
                or service['exchange'] != remote + '/service-' + name + '/exchange'
                or service['dual_runtime_identity'] != identity
                or service['dual_runtime_manifest'] != remote + '/RUNTIME-SOURCES.json'
                or policy['physical_devices'] != [binding['device']]
                or policy.get('process_device_scope') != expected_scope
                or (len(mounted) > 1 and (not isinstance(module, dict)
                    or module.get('name') != 'context_configurable_device_scope'
                    or module.get('sha256') != sha(package / 'source/tools/jev/context_configurable_device_scope.py')))
                or (len(mounted) == 1 and 'device_scope_module' in policy)
                or policy['max_batch_queries'] != plan['configuration']['inference_batch_queries']
                or policy['max_batch_tokens'] <= policy['max_model_len']
                or policy['policy_id'] != config['source_policy_id']
                or service['pins'] != config['pins']):
            raise ValueError('matched isolated inference service required: ' + name)
    if len({(s['policy']['snapshot'], s['policy']['adapter']) for s in services.values()}) != 1:
        raise ValueError('all inference services must start from the same policy')
    if (config['dual_runtime_manifest'] != remote + '/RUNTIME-SOURCES.json'
            or sha(package / 'RUNTIME-SOURCES.json') != identity['source_manifest_sha256']):
        raise ValueError('common authenticated runtime required')
    runtime = json.loads((package / 'RUNTIME-SOURCES.json').read_text())
    if runtime != {k.removeprefix('source/'): v for k, v in actual.items()
                   if k.startswith('source/') and k.endswith('.py')}:
        raise ValueError('runtime source coverage differs')
    for name, digest in config['pins'].items():
        path(name)
        if name.startswith(remote + '/') and actual.get(name[len(remote) + 1:]) != digest:
            raise ValueError('internal dependency pin differs')
    if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*', c['host']):
        raise ValueError('safe host alias required')
    if not re.fullmatch('[a-z0-9][a-z0-9-]*', c['tag']):
        raise ValueError('safe unique tag required')
    evidence = release_evidence(c, config)
    if evidence and (Path(evidence[1]).is_relative_to(remote)
                     or Path(remote).is_relative_to(evidence[1])):
        raise ValueError('release evidence destination must be disjoint from the run root')
    return m, config, services


def command(c, m, config):
    remote = m['remote_root']; shared = path(c['shared_root']); driver = path(c['driver_root'])
    initial = str(Path(path(config['source_policy'])).parent)
    cache = path(config['encoded_audio_cache']); mirror = path(config['audio_mirror_root'])
    mounts = [(remote, remote, False), (remote + '/exchange', '/jev-exchange', False),
              (remote + '/home', '/root', False), (remote + '/cache', '/jev-tmp', False),
              (shared, shared, True), (driver, '/usr/local/Ascend/driver', True),
              (initial, initial, True), (str(Path(cache).parent), str(Path(cache).parent), True),
              (cache, cache, False), (mirror, mirror, True)]
    for name in m['layout']['configs'] + list(m['layout']['services'].values()) + ['source']:
        mounts.append((remote + '/' + name, remote + '/' + name, True))
    evidence = release_evidence(c, config)
    if evidence:
        if (Path(evidence[1]).is_relative_to(remote) or Path(remote).is_relative_to(evidence[1])):
            raise ValueError('release evidence destination must be disjoint from the run root')
        mounts.append((evidence[0], evidence[1], True))
    for name in config['pins']:
        if not any(Path(name).is_relative_to(dst) for _, dst, _ in mounts):
            raise ValueError('dependency outside explicit mounts: ' + name)
    if any(Path(src).is_relative_to(remote) or Path(remote).is_relative_to(src)
           for src in (shared, initial, str(Path(cache).parent), mirror, driver)):
        raise ValueError('runtime and reusable inputs must be disjoint')
    devices = sorted([*m['training_devices'], *[s['device'] for s in m['inference_services']]])
    argv = ['docker', 'create', '--pull=never', '--network=none', '--name', c['tag'],
            '--cpus=24', '--memory=128g', '--shm-size=8g', '--log-driver=json-file',
            '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2']
    for device in devices:
        argv += ['--device', f'/dev/davinci{device}:/dev/davinci{device}']
    for name in ('davinci_manager', 'devmm_svm', 'hisi_hdc'):
        argv += ['--device', '/dev/' + name + ':/dev/' + name]
    for src, dst, ro in mounts:
        argv += ['-v', src + ':' + dst + (':ro' if ro else '')]
    env = dict(PYTHONPATH=remote + '/source:' + remote + '/source/src:' + remote + '/source/tools/jev',
               PYTHONDONTWRITEBYTECODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
               HCCL_DETERMINISTIC='true', TMPDIR='/jev-tmp')
    for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
        env[key] = '/jev-tmp'
    for key, value in env.items():
        argv += ['-e', key + '=' + value]
    return argv + ['--entrypoint', '/opt/multimodal-jev/runtime-entrypoint.sh', IMAGE,
                   'python', remote + '/run.py']


def launch(c, provenance):
    m, config, _ = validate(c); argv = command(c, m, config)
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('finite training dispatch not enabled')
    base = Path(c['operations_root']); base.mkdir(parents=True, exist_ok=True)
    with (base / (c['tag'] + '.lock')).open('a') as owner, Path(c['lease']).open('r') as lease:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = base / (c['tag'] + '-attempt.json'); handle = base / (c['tag'] + '-launch.json')
        if attempt.exists() or handle.exists():
            raise ValueError('existing attempt: reconcile exact handle')
        check_paths({'operations': base})
        validate(c)
        def remote(args, bounded=True):
            return subprocess.run(['ssh', '-o', 'ConnectTimeout=10', c['host'], shlex.join(args)],
                                  capture_output=True, text=True, timeout=55 if bounded else None)
        def checked(args):
            result = remote(args)
            if result.returncode:
                raise RuntimeError(result.stderr)
            return result.stdout
        if json.loads(checked(['docker', 'image', 'inspect', IMAGE]))[0]['Id'] != IMAGE:
            raise ValueError('fixed image missing')
        usage = checked(['npu-smi', 'info'])
        for device in sorted([*m['training_devices'], *[s['device'] for s in m['inference_services']]]):
            if f'No running processes found in NPU {device}' not in usage:
                raise ValueError(f'leased NPU {device} occupied; retain existing work')
        root = m['remote_root']; package = Path(c['package'])
        storage = (package / 'source/src/multimodal_jev/local_runtime_storage.py').read_text()
        readonly_cache = (config['encoded_audio_cache']+':'+config['encoded_audio_cache']+':ro') in argv
        locations = {'run': root} if readonly_cache else {'run': root, 'cache': config['encoded_audio_cache']}
        check = storage + '\ncheck_paths(' + repr(locations) + ')\n'
        if readonly_cache:
            check += 'assert Path(' + repr(config['encoded_audio_cache']) + ').is_dir()\n'
        check += 'assert not Path(' + repr(root) + ').exists()\n'
        external = {k: v for k, v in config['pins'].items() if not k.startswith(root + '/')}
        check += ('import hashlib\nfor name,digest in ' + repr(external) + '.items():\n'
                  ' assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name\n')
        checked(['python3', '-c', check])
        archive = base / (c['tag'] + '-package.tar.gz')
        with tarfile.open(archive, 'x:gz') as tar:
            for p in sorted(package.rglob('*')):
                if p.is_file():
                    tar.add(p, arcname=str(p.relative_to(package)), recursive=False)
        write(attempt, dict(config=c, provenance=provenance, argv=argv, archive_sha256=sha(archive)))
        checked(['mkdir', '-p', str(Path(root).parent)])
        checked(['mkdir', root])
        for name in ('exchange', 'cache', 'home'):
            checked(['mkdir', root + '/' + name])
        subprocess.run(['scp', str(archive), c['host'] + ':' + root + '/package.tar.gz'],
                       check=True, capture_output=True, timeout=55)
        checked(['tar', '-xzf', root + '/package.tar.gz', '-C', root])
        script = 'from pathlib import Path\nimport hashlib,json\nr=Path(' + repr(root) + ')\n'
        script += 'assert hashlib.sha256((r/"PREPARED.json").read_bytes()).hexdigest()==' + repr(c['manifest_sha256']) + '\n'
        script += ('m=json.loads((r/"PREPARED.json").read_text())\n'
                   'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m["files"].items())\n')
        checked(['python3', '-c', script])
        cid = checked(argv).strip()
        if not re.fullmatch('[0-9a-f]{64}', cid):
            raise ValueError('invalid container handle')
        record = dict(container_id=cid, image=IMAGE, remote_root=root, host=c['host'], argv=argv,
                      manifest_sha256=c['manifest_sha256'],
                      config_sha256=sha(package / 'config.json'), layout=m['layout'],
                      services={name: sha(package / path_) for name, path_ in m['layout']['services'].items()})
        write(handle, record)
        try:
            response = remote(['docker', 'start', cid])
            write(base / (c['tag'] + '-start-response.json'),
                  dict(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        except subprocess.TimeoutExpired:
            write(base / (c['tag'] + '-start-response.json'), dict(observation_timeout=True))
        code = wait_exact(remote, cid)
        write(base / (c['tag'] + '-wait.json'), dict(container_id=cid, exit_code=code))
        inspected = json.loads(checked(['docker', 'inspect', cid]))[0]
        write(base / (c['tag'] + '-inspect.json'), inspected)
        result = collect(record, base / (c['tag'] + '-collected'), host=c['host'])
        write(base / (c['tag'] + '-collection.json'), result)
        from tools.reports.context_dual_probe_operations import check_runtime
        check_runtime(dict(argv=argv, image=IMAGE), record, inspected)
        return dict(container_id=cid, exit_code=code,
                    status='collected_independent_training_audit_required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    cfg, prov = load_config(args.config, args.local)
    if args.launch:
        result = launch(cfg, prov)
    else:
        m, config, _ = validate(cfg); argv = command(cfg, m, config)
        result = dict(status='validated_local_only', argv=argv,
                      finite_training_dispatch_allowed=cfg.get('finite_training_dispatch_allowed'))
    print(json.dumps(result, indent=2))
