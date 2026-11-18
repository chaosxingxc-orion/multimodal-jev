"""Explicit composition of the four-card controller path.

The frozen controller reaches its behaviour by editing source text at run time:
two mount decisions in the shared Docker command builder, four anchors in the
shared launch lifecycle, and three function-local imports in the selection
curve. An anchor that stops matching fails at run time; a matched anchor that
silently changes meaning does not fail at all.

This module writes the same lifecycle out as ordinary code with every seam as a
parameter. Nothing is parsed, rewritten or compiled from text. The frozen
helpers it delegates to are unchanged; only the code that used to be rewritten
is now written, so a reviewer reads the behaviour instead of reconstructing it
from three string edits.

Scope this round is construction: ``validate`` admits a frozen controller
config, and ``run`` refuses to dispatch. ``tools/reports/audit_context_composed
_pipeline`` drives this module and the frozen rewrite on the same frozen inputs
and writes the differential report; dispatch is authorised only after that
report is accepted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.jev.collect_context_replica_training import collect
from tools.reports import context_replica_training_operations as shared
from tools.reports.context_cached_storage_operations import (
    audit_mounts, mounts, overlaps, storage_preflight)
from tools.reports.context_dual_probe_operations import check_runtime
from tools.reports.launch_context_batch_probe_candidate import wait_exact
from tools.reports.prepare_context_dual_throughput_pair import sha, write

SCHEMA = 'context-composed-pipeline-v1'
path = shared.path

# Every external interaction the lifecycle performs, in one place. The
# differential audit swaps this whole namespace for recording doubles; the
# production path uses the frozen implementations unchanged.
EFFECTS = SimpleNamespace(subprocess=subprocess, tarfile=tarfile, fcntl=fcntl,
                          check_paths=shared.check_paths, write=write, sha=sha,
                          collect=collect, check_runtime=check_runtime,
                          wait_exact=wait_exact)


def read(path):
    return json.loads(Path(path).read_text())


def bind(fn, **names):
    """Return ``fn`` with named globals replaced; the code object is unchanged.

    Unlike a source rewrite this cannot silently change meaning: the bytecode is
    the frozen function's own, so a renamed global becomes a missing name and
    fails here instead of dispatching the wrong auditor. Injecting a local is
    refused rather than ignored.
    """
    code = fn.__code__
    for name in names:
        if name in code.co_varnames:
            raise ValueError('cannot inject a local name: ' + name)
        if name not in code.co_names:
            raise ValueError('frozen function does not reference: ' + name)
    return FunctionType(code, {**fn.__globals__, **names})


def mount_rows(c, m, config, *, producer):
    """The bind mounts, with the two storage decisions as parameters.

    The frozen builder writes the cache read-only and lists the packaged
    endpoint under ``run/initial`` in the run-root disjointness check, which is
    nested inside the run root by construction and therefore always fires. Both
    are decisions, not mechanics: CPU collection needs no writable cache, and a
    training consumer mounts the reference publication read-only while a
    producer must not see one at all.
    """
    remote = m['remote_root']
    shared_root = path(c['shared_root'])
    driver = path(c['driver_root'])
    initial = str(Path(path(config['source_policy'])).parent)
    cache = path(config['encoded_audio_cache'])
    mirror = path(config['audio_mirror_root'])
    reference = None if producer else path(config['reference_dataset']['root'])
    if initial != remote + '/initial':
        raise ValueError('packaged current endpoint under run/initial required')
    if reference is not None and overlaps(reference, remote):
        raise ValueError('reference publication must be disjoint from writable runtime')
    rows = [(remote, remote, False), (remote + '/exchange', '/jev-exchange', False),
            (remote + '/home', '/root', False), (remote + '/cache', '/jev-tmp', False),
            (shared_root, shared_root, True), (driver, '/usr/local/Ascend/driver', True),
            (initial, initial, True),
            (str(Path(cache).parent), str(Path(cache).parent), True),
            (cache, cache, True), (mirror, mirror, True)]
    if reference is not None:
        rows.append((reference, reference, True))
    for name in m['layout']['configs'] + list(m['layout']['services'].values()) + ['source']:
        rows.append((remote + '/' + name, remote + '/' + name, True))
    evidence = shared.release_evidence(c, config)
    if evidence:
        if (Path(evidence[1]).is_relative_to(remote)
                or Path(remote).is_relative_to(evidence[1])):
            raise ValueError('release evidence destination must be disjoint from the run root')
        rows.append((evidence[0], evidence[1], True))
    for name in config['pins']:
        if not any(Path(name).is_relative_to(dst) for _, dst, _ in rows):
            raise ValueError('dependency outside explicit mounts: ' + name)
    reusable = [shared_root, str(Path(cache).parent), mirror, driver]
    if reference is not None:
        reusable.append(reference)
    if any(Path(src).is_relative_to(remote) or Path(remote).is_relative_to(src)
           for src in reusable):
        raise ValueError('runtime and reusable inputs must be disjoint')
    return rows


def command(c, m, config, *, producer=False):
    """The Docker create argv for one stage; identical to the frozen builder."""
    devices = sorted([*m['training_devices'], *[s['device'] for s in m['inference_services']]])
    remote = m['remote_root']
    argv = ['docker', 'create', '--pull=never', '--network=none', '--name', c['tag'],
            '--cpus=24', '--memory=128g', '--shm-size=8g', '--log-driver=json-file',
            '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2']
    for device in devices:
        argv += ['--device', f'/dev/davinci{device}:/dev/davinci{device}']
    for name in ('davinci_manager', 'devmm_svm', 'hisi_hdc'):
        argv += ['--device', '/dev/' + name + ':/dev/' + name]
    for source, destination, readonly in mount_rows(c, m, config, producer=producer):
        argv += ['-v', source + ':' + destination + (':ro' if readonly else '')]
    env = dict(PYTHONPATH=remote + '/source:' + remote + '/source/src:' + remote + '/source/tools/jev',
               PYTHONDONTWRITEBYTECODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
               HCCL_DETERMINISTIC='true', TMPDIR='/jev-tmp')
    for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
        env[key] = '/jev-tmp'
    for key, value in env.items():
        argv += ['-e', key + '=' + value]
    argv += ['--entrypoint', '/opt/multimodal-jev/runtime-entrypoint.sh', shared.IMAGE,
             'python', remote + '/run.py']
    if config.get('resume_root'):
        resume = path(config['resume_root'])
        if (producer or resume != remote + '/resume' or config.get('start_update') != 1
                or config.get('stop_update') != 2):
            raise ValueError('packaged first-step recovery input required')
        for name in ('checkpoint-0001-committed.json', 'rank-0/checkpoint-0001.pt',
                     'rank-1/checkpoint-0001.pt', 'policy-0001/MANIFEST.json'):
            if resume + '/' + name not in config['pins']:
                raise ValueError('recovery asset pin missing: ' + name)
        index = argv.index('--entrypoint')
        argv[index:index] = ['-v', resume + ':' + resume + ':ro']
    # Source binding/audit receipts at the package top are dependencies too;
    # protect each otherwise writable pinned file under the run-root mount.
    for name in config['pins']:
        parents = [row for row in mounts(argv) if Path(name).is_relative_to(row[1])]
        if parents and not max(parents, key=lambda row: len(Path(row[1]).parts))[2]:
            index = argv.index('--entrypoint')
            argv[index:index] = ['-v', name + ':' + name + ':ro']
    audit_mounts(argv, config, producer=producer)
    return argv


def stage_launcher(validate, queue, priority, *, producer, authorization, effects=EFFECTS):
    """One stage's ``launch(c, provenance)`` with every seam bound."""

    def launch(c, provenance):
        return run_stage(c, provenance, validate=validate, queue=queue, priority=priority,
                         producer=producer, authorization=authorization, effects=effects)
    return launch


def run_stage(c, provenance, *, validate, queue, priority, producer, authorization,
              effects=EFFECTS):
    """One authorize → launch → collect lifecycle with every seam injected.

    Mirrors the frozen lifecycle except in the four places the frozen controller
    edits source text: the storage preflight is the parameterised one, admission
    runs once before ownership instead of twice, a failed container aborts the
    queue, and collection runs after the lease is released.
    """
    if c.get(authorization) is not True:
        raise ValueError('explicit dispatch authorization required: ' + authorization)
    queue.check()
    try:
        result = _launch(dict(c, finite_training_dispatch_allowed=True), provenance,
                         validate=validate, queue=queue, priority=priority,
                         producer=producer, effects=effects)
        if result['exit_code'] != 0:
            raise ValueError('NPU execution failed; collection preserved')
        status = ('transfer_collected_independent_audit_required'
                  if authorization == 'transfer_dispatch_allowed'
                  else 'selection_collected_independent_audit_required' if producer
                  else 'collected_independent_cached_training_audit_required')
        return dict(result, status=status)
    except BaseException:
        queue.abort()
        raise


def _launch(c, provenance, *, validate, queue, priority, producer, effects):
    m, config, _ = validate(c)
    argv = command(c, m, config, producer=producer)
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('finite training dispatch not enabled')
    base = Path(c['operations_root'])
    base.mkdir(parents=True, exist_ok=True)
    with (queue.slot(priority, c['tag']), (base / (c['tag'] + '.lock')).open('a') as owner,
          Path(c['lease']).open('r') as lease):
        effects.fcntl.flock(owner, effects.fcntl.LOCK_EX | effects.fcntl.LOCK_NB)
        effects.fcntl.flock(lease, effects.fcntl.LOCK_EX | effects.fcntl.LOCK_NB)
        attempt = base / (c['tag'] + '-attempt.json')
        handle = base / (c['tag'] + '-launch.json')
        if attempt.exists() or handle.exists():
            raise ValueError('existing attempt: reconcile exact handle')
        effects.check_paths({'operations': base})

        def remote(args, bounded=True):
            return effects.subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=10', c['host'], shlex.join(args)],
                capture_output=True, text=True, timeout=55 if bounded else None)

        def checked(args):
            result = remote(args)
            if result.returncode:
                raise RuntimeError(result.stderr)
            return result.stdout

        if json.loads(checked(['docker', 'image', 'inspect', shared.IMAGE]))[0]['Id'] != shared.IMAGE:
            raise ValueError('fixed image missing')
        usage = checked(['npu-smi', 'info'])
        for device in sorted([*m['training_devices'],
                              *[s['device'] for s in m['inference_services']]]):
            if f'No running processes found in NPU {device}' not in usage:
                raise ValueError(f'leased NPU {device} occupied; retain existing work')
        root = m['remote_root']
        package = Path(c['package'])
        storage = (package / 'source/src/multimodal_jev/local_runtime_storage.py').read_text()
        check = storage + '\n' + storage_preflight(config, root, producer=producer)
        checked(['python3', '-c', check])
        archive = base / (c['tag'] + '-package.tar.gz')
        with effects.tarfile.open(archive, 'x:gz') as tar:
            for entry in sorted(package.rglob('*')):
                if entry.is_file():
                    tar.add(entry, arcname=str(entry.relative_to(package)), recursive=False)
        effects.write(attempt, dict(config=c, provenance=provenance, argv=argv,
                                    archive_sha256=effects.sha(archive)))
        checked(['mkdir', '-p', str(Path(root).parent)])
        checked(['mkdir', root])
        for name in ('exchange', 'cache', 'home'):
            checked(['mkdir', root + '/' + name])
        effects.subprocess.run(['scp', str(archive), c['host'] + ':' + root + '/package.tar.gz'],
                               check=True, capture_output=True, timeout=55)
        checked(['tar', '-xzf', root + '/package.tar.gz', '-C', root])
        script = 'from pathlib import Path\nimport hashlib,json\nr=Path(' + repr(root) + ')\n'
        script += ('assert hashlib.sha256((r/"PREPARED.json").read_bytes()).hexdigest()=='
                   + repr(c['manifest_sha256']) + '\n')
        script += ('m=json.loads((r/"PREPARED.json").read_text())\n'
                   'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m["files"].items())\n')
        checked(['python3', '-c', script])
        cid = checked(argv).strip()
        if not re.fullmatch('[0-9a-f]{64}', cid):
            raise ValueError('invalid container handle')
        record = dict(container_id=cid, image=shared.IMAGE, remote_root=root, host=c['host'],
                      argv=argv, manifest_sha256=c['manifest_sha256'],
                      config_sha256=effects.sha(package / 'config.json'), layout=m['layout'],
                      services={name: effects.sha(package / path_)
                                for name, path_ in m['layout']['services'].items()})
        effects.write(handle, record)
        try:
            response = remote(['docker', 'start', cid])
            effects.write(base / (c['tag'] + '-start-response.json'),
                          dict(returncode=response.returncode, stdout=response.stdout,
                               stderr=response.stderr))
        except effects.subprocess.TimeoutExpired:
            effects.write(base / (c['tag'] + '-start-response.json'),
                          dict(observation_timeout=True))
        code = effects.wait_exact(remote, cid)
        effects.write(base / (c['tag'] + '-wait.json'),
                      dict(container_id=cid, exit_code=code))
        inspected = json.loads(checked(['docker', 'inspect', cid]))[0]
        effects.write(base / (c['tag'] + '-inspect.json'), inspected)
        if code != 0:
            queue.abort()
    # Collection and runtime proof read already-collected evidence; neither needs
    # NPU ownership, so both run after the lease and the queue slot are released.
    result = effects.collect(record, base / (c['tag'] + '-collected'), host=c['host'])
    effects.write(base / (c['tag'] + '-collection.json'), result)
    effects.check_runtime(dict(argv=argv, image=shared.IMAGE), record, inspected)
    return dict(container_id=cid, exit_code=code,
                status='collected_independent_training_audit_required')


def operation(module, ops, provenance, queue, priority, *, producer=False,
              authorization, effects=EFFECTS):
    launch = stage_launcher(module.validate, queue, priority, producer=producer,
                            authorization=authorization, effects=effects)
    return bind(module.run, launch=launch)(ops, provenance)


def locations(c, name):
    return dict(package=str(Path(c['prepared_root'])/name),
                operations_root=str(Path(c['operations_root'])/name),
                remote_root=str(Path(c['remote_root'])/name),
                tag=c['tag_prefix'] + '-' + name)


def selection_operations(preparation):
    """The selection point lifecycle with the audit engine's preparation injected."""
    from tools.reports import context_replica_selection_operations as point
    validate = bind(point.validate, preparation=preparation)
    launch = bind(point.launch, validate=validate)
    audit = bind(point.audit, preparation=preparation)
    run = bind(point.run, launch=launch, audit=audit)
    return SimpleNamespace(validate=validate, launch=launch, audit=audit, run=run)


def selection_collector(curve, *, point, reconstruct_plan):
    """The curve publication bound to the composing reconstructor."""
    reconstruct = bind(curve.reconstruct, point=point, reconstruct_plan=reconstruct_plan)
    return reconstruct, SimpleNamespace(
        collect=lambda c, p: curve.publication.collect(c, p, reconstructor=reconstruct))


def selection_curve(c, template, steps, provenance, queue, *, preparation, point, curve,
                    operation, grpo=False):
    """One published curve; the three injected names are the frozen rewrite target."""
    if grpo:
        from tools.reports import prepare_context_cached_grpo_selection as preparation
        from tools.reports import context_cached_grpo_selection_operations as point
        from tools.reports import collect_context_cached_grpo_curve as curve
    prefix = 'grpo' if grpo else 'infonce'
    root = Path(c['output'])/prefix
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'template.json', template)

    def one(step):
        queue.check()
        where = locations(c, f'{prefix}-{step:04d}')
        spec = dict(template, source_step=step, output=where['package'],
                    remote_root=where['remote_root'])
        existing = c.get('preprepared_stage1_endpoint') if not grpo and step == steps[-1] else None
        if existing:
            # Immutable, never previously launched endpoint package. New
            # operations records remain isolated; audit all bytes on reuse.
            where.update(package=existing['package'], remote_root=existing['remote_root'])
            spec.update(output=where['package'], remote_root=where['remote_root'],
                        manifest_sha256=existing['manifest_sha256'])
            preparation.audit(spec, provenance)
        else:
            spec['manifest_sha256'] = preparation.prepare(spec, provenance)['manifest_sha256']
        ops = dict(c['deployment'], selection_package=spec,
                   manifest_sha256=spec['manifest_sha256'],
                   **{k: where[k] for k in ('package', 'operations_root', 'tag')},
                   selection_dispatch_allowed=True)
        proof = operation(point, ops, provenance, queue, 0, producer=True,
                          authorization='selection_dispatch_allowed')
        expected = ('complete_replica_cached_grpo_selection_audited' if grpo
                    else 'complete_replica_infonce_selection_audited')
        if (proof['status'] != expected or proof['source_step'] != step
                or proof['runtime_provenance_verified'] is not True):
            raise ValueError('full independent selection point audit required')
        path_ = Path(where['operations_root'])/'finalization/audit-config.json'
        entry = dict(step=step, audit_config=str(path_), audit_config_sha256=file_hash(path_))
        commit_json(root/f'completed-{step:04d}.json', entry)
        return entry
    entries = []
    # Endpoint first delivers the most useful baseline early; the published curve
    # remains in the preregistered order and uses every registered point.
    order = [steps[-1], *steps[:-1]]
    with ThreadPoolExecutor(max_workers=c['evaluation_workers'],
                           thread_name_prefix=prefix) as pool:
        futures = {pool.submit(one, step): step for step in order}
        try:
            for future in as_completed(futures):
                entries.append(future.result())
        except BaseException:
            queue.abort()
            for future in futures:
                future.cancel()
            raise
    entries.sort(key=lambda entry: steps.index(entry['step']))
    cc = dict(template, evaluations=entries, output=str(root/'curve'))
    commit_json(root/'curve-config.json', cc)
    result = curve.collect(cc, provenance)
    commit_json(root/'terminal.json',
                dict(exit_code=0, status=result['status'], points=len(entries)))
    return cc, result


def finite_and_efficiency(c, old, provenance, queue, *, effects=EFFECTS):
    from tools.reports import context_cached_acceptance_sequence as finite
    from tools.reports import context_cached_training_operations as training
    from tools.reports import context_cached_recovery_operations as recovery
    from tools.reports import context_cached_efficiency as efficiency
    from tools.reports.context_cached_efficiency_successor import cost_config
    from tools.reports.context_cached_efficiency_cost import run as report_cost
    from tools.reports.context_release_deployment import inherited
    prior = old['finite']
    ref = read(prior['reference_predecessor']['audit_config'])
    refops = read(prior['reference_predecessor']['config'])
    prefix_root = Path(prior['stages']['prefix']['operations_root'])/'finalization'
    prefix = read(prefix_root/'audit-config.json')
    # Reconstruct evidence independently; a completed prefix is not rerun on NPU.
    proof = training.audit(prefix, provenance)
    if proof['status'] != finite.STATUSES['prefix']:
        raise ValueError('prefix admission failed')
    evidence = {'prefix': prefix}
    for name, module in [('tail', training), ('recovery', recovery)]:
        proxy = SimpleNamespace(
            launch=stage_launcher(module.validate, queue, 10, producer=False,
                                  authorization='finite_training_dispatch_allowed',
                                  effects=effects),
            audit=module.audit)
        bound, proof = bind(finite.run_stage,
                            training=proxy if name == 'tail' else training,
                            recovery=proxy if name == 'recovery' else recovery)(
            name, locations(c, name), refops, ref, prefix, provenance)
        evidence[name] = bound
    stage = locations(c, 'efficiency')
    root = Path(c['output'])/'efficiency'
    root.mkdir(parents=True, exist_ok=False)
    spec = dict(parent=prefix['training_package']['parent'],
                reference_job=prefix['training_package']['reference_job'],
                acceptance=evidence, acceptance_kind='efficiency', output=stage['package'],
                remote_root=stage['remote_root'])
    commit_json(root/'preparation-config.json', spec)
    spec['manifest_sha256'] = efficiency.package_action(spec, provenance)['manifest_sha256']
    ops = dict(inherited(refops), training_package=spec, reference_audit=ref,
               manifest_sha256=spec['manifest_sha256'],
               **{k: stage[k] for k in ('package', 'operations_root', 'tag')},
               efficiency_dispatch_allowed=True)
    commit_json(root/'deployment.json', ops)
    stage_launcher(efficiency.validate, queue, 20, producer=False,
                   authorization='efficiency_dispatch_allowed', effects=effects)(ops, provenance)
    bound = finite.audit_config(ops, ref)
    commit_json(root/'audit-config.json', bound)
    proof = efficiency.audit(bound, provenance)
    if proof['status'] != 'cached_100step_efficiency_outputs_audited':
        raise ValueError('full100 efficiency audit required')
    commit_json(root/'audit.json', proof)
    measure = cost_config(bound, root/'cost.json')
    commit_json(root/'cost-config.json', measure)
    report_cost(measure, provenance)
    commit_json(root/'terminal.json', dict(exit_code=0, status=proof['status'], updates=100))
    return bound


def formal_training(c, bound, source_curve, provenance, queue, *, effects=EFFECTS):
    from tools.reports import context_cached_formal as formal
    from tools.reports.context_release_deployment import inherited
    from tools.reports.context_cached_formal_successor import monitored_training
    prior = bound['training_package']
    where = locations(c, 'formal')
    root = Path(c['output'])/'formal'
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'source-admission.json',
                dict(curve=source_curve, stage1_endpoint_initialization=True))
    spec = dict(parent=prior['parent'], reference_job=prior['reference_job'],
                efficiency_audit=bound, acceptance_kind='formal_source',
                evaluation_steps=list(formal.STEPS), output=where['package'],
                remote_root=where['remote_root'])
    commit_json(root/'preparation-config.json', spec)
    spec['manifest_sha256'] = formal.package_action(spec, provenance)['manifest_sha256']
    ops = dict(inherited(bound['deployment']), training_package=spec,
               reference_audit=bound['reference_audit'],
               manifest_sha256=spec['manifest_sha256'],
               **{k: where[k] for k in ('package', 'operations_root', 'tag')},
               formal_training_dispatch_allowed=True)
    commit_json(root/'deployment.json', ops)
    proxy = SimpleNamespace(run=lambda o, p: operation(
        formal, o, p, queue, 20, authorization='formal_training_dispatch_allowed',
        effects=effects))
    result = bind(monitored_training, formal=proxy)(ops, provenance, root)
    path_ = Path(ops['operations_root'])/'finalization/audit-config.json'
    finalization = dict(training_audit_config=str(path_),
                        training_audit_config_sha256=file_hash(path_))
    return finalization, ops, result


def transfers(c, old, source_cc, source_result, provenance, queue, *, effects=EFFECTS):
    from tools.reports import context_replica_transfer_sequence as sequence
    from tools.reports.context_cached_grpo_transfer_source import admit
    source_eval = source_result['source_evaluation']
    bindings = {}
    common = None
    for role in sequence.ROLES:
        request = dict(source_evaluation=source_eval, source_curve_config=source_cc,
                       policy_role=role, policy_id=source_result['roles'][role]['policy_id'])
        proof = admit(request)
        if proof['binding'] != source_result['roles'][role]:
            raise ValueError('source role binding differs')
        current = proof['source_operations']
        if common is not None and current != common:
            raise ValueError('different source training')
        common = current
        bindings[role] = request
    parent = common['training_package']['parent']
    matrices = {}

    def one(dataset, slug, role, index):
        binding = bindings[role]
        item = old['transfer']['datasets'][dataset]
        where = locations(c, f'{slug}-role-{index}')
        spec = dict(parent=parent, source_admission=binding, dataset=dataset,
                    portable_full_inputs=item['portable_full_inputs'],
                    portable_parity_inputs=item['portable_parity_inputs'],
                    output=where['package'], remote_root=where['remote_root'])
        if dataset == 'SLURP_clean_TEST60':
            spec['source_release'] = item['original_evaluation']['source_release']
        spec['manifest_sha256'] = sequence.preparation.prepare(spec, provenance)['manifest_sha256']
        ops = dict(c['deployment'], transfer_package=spec,
                   manifest_sha256=spec['manifest_sha256'],
                   **{k: where[k] for k in ('package', 'operations_root', 'tag')},
                   transfer_dispatch_allowed=True)
        result = operation(sequence.operations, ops, provenance, queue, 0, producer=True,
                           authorization='transfer_dispatch_allowed', effects=effects)
        if (result['status'] != 'complete_replica_transfer_audited'
                or result['target_scored'] is not False):
            raise ValueError('complete unscored transfer inference required')
        return read(Path(ops['operations_root'])/'finalization/audit-config.json')
    # Submit every unique frozen policy once; target gold is scored only after
    # all inference is complete and independently audited.
    with ThreadPoolExecutor(max_workers=c['evaluation_workers']) as pool:
        futures = {}
        for dataset, slug in zip(sequence.DATASETS, sequence.SLUGS, strict=True):
            for index, role in enumerate(sequence.ROLES):
                key = (dataset, bindings[role]['policy_id'])
                if key not in futures:
                    futures[key] = pool.submit(one, dataset, slug, role, index)
        try:
            executions = {key: future.result() for key, future in futures.items()}
        except BaseException:
            queue.abort()
            for future in futures.values():
                future.cancel()
            raise
    for dataset, slug in zip(sequence.DATASETS, sequence.SLUGS, strict=True):
        item = old['transfer']['datasets'][dataset]
        entries = {role: dict(policy_id=bindings[role]['policy_id'],
                              operations=executions[(dataset, bindings[role]['policy_id'])])
                   for role in sequence.ROLES}
        spec = dict(entries=entries, original_evaluation=item['original_evaluation'],
                    scoring=item['scoring'],
                    output=str(Path(c['output'])/'transfer-scores'/slug))
        matrices[dataset] = sequence.scoring.collect(spec, provenance)
    return matrices


def validate(c):
    """The frozen admission chain, unchanged; this module only composes below it."""
    from tools.reports.context_audit_optimized_pipeline import validate as admission
    return admission(c)


def retire(c, old):
    from tools.reports.context_audit_optimized_pipeline import retire as replacement
    return replacement(c, old)


def run(c, provenance, *, effects=EFFECTS, dispatch=False):
    """Compose the whole controller. Dispatch stays refused in this candidate."""
    if dispatch or c.get('dispatch_allowed') is not False:
        raise ValueError('composed pipeline is construction-only this round; '
                         'publish a versioned successor after the differential is accepted')
    from tools.reports import collect_context_replica_infonce_curve as curve_module
    from tools.reports import context_async_pipeline as frozen
    from tools.reports.context_infonce_selection import STEPS
    from tools.reports.context_shared_source_audit import SourceAuditEngine
    engine = SourceAuditEngine(cpu_workers=c['cpu_workers'], io_workers=c['io_workers'])
    old = validate(c)
    root = Path(c['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', c)
    commit_json(root/'provenance.json', provenance)
    queue = frozen.ReadyQueue(root/'events')
    try:
        retire(c, old)
        preparation = engine.preparation()
        point = selection_operations(preparation)
        reconstruct, collector = selection_collector(
            curve_module, point=point, reconstruct_plan=engine.reconstruct)
        template = old['selection']['point_template']
        with ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix='composed-branches') as branches:
            selection = branches.submit(selection_curve, c, template, STEPS, provenance, queue,
                                        preparation=preparation, point=point, curve=collector,
                                        operation=operation, grpo=False)
            efficiency = branches.submit(finite_and_efficiency, c, old, provenance, queue,
                                         effects=effects)
            results = {}
            try:
                for future in as_completed([selection, efficiency]):
                    results[future] = future.result()
            except BaseException:
                queue.abort()
                raise
        source_cc, source_result = results[selection]
        finalization, ops, _ = formal_training(c, results[efficiency], source_cc, provenance,
                                               queue, effects=effects)
        from tools.reports import context_cached_grpo_source_plan as source
        from tools.reports.context_cached_formal import STEPS as GRPO_STEPS
        plan = source.build(finalization, provenance)
        plan_path = root/'grpo-source-plan.json'
        commit_json(plan_path, plan)
        template = dict(parent=ops['training_package']['parent'], source_plan=str(plan_path),
                        source_plan_sha256=file_hash(plan_path),
                        source_finalization=finalization,
                        selection_inputs=template['selection_inputs'])
        cc, result = selection_curve(c, template, GRPO_STEPS, provenance, queue,
                                     preparation=preparation, point=point, curve=collector,
                                     operation=operation, grpo=True)
        result = transfers(c, old, cc, result, provenance, queue, effects=effects)
        commit_json(root/'results.json', result)
        commit_json(root/'terminal.json',
                    dict(exit_code=0, status='composed_pipeline_completed', restart=False))
        return result
    except BaseException as exc:
        queue.abort()
        commit_json(root/'terminal.json',
                    dict(exit_code=1, error=repr(exc), restart=False, preserve_handles=True))
        raise
    finally:
        commit_json(root/'audit-engine-statistics.json',
                    dict(batch_computations=engine.store.executions,
                         batch_reuses=engine.store.hits, events=engine.events,
                         model_calls=0))


def detach(template, local):
    c, _ = load_config(template, local)
    validate(c)
    root = Path(c['output']).with_name(Path(c['output']).name + '-driver')
    shared.check_paths({'driver': root})
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', '-m', 'tools.reports.context_composed_pipeline',
            '--config', str(Path(template).resolve()), '--local', str(Path(local).resolve()),
            '--action', 'run']
    with (root/'driver.log').open('x') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, argv=argv, started_unix=__import__('time').time(),
                  process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text()
                  .split(') ', 1)[1].split()[19])
    commit_json(root/'started.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--action', choices=['validate', 'detach'], default='validate')
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    if args.action == 'detach':
        result = detach(args.config, args.local)
    else:
        validate(c)
        result = dict(status='composed_pipeline_validated_not_dispatched',
                      schema=SCHEMA, npu_executed=False, dispatch_allowed=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    main()
