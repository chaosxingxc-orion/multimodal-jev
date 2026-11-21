"""Continue a collected, failed Stage2 reference attempt on the same host."""

import argparse
import json
import math
from pathlib import Path
import re
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.context_execution import width
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_reference_job_operations as operations
from tools.reports import context_reference_resume_release as resume
from tools.reports import context_stage2_endpoint_successor as endpoint
from tools.reports import prepare_context_reference_job as reference
from tools.reports.prepare_context_replica_training_candidate import rebase


def recovery_lease(c, original):
    lease = original['lease']
    if 'renewed_lease' not in c:
        return lease
    renewed = c['renewed_lease']
    if set(renewed) != {'deadline_unix'}:
        raise ValueError('a renewed lease can only set the new deadline')
    deadline = renewed['deadline_unix']
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline <= lease['deadline_unix']):
        raise ValueError('renewal requires a later authorized deadline')
    if time.time() < lease['deadline_unix']:
        raise ValueError('reconcile original attempts after their lease expires')
    effective = dict(lease, deadline_unix=deadline)
    endpoint.check_budget(effective, 0, stage='reference_renewal')
    return effective


def validate(c):
    original = endpoint.read_pin(c['predecessor']['config'])
    old = Path(original['output'])
    driver = Path(c['predecessor']['driver']['path'])
    output = Path(c['output'])
    if (Path(c['predecessor']['config']['path']) != old/'config.json'
            or driver != Path(str(old)+'-driver')/'started.json'
            or output.name != old.name or output.parent == old.parent
            or output.parent.is_relative_to(old.parent)
            or old.parent.is_relative_to(output.parent)):
        raise ValueError('pinned prior successor and fresh disjoint candidate root required')
    settings = c['recovery']
    if (set(settings) != {'io_workers', 'tag_suffix', 'minimum_seconds', 'legacy_engine_timeout_seconds',
                         'shutdown_timeout_seconds'}
            or not re.fullmatch(r'-[a-z0-9]+', settings['tag_suffix'])
            or type(settings['minimum_seconds']) is not int or settings['minimum_seconds'] <= 0):
        raise ValueError('explicit bounded recovery policy required')
    width(settings['io_workers'])
    reference.reference_runner([], settings['shutdown_timeout_seconds'], resume=True,
                               legacy_engine_timeout=settings['legacy_engine_timeout_seconds'])
    recovery_lease(c, original)
    if 'renewed_lease' in c and 'stopped_fallback' not in c:
        raise ValueError('reconcile the original fallback before opening a renewed lease')
    check_paths({'recovery': output})
    return original, endpoint.read_pin(c['predecessor']['driver'])


def stopped_reference(original):
    root = Path(original['output'])
    terminal = endpoint.read(root/'terminal.json')
    if terminal['exit_code'] == 0 or terminal.get('stage') != 'train_parity_and_reference':
        return None
    chain = original['reference']
    final = Path(chain['reference_operations']['operations_root'])/'finalization'
    if not (final/'terminal.json').is_file() or endpoint.read(final/'terminal.json')['exit_code'] != 1:
        return None
    prior = endpoint.read(final/'config.json')
    declared = dict(chain['deployment'], **chain['reference_operations'])
    declared.update(lease_deadline_unix=original['lease']['deadline_unix'],
                    lease_stop_grace_seconds=original['lease']['stop_grace_seconds'])
    for key in ('host', 'lease', 'lease_deadline_unix', 'lease_stop_grace_seconds', 'tag', 'operations_root'):
        expected = declared[key]
        if prior[key] != expected:
            raise ValueError('prior reference differs from exact deployment: '+key)
    if prior['reference_job']['parent'] != endpoint.read(root/'bound-runtime.json'):
        raise ValueError('prior reference is not bound to this endpoint')
    stem = Path(prior['operations_root'])/prior['tag']
    wait = Path(str(stem)+'-wait.json')
    transport = Path(str(stem)+'-collected.transport.json')
    if (not wait.is_file() or endpoint.read(wait)['exit_code'] == 0 or not transport.is_file()
            or endpoint.read(transport)['status'] != 'transport_complete_independent_audit_required'):
        return None
    return prior


def reconcile_fallback(c, prior):
    """Return the newest collected failed scoring attempt, or the undispatched fallback's source."""
    f = c['stopped_fallback']
    if set(f) != {'config', 'driver', 'terminal'}:
        raise ValueError('pinned fallback config, driver and terminal required')
    cfg = endpoint.read_pin(f['config'])
    output = Path(cfg['output'])
    if (cfg['predecessor'] != c['predecessor']
            or Path(f['config']['path']) != output/'config.json'
            or Path(f['driver']['path']) != Path(str(output)+'-driver')/'started.json'
            or Path(f['terminal']['path']) != output/'terminal.json'
            or cfg['recovery']['tag_suffix'] == c['recovery']['tag_suffix']
            or Path(c['output']).parent.is_relative_to(output.parent)
            or output.parent.is_relative_to(Path(c['output']).parent)):
        raise ValueError('exact fallback for the same predecessor required')
    endpoint.wait_exact(endpoint.read_pin(f['driver']), output/'terminal.json')
    terminal = endpoint.read_pin(f['terminal'])
    stage = terminal.get('stage')
    if terminal['exit_code'] != 1:
        raise ValueError('fallback did not fail during scoring; use its published evidence')
    op = output/'reference-operations.json'
    if prior is None:
        raise ValueError('original collected scoring attempt required')
    original = endpoint.read_pin(c['predecessor']['config'])
    old_root = Path(original['output']).parent
    target = rebase(original, str(old_root)+'/', str(output.parent)+'/')
    operations_root = Path(target['reference']['reference_operations']['operations_root'])
    settings = cfg['recovery']
    stem = operations_root/(prior['tag']+settings['tag_suffix'])
    if stage in ('seal_stopped_reference', 'prepare_reference_resume'):
        if any(Path(str(stem)+suffix).exists() for suffix in ('-attempt.json','-launch.json')):
            raise ValueError('fallback scoring already attempted; reconcile that newer handle')
        return prior
    if stage != 'reference_resume' or not op.is_file():
        raise ValueError('fallback stopped outside a resumable scoring stage')
    latest = endpoint.read(op)
    spec = latest['reference_job']
    parent_spec = prior['reference_job']
    prefix = spec['resume']
    prefix_path = output/'prefix'
    receipt = endpoint.read_pin(dict(path=str(prefix_path/'receipt.json'), sha256=prefix['receipt_sha256']))
    prior_stem = Path(prior['operations_root'])/prior['tag']
    prior_transport = Path(str(prior_stem)+'-collected.transport.json')
    expected_spec = dict(parent_spec, output=target['reference']['reference']['output'],
                         remote_root=parent_spec['remote_root']+settings['tag_suffix'],
                         manifest_sha256=spec['manifest_sha256'], resume=prefix,
                         legacy_engine_timeout_seconds=settings['legacy_engine_timeout_seconds'],
                         shutdown_timeout_seconds=settings['shutdown_timeout_seconds'])
    effective = recovery_lease(cfg, original)
    expected = dict(prior, tag=stem.name, operations_root=str(operations_root),
                    package=spec['output'], remote_root=spec['remote_root'],
                    lease_deadline_unix=effective['deadline_unix'],
                    manifest_sha256=spec['manifest_sha256'], reference_job=expected_spec)
    stop = operations_root/'finalization'
    wait = Path(str(stem)+'-wait.json')
    transport = Path(str(stem)+'-collected.transport.json')
    if (latest != expected or prefix['prefix_root'] != str(prefix_path)
            or receipt['original_transport_sha256'] != file_hash(prior_transport)
            or endpoint.read(stop/'config.json') != latest
            or endpoint.read(stop/'terminal.json')['exit_code'] != 1
            or not wait.is_file() or endpoint.read(wait)['exit_code'] <= 0
            or not transport.is_file()
            or endpoint.read(transport)['status'] != 'transport_complete_independent_audit_required'):
        raise ValueError('latest fallback lacks an exact failed and collected reference attempt')
    return latest


def resume_plan(prior, prefix_root, receipt, lease, settings, minimum_full_seconds):
    package = Path(prior['package'])
    build = endpoint.read(package/'reference-build-config.json')
    prefix = endpoint.read(prefix_root/'reference-prefix.json')
    stem = Path(prior['operations_root'])/prior['tag']
    result_manifest = Path(str(stem)+'-collected')/'reference-build/dataset/MANIFEST.json'
    # A published dataset size also covers releases with identical inputs.
    distinct = endpoint.read(result_manifest)['distinct_inputs'] if result_manifest.is_file() else build['expected_population']
    macro = build['macro_client']['max_queries']
    total = math.ceil(distinct / macro)
    committed = receipt['commits']
    if len(prefix['commits']) != committed or not 0 < committed <= total:
        raise ValueError('committed prefix exceeds the expected reference population')
    replay_last = committed == total
    remaining = 1 if replay_last else total - committed
    seconds = settings['minimum_seconds'] + math.ceil(minimum_full_seconds * remaining / total)
    return dict(committed_macros=committed, expected_macros=total, replay_last=replay_last,
                expected_inference_macros=remaining, minimum_seconds=seconds,
                deadline_unix=lease['deadline_unix'])


def seal_resume_source(prior, newest, root, settings, provenance):
    """Keep proven blocks even if the newer container stopped while restoring them."""
    def prepare(operations, dest, **kwargs):
        return resume.prepare_prefix(dict(prior_operations=operations, output=str(dest),
                                          io_workers=settings['io_workers']), provenance, **kwargs)

    prefix_root = root/'prefix'
    if newest != prior:
        latest = root/'latest-prefix'
        ready = prepare(newest, latest, allow_empty=True)
        if ready['commits'] == 0:
            return prior, prefix_root, prepare(prior, prefix_root)
        binding = newest['reference_job']['resume']
        source = Path(binding['prefix_root'])
        original = endpoint.read_pin(dict(path=str(source/'receipt.json'), sha256=binding['receipt_sha256']))
        old = endpoint.read_pin(dict(path=str(source/'reference-prefix.json'), sha256=original['prefix_sha256']))
        new_receipt = endpoint.read_pin(dict(path=str(latest/'receipt.json'), sha256=ready['receipt_sha256']))
        new = endpoint.read_pin(dict(path=str(latest/'reference-prefix.json'), sha256=new_receipt['prefix_sha256']))
        if old['lineage'] != new['lineage'] or len(old['commits']) != original['commits']:
            raise ValueError('fallback prefix belongs to different reference inputs')
        immutable = len(old['commits'])-int(binding['replay_last'])
        shared = min(immutable, len(new['commits']))
        if old['commits'][:shared] != new['commits'][:shared]:
            raise ValueError('fallback changed an already committed reference block')
        if len(new['commits']) >= len(old['commits']):
            if binding['replay_last']:
                from zipfile import ZipFile
                rel = f'batch-{immutable:08d}/COMPLETE.json'
                old_archive = source/'reference-prefix.zip'
                new_archive = latest/'reference-prefix.zip'
                if (file_hash(old_archive) != original['archive_sha256']
                        or file_hash(new_archive) != new_receipt['archive_sha256']):
                    raise ValueError('replayed tail archive differs from receipt')
                with ZipFile(old_archive) as a, ZipFile(new_archive) as b:
                    if json.loads(a.read(rel))['result_sha256'] != json.loads(b.read(rel))['result_sha256']:
                        raise ValueError('fallback replay changed the last reference outputs')
            return newest, latest, ready
    return prior, prefix_root, prepare(prior, prefix_root)


def run(c, provenance):
    original, driver = validate(c)
    old = Path(original['output'])
    root = Path(c['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', c)
    commit_json(root/'provenance.json', provenance)
    stage = 'waiting_for_original'

    def progress(value):
        nonlocal stage
        stage = value
        atomic_json(root/'progress.json', dict(stage=stage, unix=time.time()))

    try:
        progress(stage)
        endpoint.wait_exact(driver, old/'terminal.json')
        prior = stopped_reference(original)
        newest = prior
        if 'renewed_lease' in c:
            newest = reconcile_fallback(c, prior)
        if prior is None:
            commit_json(root/'terminal.json', dict(exit_code=0, status='not_applicable_no_failed_scoring_attempt'))
            return
        lease, settings = recovery_lease(c, original), c['recovery']
        progress('seal_stopped_reference')
        prior, prefix_root, receipt = seal_resume_source(prior, newest, root, settings, provenance)
        decision = resume_plan(prior, prefix_root, receipt, lease, settings,
                               lease['minimum_seconds']['parity_and_reference'])
        commit_json(root/'recovery-budget.json', decision)
        endpoint.check_budget(lease, decision['minimum_seconds'], stage=stage)

        progress('prepare_reference_resume')
        target = rebase(original, str(old.parent)+'/', str(root.parent)+'/')
        target['lease'] = lease
        bound = prior['reference_job']['parent']
        original_job = prior['reference_job']
        remote = original_job['remote_root'] + settings['tag_suffix']
        spec = dict(original_job, **{k:settings[k] for k in ('legacy_engine_timeout_seconds', 'shutdown_timeout_seconds')})
        spec.update(output=target['reference']['reference']['output'], remote_root=remote,
                    resume=dict(prefix_root=str(prefix_root), receipt_sha256=receipt['receipt_sha256'],
                                replay_last=decision['replay_last']))
        spec.pop('manifest_sha256', None)
        spec['manifest_sha256'] = reference.prepare(spec, provenance)['manifest_sha256']
        op = dict(prior, tag=prior['tag']+settings['tag_suffix'],
                  operations_root=target['reference']['reference_operations']['operations_root'],
                  package=spec['output'], remote_root=remote,
                  lease_deadline_unix=lease['deadline_unix'],
                  manifest_sha256=spec['manifest_sha256'], reference_job=spec)
        commit_json(root/'reference-operations.json', op)
        endpoint.check_budget(lease, decision['minimum_seconds'], stage=stage)
        progress('reference_resume')
        operations.run_and_audit(op, provenance)
        job = endpoint.reference_job(op)
        commit_json(root/'reference-job.json', job)

        # Keep the original fixed-image CPU evidence; give all execution attempts
        # fresh local and remote identities.
        view = target['view_pipeline']
        for key in ('deployment', 'trace_replay'):
            view[key]['lease'] = original['view_pipeline'][key]['lease']
        view['window_operations']['image_cpu'] = original['view_pipeline']['window_operations']['image_cpu']
        view['trace_replay']['source'] = original['view_pipeline']['trace_replay']['source']
        for window in view['windows'].values():
            window['tag'] += settings['tag_suffix']
            window['remote_root'] += settings['tag_suffix']
        view['trace_replay']['tag'] += settings['tag_suffix']
        view['trace_replay']['remote_root'] += settings['tag_suffix']
        target['cached_parent']['remote_root'] += settings['tag_suffix']
        view['objective_parent']['remote_root'] += settings['tag_suffix']
        progress('publish_train_view')
        view, publication = endpoint.publish_view(target, bound, job, provenance, root)

        progress('stage2_windows')
        commit_json(root/'terminal.json', endpoint.run_windows(target, view, publication, provenance, root))
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, stage=stage, error=repr(exc), restart=False))
        raise


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--action', choices=('validate', 'run', 'detach'), required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    validate(config)
    if args.action == 'run':
        run(config, provenance)
    elif args.action == 'detach':
        print(json.dumps(endpoint.detach(args.config, args.local, config, module=__spec__.name)))
    else:
        print(json.dumps(dict(status='reference_recovery_configuration_validated', npu_executed=False)))
