"""Continue a stopped Stage2 view pipeline without rescoring or repeating accepted windows."""

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import width
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.stage2_view_runtime import indices, layout
from tools.reports import context_backend_trace_replay as trace
from tools.reports import context_stage2_endpoint_successor as endpoint
from tools.reports import context_stage2_view_pipeline as pipeline
from tools.reports import run_stage2_view_window as execution
from tools.reports.context_cached_acceptance_sequence import audit_config
from tools.reports.context_completed_stage_receipts import check_tree
from tools.reports.context_dual_probe_operations import check_runtime
from tools.reports.context_replica_training_operations import path as canonical


FINITE = ('prefix', 'recovery', 'tail')


def same_result(a, b):
    return {k: v for k, v in a.items() if k != 'provenance'} == {
        k: v for k, v in b.items() if k != 'provenance'}


def recorded_replay(config, prior, binding, *, workers):
    """Check the captured NPU replay, then expose it as a read-only audit input."""
    recorded = endpoint.read_pin(binding)
    root = Path(binding['path']).parent
    if Path(binding['path']) != root/'accepted.json' or root.parent != Path(prior['trace_replay']['output']):
        raise ValueError('exact prior backend replay location required')
    launch = endpoint.read(root/'launch.json')
    inspected = endpoint.read(root/'inspect.json')
    check_runtime(dict(argv=launch['argv'], image=prior['trace_replay']['image']), launch, inspected)
    remote_root = prior['trace_replay']['remote_root']+'/'+root.name
    remote, mapping = trace.remote_configuration(config, remote_root)
    identity = endpoint.read(root/'package/FILES.json')
    state = inspected['State']
    if (launch['config'] != config or launch['remote_config'] != remote or launch['files'] != identity
            or inspected['Id'] != launch['container_id'] or inspected['Image'] != prior['trace_replay']['image']
            or state['Status'] != 'exited' or state['Running'] or state['ExitCode'] != 0 or state['OOMKilled']
            or endpoint.read(root/'terminal.json') != dict(exit_code=0, files=identity, config=remote,
                                                          model_calls=0, npu_executed=True)):
        raise ValueError('exact terminal NPU replay evidence required')
    check_tree(root/'package', dict(identity, **{'FILES.json': file_hash(root/'package/FILES.json')}),
               workers=workers)
    if (endpoint.read(root/'package/config.json') != remote
            or any(identity.get(name) != spec['sha256'] for name, spec in mapping.items())):
        raise ValueError('archived replay inputs differ')
    source = trace.TraceReplayEngine(prior['trace_replay'])
    if identity.keys() != source.sources.keys() | mapping.keys() | {'config.json'} or any(
            identity.get(name) != digest for name, digest in source.sources.items()):
        raise ValueError('archived backend auditor differs from frozen source')
    raw = trace.accepted(endpoint.read(root/'report.json'), config, remote)
    if raw != recorded:
        raise ValueError('archived replay report differs')
    return recorded


def completed_window(kind, entry, prior, prepared, prefix, *, workers):
    controller = Path(prior['output'])
    expected = prior.get('collected_prefix') if kind == 'prefix' else None
    op_pin = entry['operation']
    if (expected is not None and op_pin != expected
            or expected is None and Path(op_pin['path']) != controller/(kind+'-window.json')
            or Path(entry['audit']['path']) != controller/kind/'audit.json'):
        raise ValueError('reuse requires exact previous window records')
    op = endpoint.read_pin(op_pin)
    audit = endpoint.read_pin(entry['audit'])
    execution.request(op)
    deployed = op['deployment']
    if (op['window'] != kind or any(deployed[k] != v for k, v in prior['windows'][kind].items())
            or any(deployed[k] != prior['deployment'][k] for k in ('host', 'lease', 'driver_root'))
            or op['objective_loss_atol'] != prior['window_operations']['objective_loss_atol']
            or op['training_package']['reference_job'] != prior['dataset']['reference_job']):
        raise ValueError('window configuration differs from the stopped pipeline')
    if expected is None:
        terminal = endpoint.read(controller/kind/'terminal.json')
        if terminal != dict(exit_code=0, status=audit['status'], audit_sha256=entry['audit']['sha256']):
            raise ValueError('accepted window must have its exact audit terminal')
    package = Path(deployed['package'])
    manifest = endpoint.read_pin(dict(path=str(package/'PREPARED.json'), sha256=deployed['manifest_sha256']))
    cfg = endpoint.read_pin(dict(path=str(package/'config.json'), sha256=manifest['files']['config.json']))
    lineage = cfg['view_lineage']
    source = prior['window_package']
    if (lineage != audit['lineage'] or lineage['parent_sha256'] != prepared['parent_sha256']
            or lineage['dataset_sha256'] != prior['dataset']['manifest_sha256']
            or lineage['objective_design_sha256'] != source['objective_design']['sha256']
            or lineage['assets'] != source.get('assets')
            or ('topology' in source and lineage['topology'] != source['topology'])
            or cfg['view_window'] != kind or audit['world_size'] != layout(cfg)['dp_replicas']):
        raise ValueError('previous numerical acceptance is for a different view or layout')
    indices(cfg)
    # The archived launch receipt already supplies the exact execution
    # identity. A new launch rechecks the entire fixed-image CPU contract.
    cpu = endpoint.read_pin(prior['window_operations']['image_cpu'])
    collection = Path(cpu['collection'])
    tested = endpoint.read_pin(dict(path=str(collection/'package/FILES.json'),
                                    sha256=cpu['source_manifest_sha256']))
    for name in ('audit_stage2_view_window.py', 'audit_context_replica_training.py',
                 'audit_objective_recovery.py', 'audit_context_cached_recovery.py'):
        path = 'tools/reports/'+name
        if tested.get(path) != file_hash(execution.ROOT/path):
            raise ValueError('archived window auditor source changed: '+path)
    original_prefix = {k:prefix[k] for k in ('audit', 'run')} if kind == 'recovery' else None
    if op['prefix'] != original_prefix:
        raise ValueError('recovery must bind the original verified prefix')
    ac = audit_config(dict(deployed, training_package=op['training_package']), None)
    ac.update(objective_loss_atol=op['objective_loss_atol'], prefix=original_prefix)

    def verified_replay(config, provenance):
        return dict(recorded_replay(config, prior, entry['backend_replay'], workers=workers),
                    provenance=provenance)

    # Recheck exact exit, isolated runtime, complete transport, schedule, and
    # training output. Only the already completed accelerator replay is reused.
    kwargs = {'trace_auditor': verified_replay} if kind != 'tail' else {}
    replayed = execution.audit(ac, audit['provenance'], **kwargs)
    old_trace = audit.get('trace_audit', {})
    new_trace = replayed.get('trace_audit', {})
    if (not same_result(old_trace, new_trace)
            or not same_result({k:v for k,v in audit.items() if k != 'trace_audit'},
                               {k:v for k,v in replayed.items() if k != 'trace_audit'})
            or replayed['status'] != 'stage2_view_'+kind+'_outputs_audited'):
        raise ValueError('recomputed window audit differs: '+kind)
    return dict(audit=entry['audit'], transport=pipeline.pin(ac['transport']), run=str(Path(ac['collection'])/'run'))


def assemble(c):
    """Derive the continuation bindings from one pinned, stopped pipeline."""
    if set(c) != {'prior_pipeline', 'prior_terminal', 'output', 'local_output', 'lease',
                  'io_workers', 'trace_destination', 'windows'}:
        raise ValueError('explicit prior evidence, lease, and new attempt locations required')
    prior = endpoint.read_pin(c['prior_pipeline'])
    root = Path(prior['output'])
    if (Path(c['prior_pipeline']['path']) != root/'config.json'
            or Path(c['prior_terminal']['path']) != root/'terminal.json'
            or endpoint.read_pin(c['prior_terminal'])['exit_code'] != 1):
        raise ValueError('exact stopped Stage2 pipeline required')
    out = Path(canonical(c['local_output'])).resolve()
    target = Path(canonical(c['output'])).resolve()
    if (out.parent != target.parent or out in (target, Path(str(target)+'-driver'))
            or out.is_relative_to(root.resolve())):
        raise ValueError('write the local config next to the independent continuation output')
    ds = prior['dataset']
    parent = Path(prior['objective_parent']['output'])
    prepared = dict(parent_sha256=file_hash(parent/'PREPARED.json'),
                    bound_config=pipeline.pin(ds['stage2_output']),
                    dataset_receipt=pipeline.pin(ds['receipts']['bind']))
    completed, replays = {}, None
    for kind in FINITE:
        path = root/kind/'audit.json'
        if not path.exists():
            continue
        if kind != FINITE[len(completed)]:
            raise ValueError('previous audits must form a contiguous finite prefix')
        audit = pipeline.pin(path)
        report = endpoint.read_pin(audit)
        entry = dict(audit=audit, operation=(
            prior.get('collected_prefix') if kind == 'prefix' else None)
            or pipeline.pin(root/(kind+'-window.json')))
        if kind != 'tail':
            if replays is None:
                trace_root = Path(prior['trace_replay']['output'])
                pins = [pipeline.pin(p) for p in sorted(trace_root.glob('[0-9a-f]'*16+'/accepted.json'))]
                replays = [(binding, endpoint.read_pin(binding)) for binding in pins]
            matches = [binding for binding, replay in replays
                       if same_result(report['trace_audit'], replay)]
            if len(matches) != 1:
                raise ValueError('one exact prior backend replay required for '+kind)
            entry['backend_replay'] = matches[0]
        completed[kind] = entry
    required = set((*FINITE, 'formal')) - set(completed)
    if not required <= c['windows'].keys() or not c['windows'].keys() <= set((*FINITE, 'formal')):
        raise ValueError('provide destinations for all outstanding windows')
    windows = {k:v for k,v in c['windows'].items() if k in required}
    destinations = [c['trace_destination']['output'],
                    *(loc[key] for loc in windows.values() for key in ('package', 'operations_root'))]
    for name in destinations:
        path = Path(canonical(name)).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError('local config overlaps a new attempt directory')
    return dict({k:v for k,v in c.items() if k not in ('local_output', 'windows')},
                prepared_inputs=prepared, completed=completed, windows=windows)


def prepare(c):
    prior = endpoint.read_pin(c['prior_pipeline'])
    old = Path(prior['output'])
    terminal = endpoint.read_pin(c['prior_terminal'])
    if (Path(c['prior_pipeline']['path']) != old/'config.json'
            or Path(c['prior_terminal']['path']) != old/'terminal.json'
            or terminal['exit_code'] != 1):
        raise ValueError('stopped Stage2 pipeline with incomplete formal training required')
    formal = prior['windows']['formal']
    formal_stem = Path(formal['operations_root'])/formal['tag']
    if any(Path(str(formal_stem)+suffix).exists() for suffix in ('-attempt.json', '-launch.json')):
        raise ValueError('formal launch was attempted; reconcile its exact handle first')
    use = c['prepared_inputs']
    ds = prior['dataset']
    if (use['bound_config']['path'] != ds['stage2_output']
            or use['dataset_receipt']['path'] != ds['receipts']['bind']):
        raise ValueError('use the original bound view and receipt')
    receipt = endpoint.read_pin(use['dataset_receipt'])
    bound = endpoint.read_pin(use['bound_config'])
    if (receipt['status'] != 'stage2_dataset_release_audited'
            or receipt['manifest_sha256'] != ds['manifest_sha256']
            or receipt['bound_config_sha256'] != use['bound_config']['sha256']
            or bound['training_dataset']['manifest_sha256'] != ds['manifest_sha256']):
        raise ValueError('independently bound TRAIN view required')
    view = copy.deepcopy(prior)
    view.pop('collected_prefix', None)
    view.update(output=c['output'], prepared_inputs=use)
    lease = c['lease']
    deadline = lease['deadline_unix']
    if (set(lease['minimum_seconds']) != set(FINITE)
            or any(type(v) is not int or v <= 0 for v in lease['minimum_seconds'].values())
            or (deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)))
            or type(lease['stop_grace_seconds']) is not int or lease['stop_grace_seconds'] < 1
            or type(lease['reserve_seconds']) is not int or lease['reserve_seconds'] < lease['stop_grace_seconds'] + 30
            or type(lease['formal_step_multiplier']) not in (int, float)
            or not math.isfinite(lease['formal_step_multiplier']) or lease['formal_step_multiplier'] < 1):
        raise ValueError('explicit new lease and finite/formal budgets required')
    endpoint.check_budget(lease, 0, stage='reattach')
    workers = width(c['io_workers'])
    for section in ('deployment', 'trace_replay'):
        view[section].pop('lease_deadline_unix', None)
        view[section].pop('lease_stop_grace_seconds', None)
        if deadline is not None:
            view[section].update(lease_deadline_unix=deadline, lease_stop_grace_seconds=lease['stop_grace_seconds'])
    if set(c['trace_destination']) != {'output', 'remote_root', 'tag'}:
        raise ValueError('only trace attempt location can change')
    view['trace_replay'].update(**c['trace_destination'])
    entries = c['completed']
    stages = FINITE[:len(entries)]
    if set(entries) != set(stages) or set(c['windows']) != (set(FINITE) | {'formal'}) - set(stages):
        raise ValueError('reuse a contiguous prefix and declare every remaining window')
    required_location = {'tag', 'package', 'operations_root', 'remote_root'}
    if any(set(loc) != required_location for loc in c['windows'].values()):
        raise ValueError('each new window requires exact launch locations')
    tags = [c['trace_destination']['tag'], *(v['tag'] for v in c['windows'].values())]
    previous_tags = {prior['trace_replay']['tag'], *(v['tag'] for v in prior['windows'].values())}
    if (len(set(tags)) != len(tags) or set(tags) & previous_tags
            or any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*', tag) for tag in tags)):
        raise ValueError('fresh distinct launch tags required')
    for kind in (w for w in FINITE if w not in stages):
        previous = prior['windows'][kind]
        stem = Path(previous['operations_root'])/previous['tag']
        handle_path = Path(str(stem)+'-launch.json')
        if not handle_path.exists() and Path(str(stem)+'-attempt.json').exists():
            raise ValueError('resolve the missing '+kind+' container handle before retrying')
        if handle_path.exists():
            handle = endpoint.read(handle_path)
            waited = endpoint.read(str(stem)+'-wait.json')
            inspected = endpoint.read(str(stem)+'-inspect.json')
            if (waited != dict(container_id=handle['container_id'], exit_code=inspected['State']['ExitCode'])
                    or inspected['Id'] != handle['container_id'] or inspected['Image'] != handle['image']
                    or inspected['State']['Status'] != 'exited' or inspected['State']['Running']
                    or (Path(prior['output'])/kind/'audit.json').exists()):
                raise ValueError('resolve the exact previous '+kind+' result before retrying')
            if waited['exit_code'] == 0 and (kind != 'recovery' or stages != ('prefix',)
                                         or terminal.get('stage') != 'recovery'):
                raise ValueError('prior '+kind+' completed; independently audit its original result')
            if waited['exit_code'] == 0:
                view['collected_recovery'] = pipeline.pin(old/'recovery-window.json')
    locations = {k: canonical(v) for k, v in {'controller':c['output'], 'trace':c['trace_destination']['output'],
        **{f'{w}-{k}':l[k] for w,l in c['windows'].items() for k in ('package', 'operations_root')}}.items()}
    remote = [Path(canonical(c['trace_destination']['remote_root'])),
              *(Path(canonical(v['remote_root'])) for v in c['windows'].values())]
    old_remote = [Path(prior['trace_replay']['remote_root']), *(Path(v['remote_root']) for v in prior['windows'].values())]
    for i, loc in enumerate(remote):
        if not loc.is_absolute() or any(loc.is_relative_to(p) or p.is_relative_to(loc) for p in old_remote+remote[:i]):
            raise ValueError('fresh disjoint remote window and replay directories required')
    local = [Path(v).resolve() for v in locations.values()]
    for i, loc in enumerate(local):
        if loc.exists() or any(loc.is_relative_to(p) or p.is_relative_to(loc) for p in local[:i]):
            raise ValueError('fresh disjoint local attempt locations required')
    check_paths(locations)
    view['windows'].update(c['windows'])
    checked = {}
    for kind in stages:
        checked[kind] = completed_window(kind, entries[kind], prior, use, checked.get('prefix'), workers=workers)
    endpoint.check_budget(lease, 0, stage='reattach_after_audit')
    return view, receipt['consumer'], checked


def run(c, provenance):
    view, consumer, completed = prepare(c)
    output = Path(view['output'])
    guard = endpoint.window_guard(c['lease'], consumer, output)
    if recovery := view.pop('collected_recovery', None):
        op = endpoint.read_pin(recovery)
        prior = endpoint.read_pin(c['prior_pipeline'])
        deployed = op['deployment']
        prefix = completed['prefix']
        if (op['window'] != 'recovery'
                or any(deployed[k] != v for k, v in prior['windows']['recovery'].items())
                or any(deployed[k] != prior['deployment'][k] for k in ('host', 'lease', 'driver_root'))
                or op['training_package']['reference_job'] != prior['dataset']['reference_job']
                or op['objective_loss_atol'] != prior['window_operations']['objective_loss_atol']
                or op['prefix'] != {k: prefix[k] for k in ('run', 'audit')}):
            raise ValueError('recovery collection does not belong to the audited prefix')
        ac = audit_config(dict(deployed, training_package=op['training_package']), None)
        ac.update(objective_loss_atol=op['objective_loss_atol'], prefix=op['prefix'])
        replay = trace.TraceReplayEngine(dict(view['trace_replay'], wait_for_lease=True))
        proof = execution.audit(ac, provenance, trace_auditor=replay.audit)
        if proof['status'] != 'stage2_view_recovery_outputs_audited':
            raise ValueError('independent recovery audit required')
        archived = Path(str(output)+'-collected-recovery')
        check_paths({'collected recovery': archived})
        archived.mkdir()
        commit_json(archived/'audit.json', proof)
        completed['recovery'] = dict(audit=pipeline.pin(archived/'audit.json'),
                                    transport=pipeline.pin(ac['transport']), run=str(Path(ac['collection'])/'run'))

        def guard(stage, proofs):
            with Path(view['deployment']['lease']).open('r') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX)
            endpoint.window_guard(c['lease'], consumer, output)(stage, proofs)
    pipeline.run(view, provenance, completed_windows=completed,
                 before_window=guard)


def forecast(lease, consumer, completed, *, now=None):
    """Report which budgets are known without claiming that a lease is authorized."""
    now = time.time() if now is None else now
    remaining = [stage for stage in FINITE if stage not in completed]
    budgets = {stage: lease['minimum_seconds'][stage] for stage in remaining}
    if not remaining:
        proofs = {stage: entry['audit'] for stage, entry in completed.items()}
        budgets['formal'] = math.ceil(endpoint.window_seconds(lease, consumer, 'formal', proofs))
    available = None if lease['deadline_unix'] is None else lease['deadline_unix'] - now
    known = sum(budgets.values()) + lease['reserve_seconds']
    return dict(updates=consumer['stop_update'], reused=list(completed), remaining=remaining+['formal'],
                minimum_stage_seconds=budgets, deadline_unix=lease['deadline_unix'],
                remaining_lease_seconds=math.floor(available) if available is not None else None,
                reserve_seconds=lease['reserve_seconds'],
                required_seconds_including_reserve=known if not remaining else None,
                fits_complete_run=(available > known) if not remaining and available is not None else None)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--action', choices=('assemble', 'validate', 'run', 'detach'), required=True)
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    if a.action == 'run':
        run(c, provenance)
    elif a.action == 'assemble':
        out = Path(canonical(c['local_output']))
        if out.exists():
            raise FileExistsError('use a fresh local config path')
        binding = assemble(c)
        _, consumer, completed = prepare(binding)
        commit_json(out, dict(schema='jev-local-macros-v1', macros=dict(STAGE2_VIEW_SUCCESSOR=binding)))
        print(json.dumps(dict(status='stage2_continuation_config_prepared', local=str(out),
                              sha256=file_hash(out), npu_executed=False,
                              **forecast(binding['lease'], consumer, completed)), indent=2))
    elif a.action == 'detach':
        # Validate locally first, then verify the same evidence again before dispatch.
        _, consumer, completed = prepare(c)
        if tuple(completed) == FINITE:
            proofs = {stage: entry['audit'] for stage, entry in completed.items()}
            endpoint.check_budget(c['lease'], endpoint.window_seconds(c['lease'], consumer, 'formal', proofs),
                                  stage='formal')
        print(endpoint.detach(a.config, a.local, c, module=__spec__.name))
    else:
        _, consumer, completed = prepare(c)
        print(json.dumps(dict(status='stage2_continuation_validated', npu_executed=False,
                              **forecast(c['lease'], consumer, completed)), indent=2))
