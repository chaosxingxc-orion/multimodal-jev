"""Advance an independently audited Stage1 endpoint to Stage2 on its lease."""

import argparse
import copy
from functools import partial
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.audit_content_session import ContentProofSession
from multimodal_jev.context_execution import width
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.stage2_view_runtime import pinned
from tools.reports import context_infonce_stage2_source as source
from tools.reports import prepare_context_stage2_runtime as runtime
from tools.reports import prepare_context_cached_training as cached
from tools.reports import prepare_stage2_dataset as dataset
from tools.reports import context_stage2_view_pipeline as pipeline
from tools.reports import run_stage1_train_reference_chain as reference
from tools.reports.context_cached_storage_operations import reference_command
from tools.reports.finalize_context_infonce import wait_exact


def read(path):
    return json.loads(Path(path).read_text())


def read_pin(binding):
    return pinned(binding['path'], binding['sha256'])


def check_budget(lease, seconds, *, stage, now=None):
    now = time.time() if now is None else now
    if lease['deadline_unix'] is None:return
    if lease['deadline_unix'] - now <= seconds + lease['reserve_seconds']:
        raise ValueError(f'lease cannot fit {stage}: needs {math.ceil(seconds)}s plus reserve')


def validate(c, *, prior_deadline=None):
    prior = c['predecessor']
    final = read_pin(prior['config'])
    if (Path(prior['driver']['path']).parent != Path(final['finalization_root'] + '-driver')
            or Path(prior['driver']['path']).name != 'started.json'
            or Path(prior['config']['path']) != Path(final['finalization_root'])/'config.json'):
        raise ValueError('exact Stage1 finalizer identity required')
    ops, _ = load_config(final['operations_config'], final['operations_local'])
    deployment = c['reference']['deployment']
    lease = c['lease']
    if (any(ops[k] != deployment[k] for k in ('host', 'lease'))
            or deployment != c['view_pipeline']['deployment']
            or (lease['deadline_unix'] is not None and (type(lease['deadline_unix']) not in (int,float)
                or not math.isfinite(lease['deadline_unix'])))
            or (lease['deadline_unix'] if prior_deadline is None else prior_deadline) != ops.get('lease_deadline_unix')
            or type(lease['stop_grace_seconds']) is not int
            or lease['stop_grace_seconds'] < 1
            or type(lease['reserve_seconds']) is not int
            or lease['reserve_seconds'] < lease['stop_grace_seconds'] + 30):
        raise ValueError('same host, lock and hard deadline as Stage1 required')
    if (set(lease['minimum_seconds']) != {'parity_and_reference', 'prefix', 'recovery', 'tail'}
            or any(type(v) is not int or v <= 0 for v in lease['minimum_seconds'].values())
            or type(lease['formal_step_multiplier']) not in (float, int)
            or not math.isfinite(lease['formal_step_multiplier']) or lease['formal_step_multiplier'] < 1):
        raise ValueError('explicit finite window budgets and formal step margin required')
    if (c['reference']['parity_operations']['finite_inference_dispatch_allowed'] is not True
            or c['reference']['reference_operations']['reference_dispatch_allowed'] is not True
            or c['reference']['reference_operations']['single_admission_launch'] is not True
            or c['view_pipeline']['window_operations']['dispatch'] is not True):
        raise ValueError('explicit permission for all declared stages required')
    replay = c['view_pipeline']['trace_replay']
    if any(replay[k] != deployment[k] for k in ('host', 'lease', 'driver_root')):
        raise ValueError('replay must use the same host and lock as training')
    check_paths({'controller': c['output']})
    return final, read_pin(prior['driver'])


def renew_before_scoring(c, *, now=None):
    """Reattach an audited endpoint whose successor stopped before any dispatch."""
    from tools.reports.context_replica_training_operations import path as canonical
    from tools.reports.prepare_context_replica_training_candidate import rebase
    if set(c) != {'original', 'deadline_unix', 'tag_suffix', 'output'}:
        raise ValueError('explicit original, renewed deadline and fresh attempt required')
    original = read_pin(c['original']['config'])
    old = Path(original['output'])
    driver_pin, terminal_pin = (c['original'][name] for name in ('driver', 'terminal'))
    if (Path(c['original']['config']['path']) != old/'config.json'
            or Path(driver_pin['path']) != Path(str(old)+'-driver')/'started.json'
            or Path(terminal_pin['path']) != old/'terminal.json'):
        raise ValueError('exact original successor receipts required')
    terminal = read_pin(terminal_pin)
    if (terminal['exit_code'] != 1 or terminal.get('stage') != 'train_parity_and_reference'
            or not terminal.get('error', '').startswith("ValueError('lease cannot fit train_parity_and_reference:")):
        raise ValueError('original successor did not stop at scoring admission')
    old_deadline = original['lease']['deadline_unix']
    now = time.time() if now is None else now
    deadline = c['deadline_unix']
    if (now < old_deadline or type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline <= old_deadline):
        raise ValueError('a later authorized deadline after the original lease is required')
    wait_exact(read_pin(driver_pin), old/'terminal.json')
    chain = original['reference']
    if Path(chain['output']).exists() or any(Path(chain[phase+'_operations']['operations_root']).exists()
                                             for phase in ('parity', 'reference')):
        raise ValueError('original scoring lifecycle entered; reconcile its exact handles')

    new = Path(canonical(c['output']))
    old_parent, new_parent = old.parent.resolve(), new.parent.resolve()
    if (new.name != old.name or new.exists() or new_parent.is_relative_to(old_parent)
            or old_parent.is_relative_to(new_parent) or not re.fullmatch(r'-[a-z0-9]+', c['tag_suffix'])):
        raise ValueError('fresh independent run root and tag suffix required')
    target = rebase(original, str(old.parent)+'/', str(new.parent)+'/')
    for name in ('source', 'runtime'):
        target[name] = original[name]
    target['lease']['deadline_unix'] = deadline
    suffix = c['tag_suffix']
    for section in (target['reference'], target['view_pipeline']):
        section['deployment'] = original['reference']['deployment']
    c_ref, c_view = target['reference'], target['view_pipeline']
    c_view['window_operations']['image_cpu'] = original['view_pipeline']['window_operations']['image_cpu']
    for entry in (target['cached_parent'], c_view['objective_parent'], c_view['trace_replay'],
                  c_ref['parity'], c_ref['reference'], *c_view['windows'].values()):
        entry['remote_root'] += suffix
        if 'tag' in entry:
            entry['tag'] += suffix
    c_view['trace_replay']['source'] = original['view_pipeline']['trace_replay']['source']
    c_view['trace_replay']['lease'] = original['view_pipeline']['trace_replay']['lease']
    for phase in ('parity', 'reference'):
        c_ref[phase+'_operations']['tag'] += suffix
    fresh = [c_ref['output'], c_ref['chain_lock'], c_view['output'], target['cached_parent']['output'],
             c_view['objective_parent']['output'], c_view['trace_replay']['output'],
             c_view['dataset']['output'], c_view['dataset']['stage2_output'],
             *c_view['dataset']['receipts'].values(), c_ref['parity']['output'], c_ref['reference']['output'],
             *(c_ref[phase+'_operations']['operations_root'] for phase in ('parity', 'reference')),
             *(loc[key] for loc in c_view['windows'].values() for key in ('package', 'operations_root'))]
    if any(not Path(p).resolve().is_relative_to(new_parent) or Path(p).exists() for p in fresh):
        raise ValueError('all renewed write destinations must be fresh inside the independent run root')

    final, _ = validate(target, prior_deadline=old_deadline)
    final_root = Path(final['finalization_root'])
    end = read(final_root/'terminal.json')
    if end['exit_code'] != 0 or end['status'] != 'formal_infonce_audited_curve_bound':
        raise ValueError('independently audited Stage1 endpoint required')
    bound = read(old/'bound-runtime.json')
    expected = dict(original['runtime'])
    plan = final_root/'source-evaluation-plan.json'
    expected_source = dict(original['source'], source_plan=str(plan), source_plan_sha256=file_hash(plan),
        source_finalization={k:final[k] for k in ('operations_config', 'operations_local', 'initial_policy')})
    if ({k:v for k,v in bound.items() if k not in ('manifest_sha256', 'source')} != expected
            or {k:v for k,v in bound['source'].items() if k != 'manifest_sha256'} != expected_source):
        raise ValueError('original runtime must bind the exact Stage1 endpoint')
    check_budget(target['lease'], target['lease']['minimum_seconds']['parity_and_reference'],
                 stage='renewed_train_parity_and_reference', now=now)
    return target, bound


def window_seconds(lease, consumer, stage, proofs):
    if stage != 'formal':
        return lease['minimum_seconds'][stage]
    timings = [u['slowest_rank_step_seconds']
               for name in ('prefix', 'tail')
               for u in read_pin(proofs[name])['updates']]
    if not timings or any(not math.isfinite(v) or v <= 0 for v in timings):
        raise ValueError('finite positive step timings required for the formal budget')
    return max(timings) * consumer['stop_update'] * lease['formal_step_multiplier']


def window_guard(lease, consumer, root):
    def guard(stage, proofs):
        seconds = window_seconds(lease, consumer, stage, proofs)
        commit_json(root/f'{stage}-budget.json', dict(stage=stage, decision_unix=time.time(),
            minimum_seconds=math.ceil(seconds), deadline_unix=lease['deadline_unix'],
            reserve_seconds=lease['reserve_seconds']))
        check_budget(lease, seconds, stage=stage)
    return guard


def reference_job(operations):
    root = Path(operations['operations_root'])/'finalization'
    terminal = read(root/'terminal.json')
    if terminal['exit_code'] != 0 or terminal['status'] != 'reference_collected_and_audited':
        raise ValueError('independently audited full reference required')
    a = read(root/'audit-config.json')
    job = dict(a['reference_job'], **{k:a[k] for k in
               ('deployment','launch','inspect','transport','collection','results_audit_output')})
    prepared = Path(job['output'])
    job['expected_argv'] = reference_command(a['deployment'], read(prepared/'PREPARED.json'),
                                             read(prepared/'config.json'))
    return job


def publish_view(c, bound, job, provenance, root, *, parent_auditor=None):
    view = copy.deepcopy(c['view_pipeline'])
    parent = dict(c['cached_parent'], parent=bound, reference_job=job)
    parent.update(manifest_sha256=cached.prepare(parent, provenance, parent_auditor=parent_auditor)['manifest_sha256'])
    view['objective_parent']['parent'] = parent
    point = Path(job['output'])/'initial/endpoint-commit.json'
    endpoint = read(point)
    ds = view['dataset']
    ds.update(reference_job=job, stage1_expected=dict(policy_id=endpoint['joint_policy']['policy_id'],
        stage1_commit_sha256=file_hash(point), endpoint=endpoint['updates']))
    if 'expected_stage1_layout' in c['source']:
        ds['stage1_expected']['stage1_layout']=c['source']['expected_stage1_layout']
    publication = dataset.run(ds, provenance, 'prepare')
    ds['manifest_sha256'] = publication['manifest_sha256']
    lease = c['lease']
    cutoff = (dict(lease_deadline_unix=lease['deadline_unix'],
                  lease_stop_grace_seconds=lease['stop_grace_seconds']) if lease['deadline_unix'] is not None else {})
    view['deployment'].update(cutoff)
    view['trace_replay'].update(cutoff)
    commit_json(root/'view-pipeline.json', view)
    return view, publication


def run_windows(c, view, publication, provenance, root):
    pipeline.run(view, provenance, before_window=window_guard(c['lease'], publication['consumer'], root))
    terminal = Path(view['output'])/'terminal.json'
    proof = read(terminal)
    if proof['exit_code'] != 0 or proof['status'] != 'formal_stage2_view_audited':
        raise ValueError('complete Stage2 audit required')
    return dict(exit_code=0, status=proof['status'], updates=proof['updates'],
                pipeline_terminal_sha256=file_hash(terminal))


def continue_reference(c, bound, provenance, root, progress, *, parent_auditor=None, proof_session=None):
    lease = c['lease']
    cutoff = (dict(lease_deadline_unix=lease['deadline_unix'],
                  lease_stop_grace_seconds=lease['stop_grace_seconds']) if lease['deadline_unix'] is not None else {})
    progress('train_parity_and_reference')
    check_budget(lease, lease['minimum_seconds']['parity_and_reference'],
                 stage='train_parity_and_reference')
    chain = c['reference']
    chain.update(parent=bound, parent_driver=None)
    for phase in ('parity', 'reference'):
        chain[phase+'_operations'].update(cutoff)
    reference.run(chain, provenance, parent_auditor=parent_auditor, proof_session=proof_session)
    job = reference_job(chain['reference_operations'])
    commit_json(root/'reference-job.json', job)

    progress('publish_train_view')
    view, publication = publish_view(c, bound, job, provenance, root, parent_auditor=parent_auditor)

    progress('stage2_windows')
    commit_json(root/'terminal.json', run_windows(c, view, publication, provenance, root))


def run(config, provenance):
    c = copy.deepcopy(config)
    final, prior = validate(c)
    root = Path(c['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', c)
    commit_json(root/'provenance.json', provenance)
    stage = 'stage1_finalization'

    def progress(value):
        nonlocal stage
        stage = value
        atomic_json(root/'progress.json', dict(stage=stage, unix=time.time()))

    try:
        project = Path(__file__).resolve().parents[2]
        code = [p for folder in ('src/multimodal_jev', 'tools/reports', 'tools/jev')
                for p in (project/folder).glob('*.py')]
        execution = c['reference']['reference_operations']['admission_execution']
        session = ContentProofSession(code, io_workers=width(execution['io_workers']))

        def common(spec, p):
            # The isolated subprocess only imports the already byte-verified
            # payload; the audit reads every file before invoking it.
            proof = session.get(('common_runtime', json.dumps(spec, sort_keys=True)),
                                lambda: runtime.audit_runtime(spec, p))
            return dict(proof, provenance=p)

        def reconstruct(spec):
            key = ('stage1_source', json.dumps({k: spec[k] for k in
                   ('source_plan', 'source_plan_sha256', 'source_finalization')}, sort_keys=True))
            return session.get(key, lambda: source.reconstruct_plan(spec))

        progress('common_runtime_audit')
        common(c['runtime']['runtime'], provenance)
        progress('stage1_finalization')
        final_root = Path(final['finalization_root'])
        wait_exact(prior, final_root/'terminal.json')
        terminal = read(final_root/'terminal.json')
        if (terminal['exit_code'] != 0 or terminal['status'] != 'formal_infonce_audited_curve_bound'):
            raise ValueError('Stage1 independent audit did not finish successfully')
        progress('bind_endpoint')
        plan = final_root/'source-evaluation-plan.json'
        starting = dict(c['source'], source_plan=str(plan), source_plan_sha256=file_hash(plan),
            source_finalization={k:final[k] for k in ('operations_config', 'operations_local', 'initial_policy')})
        starting.update(manifest_sha256=source.prepare(starting, provenance, reconstruct=reconstruct)['manifest_sha256'])
        bound = dict(c['runtime'], source=starting)
        source_auditor = partial(source.audit, reconstruct=reconstruct)
        bound.update(manifest_sha256=runtime.prepare(bound, provenance, common_auditor=common,
                                                     source_auditor=source_auditor)['manifest_sha256'])
        commit_json(root/'bound-runtime.json', bound)

        parent_auditor = partial(runtime.audit, common_auditor=common, source_auditor=source_auditor)
        continue_reference(c, bound, provenance, root, progress, parent_auditor=parent_auditor,
                           proof_session=session)
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, stage=stage, error=repr(exc), restart=False))
        raise


def run_renewed(c, provenance):
    target, bound = renew_before_scoring(c)
    root = Path(target['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', target)
    commit_json(root/'renewal.json', c)
    commit_json(root/'provenance.json', provenance)
    stage = 'reaudit_endpoint'

    def progress(value):
        nonlocal stage
        stage = value
        atomic_json(root/'progress.json', dict(stage=stage, unix=time.time()))

    try:
        progress(stage)
        runtime.audit(bound, provenance)
        commit_json(root/'bound-runtime.json', bound)
        continue_reference(target, bound, provenance, root, progress)
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, stage=stage, error=repr(exc), restart=False))
        raise


def detach(template, local, c, *, module=None):
    base = Path(c['output']+'-driver')
    check_paths({'driver': base})
    base.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', '-m', module or __spec__.name, '--config', str(Path(template).resolve()),
            '--local', str(Path(local).resolve()), '--action', 'run']
    with (base/'driver.log').open('x') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]
    record = dict(pid=child.pid, argv=argv, process_start_ticks=ticks)
    commit_json(base/'started.json', record)
    return record


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['validate', 'run', 'detach'], required=True)
    a = p.parse_args()
    config, provenance = load_config(a.config, a.local)
    renewed = 'original' in config
    if a.action != 'run':
        if renewed:
            _, bound = renew_before_scoring(config)
            runtime.audit(bound, provenance)
        else:
            validate(config)
    if a.action == 'run':
        (run_renewed if renewed else run)(config, provenance)
    elif a.action == 'detach':
        print(json.dumps(detach(a.config, a.local, config)))
    else:
        print(json.dumps(dict(status='successor_configuration_validated', npu_executed=False,
                              reuses_audited_endpoint=renewed)))
