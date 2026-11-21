import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev import local_checkpoint
from tools.reports import context_stage2_endpoint_successor as successor


def pin(path, data):
    path.write_text(json.dumps(data))
    return {'path': str(path), 'sha256': file_hash(path)}


def test_preflight_precedes_wait_and_source_is_rebuilt_once(tmp_path, monkeypatch):
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda _: None)
    previous = tmp_path/'previous'
    previous.mkdir()
    terminal = pin(previous/'terminal.json', dict(exit_code=0, status='formal_infonce_audited_curve_bound'))
    plan = pin(previous/'source-evaluation-plan.json', {'endpoint': 160})
    final = dict(finalization_root=str(previous), operations_config='ops', operations_local='local',
                 initial_policy='initial')
    monkeypatch.setattr(successor, 'validate', lambda _: (final, {'pid': 123}))
    order = []

    def common(spec, p):
        order.append('common')
        return {'provenance': p}

    def wait(prior, path):
        assert prior == {'pid': 123} and str(path) == terminal['path']
        order.append('wait')

    def reconstruct(spec):
        order.append('source')
        return successor.read_pin(plan)

    def prepare_source(spec, p, *, reconstruct):
        assert reconstruct(spec) == {'endpoint': 160}
        return {'manifest_sha256': 'source'}

    def prepare_runtime(spec, p, *, common_auditor, source_auditor):
        assert common_auditor(spec['runtime'], p) == {'provenance': p}
        source_auditor(spec['source'], p)
        return {'manifest_sha256': 'runtime'}

    def audit_runtime(spec, p, *, common_auditor, source_auditor):
        common_auditor(spec['runtime'], p)
        source_auditor(spec['source'], p)

    def continuation(c, bound, p, root, progress, *, parent_auditor, proof_session):
        parent_auditor(bound, p)
        Path(plan['path']).write_text('{"endpoint": 159}')
        parent_auditor(bound, p)

    monkeypatch.setattr(successor, 'wait_exact', wait)
    monkeypatch.setattr(successor.source, 'reconstruct_plan', reconstruct)
    monkeypatch.setattr(successor.source, 'prepare', prepare_source)
    monkeypatch.setattr(successor.source, 'audit', lambda spec, p, *, reconstruct: reconstruct(spec))
    monkeypatch.setattr(successor.runtime, 'audit_runtime', common)
    monkeypatch.setattr(successor.runtime, 'prepare', prepare_runtime)
    monkeypatch.setattr(successor.runtime, 'audit', audit_runtime)
    monkeypatch.setattr(successor, 'continue_reference', continuation)
    config = dict(output=str(tmp_path/'out'), runtime={'runtime': {}}, source={},
                  reference={'reference_operations': {'admission_execution': {'io_workers': 2}}})
    with pytest.raises(ValueError, match='frozen proof dependencies changed'):
        successor.run(config, {})
    assert order == ['common', 'wait', 'source']
    assert successor.read(tmp_path/'out/terminal.json')['restart'] is False


def test_wrong_finalizer_and_expired_lease_are_rejected(tmp_path, monkeypatch):
    final = tmp_path/'final'
    final.mkdir()
    driver = tmp_path/'final-driver'
    driver.mkdir()
    config = pin(final/'config.json', dict(finalization_root=str(final),
        operations_config='ops', operations_local='local'))
    handle = pin(driver/'started.json', {'pid': 123})
    deploy = dict(host='host', lease='/local/lease', driver_root='/local/driver')
    lease = dict(deadline_unix=2000, reserve_seconds=100, stop_grace_seconds=45,
        minimum_seconds=dict(parity_and_reference=500,prefix=100,recovery=100,tail=100),
        formal_step_multiplier=1.12)
    monkeypatch.setattr(successor, 'load_config', lambda *args: (dict(deploy,lease_deadline_unix=2000), {}))
    monkeypatch.setattr(successor, 'check_paths', lambda _: None)
    c = dict(predecessor=dict(config=config,driver=handle), lease=lease, output=str(tmp_path/'successor'),
        reference=dict(deployment=deploy, parity_operations=dict(finite_inference_dispatch_allowed=True),
            reference_operations=dict(reference_dispatch_allowed=True,single_admission_launch=True)),
        view_pipeline=dict(deployment=deploy,trace_replay=deploy,window_operations={'dispatch':True}))
    assert successor.validate(c)[1] == {'pid': 123}
    c['predecessor']['driver'] = pin(tmp_path/'started.json', {'pid': 123})
    with pytest.raises(ValueError, match='exact Stage1 finalizer identity'):
        successor.validate(c)
    with pytest.raises(ValueError, match='cannot fit parity'):
        successor.check_budget(lease, 500, stage='parity', now=2000)


def test_formal_budget_reads_pinned_slowest_steps_and_all_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda _: None)
    monkeypatch.setattr(successor.time, 'time', lambda: 1000)
    prefix = pin(tmp_path/'prefix.json', {'updates': [{'slowest_rank_step_seconds': 18}]})
    tail = pin(tmp_path/'tail.json', {'updates': [{'slowest_rank_step_seconds': 31}]})
    lease = dict(deadline_unix=2200,reserve_seconds=100,minimum_seconds={},formal_step_multiplier=1.12)
    guard = successor.window_guard(lease, {'stop_update': 40}, tmp_path)
    with pytest.raises(ValueError, match='cannot fit formal'):
        guard('formal',dict(prefix=prefix,tail=tail))
    assert json.loads((tmp_path/'formal-budget.json').read_text())['minimum_seconds'] == 1389


def test_renewal_reuses_audited_endpoint_only_before_dispatch(tmp_path, monkeypatch):
    base = tmp_path/'old'
    old = base/'successor'
    old.mkdir(parents=True)
    driver = base/'successor-driver'
    driver.mkdir()
    deploy = dict(host='host', lease=str(base/'hardware.lock'), driver_root='/local/driver')
    lease = dict(deadline_unix=1000, reserve_seconds=100, stop_grace_seconds=45,
        minimum_seconds=dict(parity_and_reference=500,prefix=100,recovery=100,tail=100),
        formal_step_multiplier=1.12)
    final = tmp_path/'final'
    final.mkdir()
    final_driver = tmp_path/'final-driver'
    final_driver.mkdir()
    final_config = dict(finalization_root=str(final), operations_config='ops', operations_local='local',
                        initial_policy={'policy':'step-zero'})
    src = dict(output=str(base/'source'))
    runtime = dict(output=str(base/'runtime'), remote_root='/local/runtime')
    plan = pin(final/'source-evaluation-plan.json', {'reference_endpoint':318})
    pin(final/'terminal.json', dict(exit_code=0, status='formal_infonce_audited_curve_bound'))
    binding = dict(runtime, manifest_sha256='runtime-audit-id', source=dict(src,
        manifest_sha256='source-audit-id', source_plan=plan['path'], source_plan_sha256=plan['sha256'],
        source_finalization={k:v for k,v in final_config.items() if k != 'finalization_root'}))
    pin(old/'bound-runtime.json', binding)

    loc = lambda name: dict(output=str(base/name), remote_root='/local/'+name)
    cpu = dict(path=str(base/'cpu/report.json'), sha256='cpu-hash')
    c = dict(output=str(old), lease=lease, source=src, runtime=runtime, cached_parent=loc('cached'),
        predecessor=dict(config=pin(final/'config.json', final_config),
                         driver=pin(final_driver/'started.json', {'pid':1})),
        reference=dict(deployment=deploy, output=str(base/'scoring'), chain_lock=str(base/'scoring.lock'),
            parity=loc('parity-package'), reference=loc('reference-package'),
            parity_operations=dict(operations_root=str(base/'parity-ops'), tag='parity',
                                   finite_inference_dispatch_allowed=True),
            reference_operations=dict(operations_root=str(base/'reference-ops'), tag='reference',
                                      reference_dispatch_allowed=True, single_admission_launch=True)),
        view_pipeline=dict(deployment=deploy, output=str(base/'windows'), objective_parent=loc('objective'),
            trace_replay=dict(deploy, **loc('replay'), tag='replay', source=str(base/'cpu/source')),
            dataset=dict(output=str(base/'view'), stage2_output=str(base/'bound.json'),
                         receipts=dict(bind=str(base/'bind.json'))),
            windows={name:dict(package=str(base/name), operations_root=str(base/(name+'-ops')),
                               remote_root='/local/'+name, tag=name)
                     for name in ('prefix','recovery','tail','formal')},
            window_operations=dict(dispatch=True, image_cpu=cpu)))
    renewed = dict(original=dict(config=pin(old/'config.json', c),
        driver=pin(driver/'started.json', {'pid':2}),
        terminal=pin(old/'terminal.json', dict(exit_code=1, stage='train_parity_and_reference',
            error="ValueError('lease cannot fit train_parity_and_reference: needs 500s plus reserve')"))),
        deadline_unix=2000, tag_suffix='-renewal', output=str(tmp_path/'new/successor'))
    monkeypatch.setattr(successor, 'load_config', lambda *a: (dict(deploy,lease_deadline_unix=1000),{}))
    monkeypatch.setattr(successor, 'check_paths', lambda _:None)
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda _:None)
    monkeypatch.setattr(successor, 'wait_exact', lambda rec, terminal: (
        rec == {'pid': 2} or pytest.fail('wrong driver')))
    monkeypatch.setattr(successor.time, 'time', lambda: 1100)
    with pytest.raises(ValueError, match='after the original lease'):
        successor.renew_before_scoring(renewed, now=999)
    (base/'reference-ops').mkdir()
    with pytest.raises(ValueError, match='reconcile its exact handles'):
        successor.renew_before_scoring(renewed)
    (base/'reference-ops').rmdir()
    with pytest.raises(ValueError, match='cannot fit renewed_train_parity_and_reference'):
        successor.renew_before_scoring(renewed, now=1400)

    calls = []
    def audit(spec, provenance):
        assert spec == binding
        calls.append('reaudit')
    def continue_reference(target, spec, provenance, root, progress):
        assert spec == binding
        assert target['reference']['deployment'] == deploy
        assert target['view_pipeline']['window_operations']['image_cpu'] == cpu
        assert target['view_pipeline']['windows']['formal']['remote_root'] == '/local/formal-renewal'
        assert target['lease']['deadline_unix'] == 2000
        assert str(root) == renewed['output']
        calls.append('reference')
    monkeypatch.setattr(successor.runtime, 'audit', audit)
    monkeypatch.setattr(successor, 'continue_reference', continue_reference)
    successor.run_renewed(renewed, {})
    assert calls == ['reaudit', 'reference']
    assert json.loads((Path(renewed['output'])/'bound-runtime.json').read_text()) == binding
    assert file_hash(old/'config.json') == renewed['original']['config']['sha256']
