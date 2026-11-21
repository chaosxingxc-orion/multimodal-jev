import json

import pytest

from tools.reports import context_stage2_reference_successor as recovery


@pytest.mark.parametrize(('committed', 'expected_replay', 'expected_work'), [
    (51, False, 108),
    (159, True, 1),
])
def test_resume_budgets_only_remaining_macros(tmp_path, committed, expected_replay, expected_work):
    (tmp_path/'reference-build-config.json').write_text(json.dumps(
        dict(expected_population=40513, macro_client={'max_queries': 256})))
    (tmp_path/'reference-prefix.json').write_text(json.dumps({'commits': [None] * committed}))
    plan = recovery.resume_plan(
        dict(package=str(tmp_path), operations_root=str(tmp_path), tag='attempt'),
        tmp_path, {'commits': committed}, {'deadline_unix': 12345}, {'minimum_seconds': 300}, 4200)
    assert plan['replay_last'] is expected_replay
    assert plan['expected_inference_macros'] == expected_work
    assert 300 < plan['minimum_seconds'] < 4500


def test_recovery_does_not_replace_uncollected_or_other_failed_stages(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    successor = tmp_path/'successor'
    operations = tmp_path/'operations'
    deployment = dict(host='worker', lease='/local/lease')
    location = dict(operations_root=str(operations), tag='reference')
    c = dict(output=str(successor), lease=dict(deadline_unix=12345, stop_grace_seconds=45),
             reference=dict(deployment=deployment, reference_operations=location))
    prior = dict(**deployment, **location, lease_deadline_unix=12345, lease_stop_grace_seconds=45,
                 reference_job={'parent': {'manifest_sha256': 'f'*64}})
    write(operations/'finalization/config.json', prior)
    write(operations/'finalization/terminal.json', {'exit_code': 1})
    write(successor/'bound-runtime.json', prior['reference_job']['parent'])
    write(operations/'reference-wait.json', {'exit_code': 1})
    write(successor/'terminal.json', {'exit_code': 1, 'stage': 'train_parity_and_reference'})
    assert recovery.stopped_reference(c) is None
    write(operations/'reference-collected.transport.json',
          {'status': 'transport_complete_independent_audit_required'})
    assert recovery.stopped_reference(c) == prior
    write(operations/'finalization/config.json', dict(prior, lease_deadline_unix=12346))
    with pytest.raises(ValueError, match='exact deployment'):
        recovery.stopped_reference(c)
    write(successor/'terminal.json', {'exit_code': 1, 'stage': 'stage2_windows'})
    assert recovery.stopped_reference(c) is None


def test_renewed_reference_resumes_on_new_deadline_only_after_fallback_reconciles(tmp_path, monkeypatch):
    from multimodal_jev import local_checkpoint
    from multimodal_jev.baseline import file_hash

    def pin(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=file_hash(path))

    old = tmp_path/'original'/'successor'
    fallback = tmp_path/'fallback'/'successor'
    output = tmp_path/'renewal'/'successor'
    settings = dict(io_workers=2, tag_suffix='-fallback', minimum_seconds=60,
                    legacy_engine_timeout_seconds=120, shutdown_timeout_seconds=150)
    lease = dict(deadline_unix=1000, stop_grace_seconds=45, reserve_seconds=100,
                 minimum_seconds={'parity_and_reference': 4200})
    deployment = dict(host='worker', lease='/local/lease')
    ops = dict(deployment, lease_deadline_unix=1000, lease_stop_grace_seconds=45,
               operations_root=str(old.parent/'operations'), tag='score',
               reference_job=dict(parent={'identity': 'fixed'}, output=str(old.parent/'package'),
                                  remote_root='/local/score'))
    prior = dict(output=str(old), lease=lease,
                 reference={'reference': dict(output=ops['reference_job']['output']),
                            'reference_operations': dict(operations_root=ops['operations_root'])},
                 view_pipeline=dict(deployment=deployment, trace_replay=dict(deployment, tag='trace', remote_root='/local/trace', source='/local/source'),
                                    window_operations={'image_cpu': {}}, windows={},
                                    objective_parent={'remote_root': '/local/objective'}),
                 cached_parent={'remote_root': '/local/cached'})
    predecessor = dict(config=pin(old/'config.json', prior),
                       driver=pin(old.with_name('successor-driver')/'started.json', {'pid': 1}))
    fallback_config = pin(fallback/'config.json', dict(output=str(fallback), predecessor=predecessor, recovery=settings))
    fallback_driver = pin(fallback.with_name('successor-driver')/'started.json', {'pid': 2})
    terminal = pin(fallback/'terminal.json', dict(exit_code=1, stage='prepare_reference_resume'))
    c = dict(output=str(output), predecessor=predecessor, renewed_lease={'deadline_unix': 2200},
             stopped_fallback=dict(config=fallback_config, driver=fallback_driver, terminal=terminal),
             recovery=dict(settings, tag_suffix='-renewed'))
    monkeypatch.setattr(recovery.time, 'time', lambda: 999)
    with pytest.raises(ValueError, match='lease expires'):
        recovery.validate(c)
    monkeypatch.setattr(recovery.time, 'time', lambda: 1100)
    monkeypatch.setattr(recovery, 'check_paths', lambda _: None)
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda _: None)
    waits, launched = [], []
    monkeypatch.setattr(recovery.endpoint, 'wait_exact', lambda record, path: waits.append(record['pid']))
    attempted = fallback.parent/'operations/score-fallback-attempt.json'
    pin(attempted, {})
    with pytest.raises(ValueError, match='newer handle'):
        recovery.reconcile_fallback(c, ops)
    attempted.unlink()
    waits.clear()
    monkeypatch.setattr(recovery, 'stopped_reference', lambda _: ops)
    monkeypatch.setattr(recovery.resume, 'prepare_prefix', lambda *a: {'receipt_sha256': 'prefix'})
    monkeypatch.setattr(recovery, 'resume_plan', lambda *a: {'replay_last': True, 'minimum_seconds': 90})
    monkeypatch.setattr(recovery.reference, 'prepare', lambda *a: {'manifest_sha256': 'package'})
    monkeypatch.setattr(recovery.operations, 'run_and_audit', lambda op, _: launched.append(op))
    monkeypatch.setattr(recovery.endpoint, 'reference_job', lambda op: op)
    monkeypatch.setattr(recovery.endpoint, 'publish_view', lambda target,*a: (target['lease'], {}))
    monkeypatch.setattr(recovery.endpoint, 'run_windows', lambda target, view, *a: {
        'exit_code': 0, 'deadline': view['deadline_unix']})
    recovery.run(c, {})
    assert waits == [1, 2]
    assert len(launched) == 1
    assert launched[0]['lease_deadline_unix'] == 2200
    assert launched[0]['lease'] == ops['lease']
    assert launched[0]['host'] == ops['host']
    assert launched[0]['reference_job']['parent'] == ops['reference_job']['parent']
    assert launched[0]['reference_job']['resume']['replay_last'] is True
    assert recovery.endpoint.read(output/'terminal.json')['deadline'] == 2200

    # If the fallback actually scored further before timing out, its collected
    # prefix is authoritative. Inputs may not fall back to the earlier archive.
    waits.clear()
    c['output'] = str(tmp_path/'next-lease'/'successor')
    terminal = pin(fallback/'terminal.json', dict(exit_code=1, stage='reference_resume'))
    c['stopped_fallback']['terminal'] = terminal
    original_transport = pin(old.parent/'operations/score-collected.transport.json', {})
    prefix = fallback/'prefix'
    original_index = pin(prefix/'reference-prefix.json', {'lineage': {'endpoint':318}, 'commits': ['old']})
    receipt = pin(prefix/'receipt.json', dict(original_transport_sha256=original_transport['sha256'],
                                              prefix_sha256=original_index['sha256'], commits=1))
    latest_spec = dict(ops['reference_job'], output=str(fallback.parent/'package'),
                       remote_root='/local/score-fallback', manifest_sha256='newer',
                       resume=dict(prefix_root=str(prefix), receipt_sha256=receipt['sha256'], replay_last=False),
                       legacy_engine_timeout_seconds=120, shutdown_timeout_seconds=150)
    fallback_ops = fallback.parent/'operations'
    latest = dict(ops, tag='score-fallback', operations_root=str(fallback_ops),
                  package=latest_spec['output'], remote_root=latest_spec['remote_root'],
                  manifest_sha256='newer', reference_job=latest_spec)
    pin(fallback/'reference-operations.json', latest)
    pin(fallback_ops/'finalization/config.json', latest)
    pin(fallback_ops/'finalization/terminal.json', dict(exit_code=1))
    pin(fallback_ops/'score-fallback-wait.json', dict(exit_code=137))
    with pytest.raises(ValueError, match='exact failed and collected'):
        recovery.reconcile_fallback(c, ops)
    pin(fallback_ops/'score-fallback-collected.transport.json',
        dict(status='transport_complete_independent_audit_required'))
    assert recovery.reconcile_fallback(c, ops) == latest

    waits.clear()
    launched.clear()
    sources = []
    def prepare_prefix(config, _, *, allow_empty=False):
        sources.append(config['prior_operations'])
        if allow_empty:
            dest = recovery.Path(config['output'])
            index = pin(dest/'reference-prefix.json', {
                'lineage': {'endpoint':318}, 'commits':['old','new']})
            ready = pin(dest/'receipt.json', {'prefix_sha256':index['sha256']})
            return dict(receipt_sha256=ready['sha256'], commits=2)
        return dict(receipt_sha256='original-prefix', commits=1)
    monkeypatch.setattr(recovery.resume, 'prepare_prefix', prepare_prefix)
    recovery.run(c,{})
    assert sources == [latest] and waits == [1,2]
    assert launched[0]['tag'] == 'score-fallback-renewed'
    assert launched[0]['reference_job']['resume']['receipt_sha256'] == file_hash(
        recovery.Path(c['output'])/'latest-prefix/receipt.json')
    assert launched[0]['lease_deadline_unix'] == 2200

    # A fallback killed before it restores the earlier prefix cannot make those
    # already committed blocks disappear.
    def empty_fallback(config, _, *, allow_empty=False):
        return {'commits':0} if allow_empty else {'commits':1,'receipt_sha256':'original-prefix'}
    monkeypatch.setattr(recovery.resume, 'prepare_prefix', empty_fallback)
    used, _, sealed = recovery.seal_resume_source(ops, latest, tmp_path/'earlier', settings, {})
    assert used == ops and sealed['receipt_sha256'] == 'original-prefix'

    def changed_fallback(config, _, *, allow_empty=False):
        assert allow_empty
        dest = recovery.Path(config['output'])
        index = pin(dest/'reference-prefix.json', {
            'lineage': {'endpoint':318}, 'commits':['changed','new']})
        ready = pin(dest/'receipt.json', {'prefix_sha256':index['sha256']})
        return dict(receipt_sha256=ready['sha256'], commits=2)
    monkeypatch.setattr(recovery.resume, 'prepare_prefix', changed_fallback)
    with pytest.raises(ValueError, match='changed an already committed reference block'):
        recovery.seal_resume_source(ops, latest, tmp_path/'changed', settings, {})
