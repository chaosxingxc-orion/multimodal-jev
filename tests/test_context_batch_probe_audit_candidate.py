import copy
import pytest
from test_context_batch_measurement_candidate import measured
from test_context_batched_service_candidate import engine_case
from tools.reports.audit_context_batch_probe_candidate import audit_measurements


def inputs():
    _, rows, _ = engine_case()
    return measured('serial'), measured('batched'), rows


def check(a, b, rows):
    return audit_measurements(a, b, rows, 'a'*64, 'a'*64, 4, 16384)


def test_independent_measurement_reconstruction():
    a, b, rows = inputs()
    result = check(a, b, rows)
    assert result['serial_over_batched_time_ratio'] == 2
    assert result['actual_call_reduction']


@pytest.mark.parametrize('fault', ['shared_bad_sample', 'shared_wrong_input', 'shared_probability_drift',
    'same_count_wrong_lengths', 'bad_warmup', 'bad_order', 'nan_time', 'bad_memory', 'missing_pass'])
def test_false_success_rejected_even_when_both_modes_agree(fault):
    a, b, rows = inputs()
    for report in (a, b):
        p = report['passes'][1]
        if fault == 'shared_bad_sample': p['sampling'][0]['seed'] += 1
        if fault == 'shared_wrong_input': p['records'][0]['request_identity']['positions'] = [0]
        if fault == 'shared_probability_drift': p['records'][0]['prob'][0] = .9
        if fault == 'same_count_wrong_lengths': p['telemetry']['counts']['language'] = [[8], [6]]
        if fault == 'bad_warmup': report['passes'][0]['telemetry']['counts']['language'] = []
        if fault == 'bad_order': report['passes'][3]['order'] = [0, 1]
        if fault == 'nan_time': p['seconds'] = float('nan')
        if fault == 'bad_memory': p['telemetry']['peak_allocated_bytes'] = 100
        if fault == 'missing_pass': report['passes'].pop()
    with pytest.raises(ValueError): check(a, b, rows)


def test_collected_artifacts_and_transport_are_bound(tmp_path, monkeypatch):
    import hashlib
    import json
    import torch
    from multimodal_jev.context_rollout_batch import bind_rollout
    from test_context_bulk_rpc_candidate import row, record
    from tools.reports import audit_context_batch_probe_candidate as module
    rows = [row(i+3) for i in range(128)]
    prepared = tmp_path/'prepared'; package = prepared/'package'; package.mkdir(parents=True)
    torch.save(rows, package/'inputs.pt')
    config = dict(policy=dict(policy_id='a'*64, max_batch_queries=4, max_batch_tokens=16384),
                  input_sha256=module.sha(package/'inputs.pt'))
    (package/'config.json').write_text(json.dumps(config))
    receipt = dict(image='image', remote_root='/remote', files_sha256='sealed')
    # The real sealed package is validated separately; only this boundary is a fixture.
    monkeypatch.setattr(module, 'validate', lambda path: (receipt, config))
    collection = tmp_path/'collected'; root = collection/'evaluation'; root.mkdir(parents=True)
    (collection/'config.json').write_bytes((package/'config.json').read_bytes())
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    launch = dict(container_id='container', image='image', remote_root='/remote',
                  files_sha256='sealed', config_sha256=module.sha(package/'config.json'))
    launch_path = tmp_path/'launch.json'; write(launch_path, launch)
    write(root/'source.json', dict(files_sha256='sealed'))
    for name in ('terminal.json','serial-exit.json','batched-exit.json','serial/terminal.json','batched/terminal.json'):
        write(root/name, dict(exit_code=0))
    reports = {}
    for mode in ('serial', 'batched'):
        records = [record('a'*64, r) for r in rows]
        samples = [bind_rollout(rec, **r, policy_id='a'*64, group_size=16, seed=317+i)['receipt']
                   for i, (r, rec) in enumerate(zip(rows, records))]
        passes = []
        for index, kind in enumerate(['warmup','timed','timed','reversed']):
            order = list(range(128)) if index != 3 else list(reversed(range(128)))
            lengths = [i+3 for i in order]
            size = 1 if mode == 'serial' else 4
            groups = [lengths[i:i+size] for i in range(0, 128, size)]
            passes.append(dict(kind=kind, order=order, seconds=2 if mode=='serial' else 1,
                records=records, sampling=samples, telemetry=dict(counts=dict(language=groups),
                    peak_allocated_bytes=10, peak_reserved_bytes=20)))
        reports[mode] = dict(mode=mode, policy_id='a'*64, input_sha256=config['input_sha256'], rows=128, passes=passes)
        write(root/mode/'measurement.json', reports[mode])
    write(root/'comparison.json', dict(exact_probabilities_and_G16=True, rows=128,
        actual_language_calls=[[128,128],[32,32]], actual_call_reduction=True,
        serial_seconds=2, batched_seconds=1, time_ratio=2))
    archive = collection.with_suffix('.tar'); archive.write_bytes(b'fixture archive')
    transport = dict(status='transport_complete_independent_audit_required',
        container_id='container', image='image', remote_root='/remote',
        state=dict(Status='exited', ExitCode=0, OOMKilled=False),
        archive_sha256=module.sha(archive),
        files={str(p.relative_to(collection)):module.sha(p) for p in collection.rglob('*') if p.is_file()})
    write(collection.with_suffix('.transport.json'), transport)
    result = module.audit(prepared, collection, launch_path)
    assert result['rows'] == 128 and result['actual_call_reduction']
    # Source-complete collection cannot silently lose a log or change a measurement.
    write(root/'serial/measurement.json', {})
    with pytest.raises(ValueError, match='collected files differ'):
        module.audit(prepared, collection, launch_path)
