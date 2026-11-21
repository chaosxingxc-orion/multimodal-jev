from pathlib import Path

import pytest

from tools.reports import context_stage2_view_successor as successor
from test_stage2_view_package import fixture, write


def test_continuation_builds_full_budget_with_exact_receipts(tmp_path, monkeypatch):
    _, package = fixture(tmp_path, monkeypatch)
    prepare = package['parent']
    parent = Path(prepare['root'])
    manifest = successor.endpoint.read(parent/'PREPARED.json')
    prep = write(parent/'PREPARED.json', dict(manifest,
        status='objective_training_candidate_prepared_only', parent_sha256='cached'))
    prior = tmp_path/'original'
    bound = package['bound_config']
    receipt = package['dataset_receipt']
    data = successor.endpoint.read_pin(receipt)
    old_windows = {w:dict(tag='old-'+w, package=str(tmp_path/('old-pkg-'+w)),
        operations_root=str(tmp_path/('old-ops-'+w)), remote_root='/data/old-'+w)
        for w in (*successor.FINITE, 'formal')}
    config = dict(output=str(prior), objective_parent=dict(output=str(parent), parent={'manifest_sha256':'cached'}),
        dataset=dict(stage2_output=bound['path'], receipts={'bind':receipt['path']},
                     manifest_sha256=data['manifest_sha256'], reference_job={}),
        deployment=dict(host='worker', lease='/local/lease'), window_operations={},
        window_package={k:package[k] for k in ('schema','objective_design')}, windows=old_windows,
        trace_replay=dict(tag='old-trace', output=str(tmp_path/'old-trace'), remote_root='/data/old-trace'))
    c = dict(output=str(tmp_path/'continued'), prior_pipeline=write(prior/'config.json',config),
        prior_terminal=write(prior/'terminal.json',{'exit_code':1}),
        prepared_inputs=dict(parent_sha256=prep['sha256'], bound_config=bound, dataset_receipt=receipt),
        trace_destination=dict(tag='new-trace', output=str(tmp_path/'new-trace'), remote_root='/data/new-trace'),
        windows=dict(formal=dict(tag='new-formal', package=str(tmp_path/'new-package'),
                                 operations_root=str(tmp_path/'new-ops'), remote_root='/data/new-formal')),
        lease=dict(deadline_unix=2000, stop_grace_seconds=45, reserve_seconds=100,
                   minimum_seconds=dict(prefix=100,recovery=100,tail=100), formal_step_multiplier=1.12),
        io_workers=2)
    assert successor.endpoint.read_pin(bound)['batch_config']['epochs'] == 2
    sandbox = tmp_path/'sample-prefix'
    from tools.reports import prepare_stage2_view_package as window
    window.run(dict(package, parent=dict(prepare, manifest_sha256=prep['sha256']), output=str(sandbox)),{})
    checks = successor.endpoint.read(sandbox/'config.json')['view_lineage']
    proofs = {}
    for kind in successor.FINITE:
        audit = dict(status='stage2_view_'+kind+'_outputs_audited',
            container_id=kind, lineage=checks, world_size=2, updates=[{'slowest_rank_step_seconds':20}]*
            (1 if kind=='recovery' else 2), storage_audit={'actual_readonly_mounts_verified':True},
            fresh_process_replay_verified=True, prefix_audit_sha256=proofs.get('prefix',{}).get('sha256'),
            replay={'exact_objective_traces':True})
        if kind != 'tail':
            replay = dict(replay_device='npu:0', updates=[0 if kind == 'prefix' else 1])
            write(Path(config['trace_replay']['output'])/(('a' if kind == 'prefix' else 'b')*16)/'accepted.json',
                  replay)
            audit['trace_audit'] = dict(replay, provenance={'source':kind})
        proofs[kind] = write(prior/kind/'audit.json', audit)
        write(prior/(kind+'-window.json'), {'window':kind})
    inputs = {k:v for k,v in c.items() if k != 'prepared_inputs'}
    inputs['local_output'] = str(tmp_path/'continued-local.json')
    c = successor.assemble(inputs)
    assert c['prepared_inputs'] == dict(parent_sha256=prep['sha256'], bound_config=bound, dataset_receipt=receipt)
    assert c['completed']['prefix']['backend_replay']['path'].endswith('aaaaaaaaaaaaaaaa/accepted.json')
    assert c['completed']['recovery']['backend_replay']['path'].endswith('bbbbbbbbbbbbbbbb/accepted.json')
    assert 'backend_replay' not in c['completed']['tail']
    monkeypatch.setattr(successor, 'completed_window', lambda kind,*a,**kw:dict(audit=proofs[kind]))
    monkeypatch.setattr(successor.endpoint.time, 'time', lambda:1000)
    monkeypatch.setattr(successor, 'check_paths', lambda _:None)
    checked = {k: {'audit': v} for k, v in proofs.items()}
    required = successor.forecast(c['lease'], data['consumer'], checked, now=1000)
    too_short = dict(c['lease'], deadline_unix=1000 + required['required_seconds_including_reserve'] - 1)
    assert successor.forecast(too_short, data['consumer'], checked, now=1000)['fits_complete_run'] is False
    partial = successor.forecast(c['lease'], data['consumer'], {'prefix': checked['prefix']}, now=1000)
    assert partial['remaining'] == ['recovery', 'tail', 'formal']
    assert partial['required_seconds_including_reserve'] is None
    assert partial['fits_complete_run'] is None
    seen = []
    def run(op, _):
        cfg = successor.endpoint.read(Path(op['training_package']['output'])/'config.json')
        seen.append((op['window'],cfg['stop_update'],cfg['view_admission']))
        result = dict(status='stage2_view_formal_outputs_audited', updates=[None]*cfg['stop_update'])
        write(Path(op['output'])/'audit.json', result)
        return result
    monkeypatch.setattr(successor.pipeline.execution, 'run', run)
    successor.run(c,{})
    assert seen == [('formal', data['consumer']['stop_update'], {k:v['sha256'] for k,v in proofs.items()})]
    write(Path(old_windows['formal']['operations_root'])/'old-formal-attempt.json', {})
    with pytest.raises(ValueError, match='reconcile its exact handle'):
        successor.prepare(c)

    trace_root = Path(config['trace_replay']['output'])
    (trace_root/'aaaaaaaaaaaaaaaa'/'accepted.json').rename(trace_root/'hidden.json')
    with pytest.raises(ValueError, match='one exact prior backend replay required for prefix'):
        successor.assemble(inputs)
    (trace_root/'hidden.json').rename(trace_root/'aaaaaaaaaaaaaaaa'/'accepted.json')
    (prior/'recovery'/'audit.json').unlink()
    with pytest.raises(ValueError, match='contiguous finite prefix'):
        successor.assemble(inputs)
