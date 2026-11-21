import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.reports import context_stage1_start_pipeline as pipeline


def test_phase_transition_reuses_cache_and_reaches_finite_prepare(tmp_path,monkeypatch):
    from tools.reports import context_instruction_operations
    launch=tmp_path/'cache-launch.json'
    launch.write_text(json.dumps(dict(container_id='cid',config=dict(host='host',image='image',remote_root='/cache'))))
    def remote(argv,**kw):
        command=argv[-1]
        if 'docker wait' in command:return '0\n'
        if 'docker inspect' in command:return json.dumps([dict(Image='image',State=dict(ExitCode=0))])
        if 'MANIFEST.json' in command:return json.dumps(dict(recordings=40513,fresh_audio_checked=32,max_tokens=3115)).encode()
        return b'{"exit_code":0}'
    monkeypatch.setattr(pipeline.subprocess,'check_output',remote)
    report=tmp_path/'cpu.json'
    report.write_text(json.dumps(dict(counts=dict(tests=1,failures=0,errors=0,skipped=0),
        image=context_instruction_operations.lifecycle.IMAGE,state=dict(ExitCode=0,Status='exited'),terminal=dict(exit_code=0))))
    monkeypatch.setattr(pipeline,'cpu_source',lambda *a,**k:('/tested',{}))
    def prepare(c,p,*,tested_runtime):
        assert c['project']=='/tested' and tested_runtime=={}
        assert c['encoded_audio_cache']=='/cache/run/cache'
        assert '/cache/run/MANIFEST.json' in c['external_pins']
        raise RuntimeError('finite prepare reached')
    monkeypatch.setattr(context_instruction_operations,'prepare',prepare)
    c=dict(output=str(tmp_path/'output'),cache_launch=str(launch),cpu_report=str(report),cpu_expected_tests=1,io_workers=1,package={'external_pins':{}})
    with pytest.raises(RuntimeError,match='finite prepare reached'):pipeline.run(c,{})
    assert json.loads((tmp_path/'output/progress.json').read_text())['stage']=='finite_prepare'
    assert 'finite prepare reached' in json.loads((tmp_path/'output/terminal.json').read_text())['error']


@pytest.mark.parametrize('available, launches', [(36900, False), (38040, True)])
def test_formal_launch_uses_complete_stage1_budget_and_hard_deadline(tmp_path, monkeypatch, available, launches):
    from tools.reports import (
        audit_context_infonce_training as audit,
        context_infonce_formal as formal,
        context_infonce_recovery as replay,
        context_infonce_tail_operations as tail,
        context_instruction_operations as finite,
    )

    monkeypatch.setattr(pipeline.time, 'time', lambda: 1000)
    monkeypatch.setattr(pipeline, 'commit_json', lambda p, data: Path(p).write_text(json.dumps(data)))
    monkeypatch.setattr(pipeline, 'cpu_source', lambda *a, **k: ('/tested', {}))
    monkeypatch.setattr(pipeline, 'input_cache', lambda *a: ('/cache', '/cache/MANIFEST.json', 'cache-id'))
    report = tmp_path/'cpu.json'
    report.write_text(json.dumps(dict(counts=dict(tests=1, failures=0, errors=0, skipped=0),
        image=finite.lifecycle.IMAGE, state=dict(ExitCode=0, Status='exited'), terminal=dict(exit_code=0))))
    for module in (finite, replay, tail):
        monkeypatch.setattr(module, 'prepare', lambda *a, **k: {'manifest_sha256': 'finite'})
        monkeypatch.setattr(module, 'launch', lambda *a: None)
    measured = lambda a, b: {'updates': [{'slowest_rank_seconds': v} for v in (a, b)]}
    monkeypatch.setattr(audit, 'audit_collected', lambda c: measured(90, 93))
    monkeypatch.setattr(replay, 'audit', lambda c: {})
    monkeypatch.setattr(tail, 'audit', lambda c: measured(95, 91))
    monkeypatch.setattr(formal, 'prepare', lambda *a: {'manifest_sha256': 'formal', 'total_updates': 318})
    called = []

    def launch(config, _):
        called.append(config['lease_deadline_unix'])
        assert config['lease_stop_grace_seconds'] == 45
        return dict(status='collected_independent_training_audit_required', exit_code=0, container_id='cid')

    monkeypatch.setattr(formal, 'launch', launch)
    monkeypatch.setattr(formal, 'audit', lambda c: dict(status='formal_infonce_artifacts_audited',
                                                    updates=list(range(318)), container_id='cid'))
    output = tmp_path/'run'
    deadline = 1000 + available
    c = dict(output=str(output), cpu_report=str(report), cpu_expected_tests=1, io_workers=1,
             package=dict(external_pins={}, initial_policy_local='/policy', design='/design'),
             not_after_unix=deadline, walltime_multiplier=1.2, release_reserve_seconds=900,
             lease_stop_grace_seconds=45, deployment={}, finite={}, replay={}, tail={}, formal={},
             finite_template='/finite-template', recovery_template='/recovery-template',
             **{f'{stage}_local': str(tmp_path/(stage+'.json')) for stage in ('finite', 'replay', 'tail', 'formal')})
    if launches:
        pipeline.run(c, {})
        assert called == [deadline]
    else:
        with pytest.raises(ValueError, match='expires before this Stage1 window can finish'):
            pipeline.run(c, {})
        assert called == []
    budget = json.loads((output/'formal-lease-budget.json').read_text())
    assert budget['estimate_seconds'] == 36252
    assert budget['required_seconds'] == 37152
    assert budget['planned_steps'] == 318
    assert budget['fits_window'] is launches
