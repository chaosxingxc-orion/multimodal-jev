import json
from pathlib import Path

import pytest

from tools.jev import context_formal_acceptance as module


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate';candidate.mkdir()
    (candidate / 'trainer.py').write_text('# fixture')
    manifest = dict(files={'trainer.py':module.sha(candidate / 'trainer.py')})
    (candidate / 'PREPARED.json').write_text(json.dumps(manifest))
    spec = dict(candidate_package=candidate, candidate_manifest_sha256=module.sha(candidate / 'PREPARED.json'))
    for i, role in enumerate(('continuous','fresh_replay','checkpoint_frequency')):
        root = tmp_path / role;root.mkdir();files = {}
        def put(name, value):
            p = root / name;p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value));files[name] = module.sha(p)
        c = dict(start_update=1 if role == 'fresh_replay' else 0, stop_update=2,
            checkpoint_interval=64 if role == 'checkpoint_frequency' else 1,
            engineering_prefix=True, training_dispatch_allowed=False, stage='task', algorithm='grpo_v1',
            physical_devices=[4,5,6,7], global_batch=128, microbatch=8,
            source_policy_id='same', source_contract={}, base_id='base', runtime_profile='runtime',
            release_sha256='data', schedule_sha256='schedule', behavior_bounds={})
        put('config.json', c);put('service-config.json', {})
        put('PREPARED.json', manifest)
        put('supervision/terminal.json', dict(passed=True, actor_started=True, forced_cleanup=False,
            actor_exit_code=0, service_exit_code=0, child_exit_codes=[0,0]))
        put('service/terminal.json', dict(exit_code=0))
        put('run/result.json', dict(passed=True, start_update=c['start_update'], stop_update=2,
                                   learning_population=40513, schedule_updates=634))
        report = dict(passed=True, actor_container=str(i), service_container=str(i), image=module.IMAGE,
            state=dict(Status='exited', ExitCode=0, OOMKilled=False), bad_pins=[], bad_package=[], files=files)
        launch = dict(container_id=str(i), image=module.IMAGE, config_sha256=files['config.json'],
                      service_config_sha256=files['service-config.json'], manifest_sha256=files['PREPARED.json'])
        rp = tmp_path / (role+'-report.json');rp.write_text(json.dumps(report))
        lp = tmp_path / (role+'-launch.json');lp.write_text(json.dumps(launch))
        spec[role] = dict(collection=root, report=rp, launch=lp)
    # This suite exercises evidence binding, not NPU or tensor comparisons.
    # The existing comparison modules have independent actual tensor tests.
    monkeypatch.setattr(module, 'verify_update', lambda *args: {'fixture':'replay'})
    monkeypatch.setattr(module, 'compare', lambda *args: {'fixture':'checkpoint'})
    monkeypatch.setattr(module, 'lifecycle', lambda *args, **kw: {'fixture':'lifecycle'})
    return spec


def test_rebind_and_reexecute_comparisons(evidence):
    result = module.verify(**evidence)
    assert len(set(result['containers'].values())) == 3
    assert result['replay'] == {'fixture':'replay'}
    assert result['status'] == 'accepted_runtime_evidence_only_no_dispatch'


@pytest.mark.parametrize('fault', ['live','failure','same_handle','bad_pins','changed_asset',
    'launch','wrong_candidate','candidate_mutation','source','partial','comparison'])
def test_reject_incomplete_or_unmatched_evidence(evidence, fault, monkeypatch):
    entry = evidence['fresh_replay'];rp = entry['report'];lp = entry['launch'];root = entry['collection']
    report, launch = json.loads(rp.read_text()), json.loads(lp.read_text())
    if fault == 'live':report['state']['Status'] = 'running'
    elif fault == 'failure':report['state']['ExitCode'] = 1
    elif fault == 'same_handle':
        report['actor_container'] = report['service_container'] = launch['container_id'] = '0'
    elif fault == 'bad_pins':report['bad_pins'] = ['changed']
    elif fault == 'changed_asset':(root / 'config.json').write_text('{}')
    elif fault == 'launch':launch['config_sha256'] = 'wrong'
    elif fault == 'wrong_candidate':evidence['candidate_manifest_sha256'] = 'wrong'
    elif fault == 'candidate_mutation':(evidence['candidate_package'] / 'trainer.py').write_text('changed')
    elif fault in ('source','partial'):
        p = root / 'config.json';c = json.loads(p.read_text())
        c['source_policy_id' if fault == 'source' else 'stop_update'] = 'different'
        p.write_text(json.dumps(c));report['files']['config.json'] = launch['config_sha256'] = module.sha(p)
    elif fault == 'comparison':
        def reject(*args):raise ValueError('Adam differs')
        monkeypatch.setattr(module, 'verify_update', reject)
    rp.write_text(json.dumps(report));lp.write_text(json.dumps(launch))
    with pytest.raises(ValueError):module.verify(**evidence)
