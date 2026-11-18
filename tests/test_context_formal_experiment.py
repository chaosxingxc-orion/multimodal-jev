import json
from pathlib import Path

import pytest

from tools.jev import run_context_formal_experiment as module


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('.local/decision-objective-research').mkdir(parents=True)
    package=tmp_path/'package';(package/'evidence').mkdir(parents=True)
    (package/'source/schedule').mkdir(parents=True)
    (package/'source/schedule/MANIFEST.json').write_text('{}')
    initial=tmp_path/'initial';initial.mkdir()
    stage1=tmp_path/'stage1';(stage1/'run/policy-0634').mkdir(parents=True)
    (stage1/'run/policy-0634/MANIFEST.json').write_text('{}')
    steps=[0,*range(64,634,64),634]
    e=dict(release_sha256='data',evaluation=dict(checkpoints=steps))
    spec=dict(package=str(package),manifest_sha256='package-sha',initial_package=str(initial),stage1_root=str(stage1))
    for name,value in [('experiment',e),('stage1_proof',{'files':{}}),('template',{'release_sha256':'data'})]:
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));spec[name]=str(p)
    (package/'evidence/experiment.json').write_text(json.dumps(e))
    (package/'evidence/training_proof.json').write_text(json.dumps({'files':{}}))
    (package/'config.json').write_text(json.dumps(dict(source_policy_id='initial',algorithm='grpo_v1',
        schedule_sha256=module.sha(package/'source/schedule/MANIFEST.json'))))
    (package/'RELEASE.json').write_text(json.dumps(dict(source_package_manifest_sha256='initial-sha')))
    monkeypatch.setattr(module,'verify',lambda *a:dict(remote_root=module.PREFIX+'context-rl-grpo-test'))
    monkeypatch.setattr(module,'verify_source',lambda *a:dict(source_policy_id='initial'))
    return spec,tmp_path/'experiment',steps


def backend(monkeypatch,fail=None):
    calls=[]
    def invoke(name,args,log):
        args=list(map(str,args));calls.append((name,args));log.write_text(name)
        if name==fail:raise ValueError('simulated failure')
        def arg(key):return Path(args[args.index(key)+1])
        if name=='launch_context_formal_training':
            Path('.local/decision-objective-research/context-rl-grpo-test-launch.json').write_text('{}')
        elif name.startswith('collect_'):
            output=arg('--output');output.mkdir();output.with_suffix('.transport.json').write_text('{}')
        elif name=='prepare_context_rl_evaluation_package':
            output=arg('--output');output.mkdir();(output/'PACKAGE.json').write_text('{}')
            (output/'remote.txt').write_text(args[args.index('--remote-root')+1])
        elif name=='launch_context_rl_evaluation':
            remote=(arg('--package')/'remote.txt').read_text()
            p=Path('.local/decision-objective-research')/(Path(remote).name+'-launch.json')
            p.write_text(json.dumps({'config_sha256':Path(remote).name}))
        elif name.startswith('audit_'):
            arg('--output').write_text(json.dumps({'fixture_audit':name}))
    monkeypatch.setattr(module,'invoke',invoke)
    return calls


def test_default_plan_no_subprocess_or_output(experiment,monkeypatch):
    spec,output,steps=experiment
    def forbidden(*a):raise AssertionError('unexpected dispatch')
    monkeypatch.setattr(module,'invoke',forbidden)
    p=module.run(spec,output)
    assert p['evaluation_steps']==steps and not output.exists()


def test_complete_11_point_flow_same_source(experiment,monkeypatch):
    spec,output,steps=experiment;calls=backend(monkeypatch)
    result=module.run(spec,output,launch=True)
    assert result==dict(fixture_audit='audit_context_rl_curve')
    names=[name for name,args in calls]
    assert names[:3]==['launch_context_formal_training','collect_context_rl_training','audit_context_rl_training']
    assert names.count('launch_context_rl_evaluation')==11 and len(calls)==48
    curve=json.loads((output/'curve-spec.json').read_text())
    assert [x['step'] for x in curve['evaluations']]==steps
    assert len({x['collection'] for x in curve['evaluations']})==11
    for step in steps:
        point=json.loads((output/f'step-{step:04d}/audit-spec.json').read_text())
        assert point['rl_proof']==str(output/'training-audit.json')
        assert point['stage1_proof']==spec['stage1_proof'] and point['algorithm']=='grpo_v1'
    assert json.loads((output/'status.json').read_text())['phase']=='complete_training_and_selection_curve'


@pytest.mark.parametrize('failure',['collect_context_rl_training','audit_context_rl_training',
    'launch_context_rl_evaluation','collect_context_rl_selection','audit_context_rl_selection','audit_context_rl_curve'])
def test_failure_stops_and_preserves_no_relaunch(experiment,monkeypatch,failure):
    spec,output,_=experiment;calls=backend(monkeypatch,failure)
    with pytest.raises(ValueError,match='simulated'):module.run(spec,output,launch=True)
    assert calls[-1][0]==failure
    status=json.loads((output/'status.json').read_text())
    assert status['phase']=='stopped_preserve_handles' and status['restart'] is False
    before=len(calls)
    with pytest.raises(FileExistsError):module.run(spec,output,launch=True)
    assert len(calls)==before


def test_nonzero_training_terminal_still_collected(experiment,monkeypatch):
    # Launchers return only after exact terminal, even exit != 0. The audit
    # refuses failure after collection; no evaluation starts or run restarts.
    spec,output,_=experiment;calls=backend(monkeypatch,'audit_context_rl_training')
    with pytest.raises(ValueError):module.run(spec,output,launch=True)
    assert [x[0] for x in calls]==['launch_context_formal_training','collect_context_rl_training','audit_context_rl_training']
    assert (output/'training-collected.transport.json').exists()


@pytest.mark.parametrize('fault',['points','data','stage1','schedule','nested'])
def test_mixed_inputs_rejected_before_dispatch(experiment,monkeypatch,fault):
    spec,output,_=experiment;calls=backend(monkeypatch)
    if fault=='points':
        e=json.loads(Path(spec['experiment']).read_text());e['evaluation']['checkpoints']=[0,634]
        Path(spec['experiment']).write_text(json.dumps(e))
    elif fault=='data':Path(spec['template']).write_text(json.dumps({'release_sha256':'wrong'}))
    elif fault=='stage1':Path(spec['stage1_proof']).write_text('{}')
    elif fault=='schedule':(Path(spec['package'])/'source/schedule/MANIFEST.json').write_text('changed')
    else:output=Path(spec['package'])/'experiment'
    with pytest.raises(ValueError):module.run(spec,output,launch=True)
    assert not calls and not output.exists()
