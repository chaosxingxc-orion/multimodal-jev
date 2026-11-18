import json
from pathlib import Path

import pytest

from tools.jev import run_context_baseline_sequence as module


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(experiment='frozen', training_root='stage1', training_proof='proof')))
    code = tmp_path/'pinned.py';code.write_text('original')
    spec = dict(pins={str(code):module.sha(code)}, dependencies=[dict(status='upstream')],
        source_spec=str(source), acceptance_spec='acceptance', selection_template='template',
        runs=[dict(algorithm=a, remote_root=a) for a in ('grpo_v1','rloo_v1')])
    calls=[]
    monkeypatch.setattr(module, 'wait_dependency', lambda entry:calls.append(('wait', entry)))
    def invoke(name, args, log):
        args=list(map(str,args));calls.append((name,args));Path(log).write_text(name)
        out=Path(args[args.index('--output')+1]);out.mkdir()
        if name=='prepare_context_stage2_source':(out/'PACKAGE.json').write_text('{}')
        elif name=='prepare_context_formal_package':(out/'PREPARED.json').write_text('{}')
        else:(out/'status.json').write_text(json.dumps(dict(phase='complete_training_and_selection_curve')))
    monkeypatch.setattr(module, 'invoke', invoke)
    return spec,tmp_path/'sequence',calls,code


def test_same_source_two_algorithms_after_dependencies(setup):
    spec,out,calls,_=setup;module.run(spec,out)
    assert [x[0] for x in calls]==['wait','prepare_context_stage2_source',
        'prepare_context_formal_package','run_context_formal_experiment',
        'prepare_context_formal_package','run_context_formal_experiment']
    experiments=[json.loads((out/(a+'-experiment.json')).read_text()) for a in ('grpo_v1','rloo_v1')]
    assert experiments[0]['initial_package']==experiments[1]['initial_package']
    assert experiments[0]['stage1_proof']==experiments[1]['stage1_proof']
    assert json.loads((out/'status.json').read_text())['phase']=='both_algorithm_curves_collected'
    with pytest.raises(FileExistsError):module.run(spec,out)
    assert len(calls)==6


@pytest.mark.parametrize('failure',['dependency','drift','publish','experiment'])
def test_stop_without_dispatch_or_retry(setup,monkeypatch,failure):
    spec,out,calls,code=setup
    def wait(entry):
        calls.append(('wait',entry))
        if failure=='dependency':raise ValueError('predecessor failed')
        if failure=='drift':code.write_text('changed')
    monkeypatch.setattr(module,'wait_dependency',wait)
    original=module.invoke
    def invoke(name,args,log):
        if (failure=='publish' and name=='prepare_context_formal_package'
                or failure=='experiment' and name=='run_context_formal_experiment'):
            raise ValueError('simulated failure')
        original(name,args,log)
    monkeypatch.setattr(module,'invoke',invoke)
    with pytest.raises(ValueError):module.run(spec,out)
    assert json.loads((out/'status.json').read_text())['restart'] is False
    assert not (out/'rloo_v1-experiment.json').exists()
    if failure in ('dependency','drift'):assert len(calls)==1


def test_failed_predecessor_rejected_with_real_lock(tmp_path):
    status=tmp_path/'status.json';status.write_text('{"phase":"failed"}')
    with pytest.raises(ValueError,match='predecessor'):
        module.wait_dependency(dict(lock=str(tmp_path/'lock'),status=str(status),success_phase='complete'))


def test_preexisting_drift_has_no_output(setup):
    spec,out,calls,code=setup;code.write_text('changed')
    with pytest.raises(ValueError,match='changed'):module.run(spec,out)
    assert not out.exists() and not calls
