import copy
import json
from pathlib import Path

import pytest

from tools.reports import run_fsc_arithmetic_experiment as runner
from tools.jev import score_fsc_context_pair as scoring


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=runner.sha(path))


@pytest.fixture
def spec(tmp_path):
    folder=tmp_path/'inputs'
    artifacts={n:write(folder/(n+'.json'),{}) for n in ('population','task','input_receipts','requests')}
    artifacts['input_package_manifest']=write(folder/'manifest.json',dict(recordings=3793,
        task_sha256=artifacts['task']['sha256'],population_sha256=artifacts['population']['sha256'],
        files={'requests.jsonl':artifacts['requests']['sha256']}))
    artifacts['input_audit']=write(folder/'audit.json',dict(
        manifest=dict(request_manifest_sha256=artifacts['input_package_manifest']['sha256']),
        independent_receipts_sha256=artifacts['input_receipts']['sha256']))
    code=tmp_path/'code.py';code.write_text('frozen')
    return dict(artifacts=artifacts,pins={str(code):runner.sha(code)},
        source_experiment=str(tmp_path/'source-run'),remote_root=runner.PREFIX+'fsc-context-pair-fixture')


def test_plan_never_dispatches(spec,tmp_path,monkeypatch):
    def reject(*a):raise AssertionError('must not dispatch')
    monkeypatch.setattr(runner,'invoke',reject)
    out=tmp_path/'output'
    assert runner.run(spec,out)['status']=='validated_label_free_plan_no_dispatch'
    assert not out.exists()


@pytest.mark.parametrize('fault',['code','task','population','request','remote'])
def test_plan_rejects_drift(spec,fault):
    if fault=='remote':spec['remote_root']='/unapproved'
    elif fault=='code':Path(next(iter(spec['pins']))).write_text('changed')
    else:
        key='requests' if fault=='request' else fault
        Path(spec['artifacts'][key]['path']).write_text('changed')
    with pytest.raises(ValueError):runner.validate(spec)


@pytest.mark.parametrize('failure',[None,'source','package','inference','collect','score'])
def test_host_order_and_failure_retains_contract(spec,tmp_path,monkeypatch,failure):
    output=tmp_path/'output';calls=[]
    def invoke(module,args,log):
        phase=Path(log).stem;calls.append(phase)
        assert (output/'scoring-contract.json').is_file()
        assert runner.read(output/'scoring-contract.json')==runner.validate(spec)
        if phase==failure:raise RuntimeError('actual predecessor failure')
        if phase=='source':write(output/'source-package/PACKAGE.json',{})
        elif phase=='package':write(output/'runtime-package/PACKAGE.json',{})
        elif phase=='score':write(output/'scores.json',dict(runtime_provenance_verified=True))
    monkeypatch.setattr(runner,'invoke',invoke)
    if failure:
        with pytest.raises(RuntimeError):runner.run(spec,output,launch=True)
        state=runner.read(output/'status.json')
        assert state['phase']=='stopped_preserve_handles' and state['restart'] is False
    else:assert runner.run(spec,output,launch=True)['phase']=='frozen_transfer_collected_and_scored'
    order=['source','package','inference','collect','score']
    assert calls==order[:len(calls)]
    assert len(calls)==(order.index(failure)+1 if failure else 5)


@pytest.fixture
def score_spec(tmp_path,monkeypatch,spec):
    package=tmp_path/'package'
    a=spec['artifacts']
    for key,name in [('input_package_manifest','MANIFEST.json'),('input_audit','audit.json'),('input_receipts','receipts.jsonl')]:
        p=package/'inputs'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(a[key]['path']).read_bytes())
    write(package/'frozen-source/PACKAGE.json',dict(policies={r:dict(policy_id=r) for r in ('initial','selected')}))
    write(package/'frozen-source/source-selection.json',dict(complete=True))
    contract=write(tmp_path/'contract.json',runner.validate(spec))
    observed=write(tmp_path/'inspect.json',dict(State='success'))
    proof=dict(status='fsc_pair_inference_independently_collected',source_manifest_sha256='source',
        predictions={r:write(tmp_path/(r+'.json'),dict(records=[])) for r in ('initial','selected')})
    report=write(tmp_path/'report.json',dict(proof,runtime_inspect=observed))
    args=dict(package=str(package),digest='runtime',collection=str(tmp_path/'collection'),
        launch=write(tmp_path/'launch.json',{})['path'],observed=observed['path'],
        transport=write(tmp_path/'transport.json',{})['path'],inference_report=report['path'],
        scoring_contract=contract)
    calls=[]
    def audit(**kw):calls.append('audit');return copy.deepcopy(proof)
    def score(path):
        calls.append('target-labels');m=scoring.read(path)
        assert m['policies']['selected']['policy_id']=='selected'
        return dict(comparison={},runtime_provenance_verified=False)
    monkeypatch.setattr(scoring,'audit',audit);monkeypatch.setattr(scoring,'score',score)
    return args,calls,proof


def test_scoring_opens_target_only_after_actual_audit(score_spec,tmp_path):
    args,calls,_=score_spec
    result=scoring.run(args,tmp_path/'scores.json')
    assert calls==['audit','target-labels'] and result['runtime_provenance_verified']


@pytest.mark.parametrize('fault',['contract','runtime','proof','definitions','inspect'])
def test_invalid_scoring_identity_never_reads_target(score_spec,tmp_path,monkeypatch,fault):
    args,calls,proof=score_spec
    if fault=='contract':Path(args['scoring_contract']['path']).write_text('{}')
    elif fault=='runtime':
        def reject(**kw):raise ValueError('runtime audit failed')
        monkeypatch.setattr(scoring,'audit',reject)
    elif fault=='proof':proof['status']='changed'
    elif fault=='definitions':
        p=Path(args['package'])/'inputs/MANIFEST.json';v=scoring.read(p);v['task_sha256']='changed';write(p,v)
    else:write(Path(args['observed']),dict(State='changed'))
    with pytest.raises(ValueError):scoring.run(args,tmp_path/'scores.json')
    assert 'target-labels' not in calls


def test_source_subprocess_uses_repaired_audit(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda args,**kw:calls.append(args))
    runner.invoke('prepare_fsc_context_source_v2',[],tmp_path/'source.log')
    runner.invoke('launch_fsc_context_pair',[],tmp_path/'infer.log')
    assert calls[0][2]=='tools.reports.prepare_fsc_arithmetic_source'
    assert calls[1][2]=='tools.jev.launch_fsc_context_pair'
