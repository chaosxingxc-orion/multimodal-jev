import copy
import json
from pathlib import Path
import shutil

import pytest

from tools.jev import prepare_fsc_context_source as module
from test_context_policy_artifacts import fixture as policy_fixture


@pytest.fixture
def source(tmp_path,monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    from multimodal_jev.context_policy_snapshot import export_policy
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,export_native_adapter
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    monkeypatch.setattr(storage,'check_paths',lambda paths:[])
    model,old,_,contract=policy_fixture(tmp_path,monkeypatch)
    contract['request_version']='jev-context-exclusive_event-v1'
    contract['token']['registration_status']='admitted'
    policy=tmp_path/'admitted';identity=export_policy(policy,model,old/'processor',contract)
    adapter=tmp_path/'adapter';export_native_adapter(adapter,read_execution_bundle(policy,
        expected_policy_id=identity,expected_contract=contract,expected_base_id=contract['base_id']))
    experiment=tmp_path/'experiment';experiment.mkdir()
    points=[];transports=[]
    for step in (0,64):
        root=tmp_path/f'point-{step}';(root/'evaluation/run').mkdir(parents=True)
        shutil.copytree(policy,root/'evaluation/run/policy');shutil.copytree(adapter,root/'evaluation/run/adapter')
        proof=dict(files={str(p.relative_to(root)):module.sha(p) for p in root.rglob('*') if p.is_file()})
        path=tmp_path/f'transport-{step}.json';path.write_text(json.dumps(proof))
        points.append(dict(step=step,collection=str(root),source_policy_id=f'source-{step}',inference_policy_id=identity))
        transports.append(dict(step=step,transport=str(path)))
    curve=dict(complete=True,algorithm='grpo_v1',initial_policy_id=identity,
        selected_checkpoint=dict(step=64,policy_id=identity),independently_audited_points=points)
    (experiment/'curve-spec.json').write_text(json.dumps(dict(evaluations=transports)))
    calls=[]
    def load(directory,algorithm):
        calls.append((directory,algorithm))
        return dict(curve=copy.deepcopy(curve),provenance={'fixture':'full source auditor replaced'})
    monkeypatch.setattr(module,'load_run',load)
    audit=tmp_path/'input-audit.json';audit.write_text(json.dumps(dict(
        state=dict(Status='exited',ExitCode=0,OOMKilled=False),
        manifest=dict(recordings=3793,decisions=31,processor_sha256=processor_cache_identity(policy)))))
    args=dict(experiment_directory=experiment,input_audit=audit,input_audit_sha256=module.sha(audit),output=tmp_path/'package')
    return args,curve,calls


def test_real_bf16_assets_and_native_adapter_preserve_selected_ids(source):
    args,curve,calls=source
    result=module.prepare(**args)
    assert calls==[(args['experiment_directory'],'grpo_v1')]
    assert result['policies']['selected']['step']==64 and result['threshold']==.5
    assert result['policies']['initial']['policy_id']==curve['initial_policy_id']
    assert result['policies']['selected']['policy_id']==curve['selected_checkpoint']['policy_id']
    assert result['target_tuning_allowed'] is False
    assert not list(args['output'].rglob('checkpoint-*.pt'))
    root=Path(curve['independently_audited_points'][0]['collection'])
    assert (args['output']/'initial/policy/execution.pt').read_bytes()==(root/'evaluation/run/policy/execution.pt').read_bytes()
    assert module.verify(args['output'],module.sha(args['output']/'PACKAGE.json'))==result


def test_selected_initial_checkpoint_is_valid_not_forced_improvement(source):
    args,curve,_=source;curve['selected_checkpoint']['step']=0
    result=module.prepare(**args)
    assert result['policies']['selected']['step']==0


@pytest.mark.parametrize('fault',['wrong_id','changed_weight','wrong_processor','unfinished_input','nested','source_audit'])
def test_invalid_source_stops_before_output(source,monkeypatch,fault):
    args,curve,_=source
    if fault=='wrong_id':curve['selected_checkpoint']['policy_id']='target-best'
    elif fault=='changed_weight':
        root=Path(curve['independently_audited_points'][0]['collection']);(root/'evaluation/run/policy/execution.pt').write_bytes(b'corrupt')
    elif fault in ('wrong_processor','unfinished_input'):
        p=args['input_audit'];v=json.loads(p.read_text())
        if fault=='wrong_processor':v['manifest']['processor_sha256']='changed'
        else:v['state']['Status']='running'
        p.write_text(json.dumps(v));args['input_audit_sha256']=module.sha(p)
    elif fault=='nested':args['output']=args['experiment_directory']/'package'
    else:
        def reject(*a):raise ValueError('full source curve required')
        monkeypatch.setattr(module,'load_run',reject)
    with pytest.raises(ValueError):module.prepare(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('fault',['policy','adapter','extra','audit','selection','symlink'])
def test_sealed_package_corruption_rejected(source,fault):
    args,_,_=source;module.prepare(**args);root=args['output'];digest=module.sha(root/'PACKAGE.json')
    paths=dict(policy='initial/policy/execution.pt',adapter='selected/adapter/adapter_model.safetensors',
        extra='extra',audit='input-audit.json',selection='source-selection.json')
    if fault=='symlink':(root/'link').symlink_to(root/'source-selection.json')
    else:(root/paths[fault]).write_bytes(b'changed')
    with pytest.raises(ValueError):module.verify(root,digest)
