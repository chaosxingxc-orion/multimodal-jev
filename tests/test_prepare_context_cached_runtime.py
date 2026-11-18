import json
from pathlib import Path

import pytest

from tools.reports.prepare_context_cached_runtime import prepare, imports, sha


def setup(tmp_path, monkeypatch):
    from test_prepare_context_replica_training_candidate import fixture
    from tools.reports.prepare_context_replica_training_candidate import prepare as parent_prepare
    import tools.reports.prepare_context_cached_runtime as module
    # Only the inherited legacy trainer transform uses fixture stand-ins.
    # Dependency discovery/copying, pins and manifest construction below are real.
    values, parent = fixture(tmp_path)
    parent_prepare(values, {'fixture': True})
    monkeypatch.setattr(module, 'check_paths', lambda _: None)
    return dict(parent=str(parent), parent_sha256=sha(parent/'PREPARED.json'),
        project=str(Path.cwd()), output=str(tmp_path/'cached'), remote_root='/remote/cached')


def test_package_seals_common_sources_and_cannot_launch_old_trainer(tmp_path, monkeypatch):
    c=setup(tmp_path,monkeypatch)
    parent=Path(c['parent']); before={str(p.relative_to(parent)):sha(p)
        for p in parent.rglob('*') if p.is_file()}
    result=prepare(c, {'fixture':True}); out=Path(c['output'])
    assert result['status']=='cached_reference_runtime_prepared_only'
    assert result['training_dispatch_allowed'] is False
    assert result['policy_provenance_admitted'] is False
    assert before=={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file()}
    runtime=json.loads((out/'RUNTIME-SOURCES.json').read_text())
    for name in ('tools/reports/context_cached_reference_trainer.py',
                 'tools/reports/context_cached_tail_trainer.py',
                 'src/multimodal_jev/context_cached_tail_window.py',
                 'tools/reports/context_cached_reference_session.py',
                 'tools/reports/build_context_reference_dataset.py',
                 'tools/reports/audit_context_reference_dataset.py',
                 'tools/reports/context_train_macro_parity.py',
                 'tools/reports/context_train_parity_inputs.py',
                 'src/multimodal_jev/readonly_encoded_audio_cache.py',
                 'src/multimodal_jev/context_reference_dataset.py'):
        assert runtime[name]==sha(out/'source'/name)
    assert runtime=={str(p.relative_to(out/'source')):sha(p)
        for p in (out/'source').rglob('*.py')}
    config=json.loads((out/'config.json').read_text())
    assert 'reference_cache' not in config and 'reference_dataset' not in config
    assert config['runtime_preparation_only'] is True
    identity=result['dual_runtime_identity']
    assert identity['source_manifest_sha256']==sha(out/'RUNTIME-SOURCES.json')
    for name in ['config.json', *result['layout']['services'].values()]:
        value=json.loads((out/name).read_text())
        assert value['dual_runtime_identity']==identity
        for key,digest in value['pins'].items():
            if key.startswith(c['remote_root']+'/'):
                assert sha(out/Path(key).relative_to(c['remote_root']))==digest
    assert result['files']=={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
        if p.is_file() and p.name!='PREPARED.json'}
    with pytest.raises(SystemExit,match='code-only runtime'):
        exec(compile((out/'run.py').read_text(),'<test>','exec'),{})


@pytest.mark.parametrize('fault',['modified','symlink','overlap'])
def test_parent_or_destination_fault_rejected_before_copy(tmp_path,monkeypatch,fault):
    c=setup(tmp_path,monkeypatch)
    if fault=='modified':(Path(c['parent'])/'run.py').write_text('changed')
    elif fault=='symlink':(Path(c['parent'])/'extra').symlink_to('run.py')
    else:c['remote_root']='/remote/replica/nested'
    with pytest.raises(ValueError):prepare(c,{})
    assert not Path(c['output']).exists()


def test_relative_and_package_imports_are_included():
    assert imports('from .context_reference_dataset import ReferenceDataset\n'
        'from tools.reports import context_cached_reference_session\n'
        'import torch\n','multimodal_jev.consumer')=={
            'multimodal_jev.context_reference_dataset','tools.reports.context_cached_reference_session'}


def test_legacy_macro_generator_is_explicitly_replaced(tmp_path,monkeypatch):
    c=setup(tmp_path,monkeypatch);parent=Path(c['parent'])
    relative='source/tools/reports/context_macro_trainer_candidate.py'
    (parent/relative).write_text('def generate(root): return "legacy"\n')
    receipt=json.loads((parent/'PREPARED.json').read_text())
    receipt['files'][relative]=sha(parent/relative)
    (parent/'PREPARED.json').write_text(json.dumps(receipt))
    c['parent_sha256']=sha(parent/'PREPARED.json')
    result=prepare(c,{})
    assert 'tools/reports/context_macro_trainer_candidate.py' in result['copied_sources']
    assert sha(Path(c['output'])/relative)==sha(Path(c['project'])/'tools/reports/context_macro_trainer_candidate.py')


def test_sealed_stage1_endpoint_replaces_stale_positive_inference_requirement(tmp_path,monkeypatch):
    import os
    import subprocess
    import sys
    c=setup(tmp_path,monkeypatch);parent=Path(c['parent'])
    relative='source/src/multimodal_jev/context_topology_batch_plan.py'
    path=parent/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("def derive(c):\n if c['inference_instances']==0: raise ValueError('legacy requires inference')\n")
    receipt=json.loads((parent/'PREPARED.json').read_text());receipt['files'][relative]=sha(path)
    (parent/'PREPARED.json').write_text(json.dumps(receipt));c['parent_sha256']=sha(parent/'PREPARED.json')
    stale=sha(path);result=prepare(c,{})
    assert sha(path)==stale
    assert 'src/multimodal_jev/context_topology_batch_plan.py' in result['copied_sources']
    root=Path(c['output'])/'source'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(str(root/p) for p in ('','src','tools/jev')))
    # The parent fixture intentionally has stubbed unrelated legacy modules.
    # Execute the actual sealed endpoint validator in isolation with its real
    # topology import; the full production audit separately imports all entries.
    check="""import ast,json,sys
from pathlib import Path
root=Path(sys.argv[1])
tree=ast.parse((root/'tools/reports/build_context_reference_dataset.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_endpoint')
scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<sealed-endpoint>','exec'),scope)
batch=dict(training_devices=4,tensor_parallel=1,pipeline_parallel=1,per_dp_replica_batch=64,
 microbatch=8,inference_instances=0,inference_batch_queries=32,source_queries=40513,epochs=2,efficiency_steps=100)
bundle=dict(policy_id='endpoint',contract={})
c=dict(stage1_endpoint=318,release_sha256='release',base_id='base',expected_population=40513)
checkpoint=dict(updates=318,joint_policy={'policy_id':'endpoint'},contract=dict(batch_config=batch,
 stage='instruction',world_size=4,source_contract={},objective={'kind':'candidate_infonce','probability_coefficient':0.},
 release_sha256='release',base_id='base'))
scope['validate_endpoint'](checkpoint,c,bundle)
print(json.dumps({'stage1_zero_inference_endpoint_contract':True}))
"""
    child=subprocess.run([sys.executable,'-c',check,str(root)],cwd=tmp_path,
        env=env,capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr
    assert json.loads(child.stdout)['stage1_zero_inference_endpoint_contract'] is True
