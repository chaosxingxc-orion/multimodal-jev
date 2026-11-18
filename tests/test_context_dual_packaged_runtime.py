"""Exercise the immutable prepared payload, not workspace import fallbacks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope='module')
def package():
    candidates=[]
    for p in Path.cwd().rglob('PREPARED.json'):
        record=json.loads(p.read_text())
        if record.get('status')=='integrated_prepared_only':
            candidates.append(p.parent)
    assert len(candidates)==1, 'one explicitly supplied integrated payload required'
    return candidates[0]


def test_actual_package_manifest_and_config(package):
    receipt=json.loads((package/'PREPARED.json').read_text())
    files={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in package.rglob('*') if p.is_file() and p!=package/'PREPARED.json'}
    assert files==receipt['files']
    config=json.loads((package/'config.json').read_text())
    service=json.loads((package/'service-config.json').read_text())
    assert config['dual_runtime_identity']==service['dual_runtime_identity']==receipt['dual_runtime_identity']
    assert config['training_dispatch_allowed'] is False
    assert config['engineering_prefix'] is True
    assert config['stop_update']-config['start_update']==2
    assert receipt['finite_npu_dispatch_allowed'] is False
    manifest=package/'RUNTIME-SOURCES.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==receipt['dual_runtime_identity']['source_manifest_sha256']
    runtime=json.loads(manifest.read_text())
    assert runtime=={k.removeprefix('source/'):v for k,v in files.items()
                     if k.startswith('source/') and k.endswith('.py')}
    for name,digest in config['pins'].items():
        prefix=receipt['remote_root']+'/'
        if name.startswith(prefix):
            assert files[name.removeprefix(prefix)]==digest


def test_isolated_packaged_trainer_actor_and_worker_imports(package):
    source=(package/'source').resolve()
    # -I excludes both cwd and PYTHONPATH. Only this exact payload is inserted.
    code=r'''
import ast, importlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path[:0]=[str(root),str(root/'src'),str(root/'tools/jev')]
from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3
initialize_verl_preserving_qwen3()
from tools.reports.context_dual_trainer_candidate import generate
text=generate(root)
assert 'validate_trainable_scope(named)' in text
namespace={'__name__':'packaged_trainer_check'}
exec(compile(text,'<packaged-trainer>','exec'),namespace)
assert callable(namespace['run'])
# Resolve imports inside the trainer as well, without executing its NPU loop.
for node in ast.walk(ast.parse(text)):
    if isinstance(node,ast.ImportFrom) and node.module and not node.level:
        module=importlib.import_module(node.module)
        for alias in node.names:
            assert hasattr(module,alias.name),(node.module,alias.name)
from tools.reports.context_dual_actor_candidate import load,MODE
assert load(root).MODE==MODE
from tools.reports.serve_context_dual_candidate import service_namespace
assert callable(service_namespace()['run'])
from tools.reports.context_dual_rollout_candidate import owner_class
assert isinstance(owner_class(root),type)
assert isinstance(owner_class(root,shared_scope=True),type)
from tools.reports.context_dual_worker_candidate import DualContextWorkerCandidate
assert callable(DualContextWorkerCandidate.install_jev_dual_candidate)
from tools.reports.context_dual_runtime_candidate import BoundDualClient
from tools.reports.context_dual_session_candidate import DualContextRolloutSession
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
import torch
assert not validate_trainable_scope([('decision_head.projection.weight',torch.nn.Parameter(torch.ones(1)))])['base_model_saved']
for name,module in tuple(sys.modules.items()):
    if name.startswith(('multimodal_jev.','tools.reports.')) and getattr(module,'__file__',None):
        assert Path(module.__file__).resolve().is_relative_to(root),(name,module.__file__)
print(json.dumps({'packaged_imports':True,'npu_execution':False}))
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(source)],
                          cwd=source,env=dict(os.environ),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert '"packaged_imports": true' in result.stdout
