import copy
import json
from pathlib import Path

import pytest

from tools.jev import prepare_context_stage2_source as module
from test_context_policy_artifacts import fixture as policy_fixture


@pytest.fixture
def source(tmp_path, monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage, 'check_paths', lambda paths: [])
    _, policy, identity, contract = policy_fixture(tmp_path, monkeypatch)
    binding = dict(source_step=634, source_policy=str(policy), source_policy_id=identity,
        reference_policy_id=identity, source_contract=contract,
        reset_optimizer_for_task_stage=True, evaluated_policy_id='different-export-id')
    # Source admission has separate full-collection tests. This suite exercises
    # real BF16 policy copying, native adapter serialization and authentication.
    monkeypatch.setattr(module, 'bind', lambda **args: copy.deepcopy(binding))
    spec = dict(training_root=str(tmp_path / 'training'), selection_root=str(tmp_path / 'selection'))
    for name in ('experiment', 'training_proof', 'selection_proof'):
        path = tmp_path / (name + '.json');path.write_text('{}');spec[name] = str(path)
    path = tmp_path / 'spec.json';path.write_text(json.dumps(spec))
    return path, binding, tmp_path / 'package'


def test_real_policy_and_adapter_shared_without_optimizer(source):
    spec, binding, output = source
    manifest = module.prepare(specification=spec, output=output)
    assert manifest['source_policy_id'] == binding['source_policy_id']
    assert manifest['training_dispatch_allowed'] is False
    assert manifest['algorithms'] == ['grpo_v1', 'rloo_v1']
    assert (output / 'policy/execution.pt').read_bytes() == (Path(binding['source_policy']) / 'execution.pt').read_bytes()
    adapter = json.loads((output / 'adapter/CONTEXT_POLICY.json').read_text())
    assert adapter['policy_id'] == binding['source_policy_id'] != binding['evaluated_policy_id']
    assert not list(output.rglob('checkpoint-*.pt'))
    assert module.verify(output, module.sha(output / 'PACKAGE.json')) == manifest


@pytest.mark.parametrize('fault', ['policy', 'adapter', 'extra', 'symlink', 'evidence', 'manifest'])
def test_corruption_rejected(source, fault):
    spec, _, output = source
    module.prepare(specification=spec, output=output)
    digest = module.sha(output / 'PACKAGE.json')
    paths = dict(policy='policy/execution.pt', adapter='adapter/adapter_model.safetensors',
                 extra='extra', evidence='evidence/000.json', manifest='PACKAGE.json')
    if fault == 'symlink':(output / 'link').symlink_to(output / 'binding.json')
    else:(output / paths[fault]).write_bytes(b'corrupt')
    with pytest.raises(ValueError):module.verify(output, digest)


def test_unfinished_stage1_rejected_before_output(source, monkeypatch):
    spec, _, output = source
    def reject(**args):raise ValueError('full clean Stage1 step634 required')
    monkeypatch.setattr(module, 'bind', reject)
    with pytest.raises(ValueError, match='step634'):module.prepare(specification=spec, output=output)
    assert not output.exists()


def test_existing_package_is_not_overwritten(source):
    spec, _, output = source
    module.prepare(specification=spec, output=output)
    before = module.sha(output / 'PACKAGE.json')
    with pytest.raises(FileExistsError):module.prepare(specification=spec, output=output)
    assert module.sha(output / 'PACKAGE.json') == before


def test_package_must_not_write_into_collection(source):
    spec, _, _ = source
    data = json.loads(spec.read_text())
    output = Path(data['training_root']) / 'package'
    with pytest.raises(ValueError, match='independent'):
        module.prepare(specification=spec, output=output)
    assert not output.exists()
