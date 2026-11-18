"""Package the completed Stage1 policy as shared Stage2 source/reference.

Reuses the full endpoint/selection binding; no remote calls or RL admission.
Does not copy Stage1 optimizer state: task-stage optimizers start afresh.
"""
import argparse
import json
from pathlib import Path
import shutil

from tools.jev.context_stage2_source import bind
from tools.jev.context_rl_evaluation_plan import bound, sha


def authenticate(policy, binding):
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    return read_execution_bundle(policy, expected_policy_id=binding['source_policy_id'],
        expected_contract=binding['source_contract'],
        expected_base_id=binding['source_contract']['base_id'])


def derive_adapter(policy, adapter, binding):
    from multimodal_jev.context_policy_artifacts import export_native_adapter, verify_native_adapter
    bundle = authenticate(policy, binding)
    export_native_adapter(adapter, bundle)
    verify_native_adapter(adapter, bundle)


def verify(package, digest):
    root = Path(package)
    if root.is_symlink() or sha(root / 'PACKAGE.json') != digest:
        raise ValueError('source package identity differs')
    manifest = json.loads((root / 'PACKAGE.json').read_text())
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('source package symlinks forbidden')
        if path.is_file() and path != root / 'PACKAGE.json':
            actual.add(str(path.relative_to(root)))
    if actual != set(manifest['files']):raise ValueError('source package population differs')
    for name in manifest['files']:bound(root, manifest, name)
    binding = json.loads((root / 'binding.json').read_text())
    if (manifest['status'] != 'prepared_source_only_no_stage2_admission'
            or manifest['training_dispatch_allowed'] is not False
            or binding['source_step'] != 634
            or binding['source_policy_id'] != binding['reference_policy_id']
            or manifest['source_policy_id'] != binding['source_policy_id']
            or manifest['algorithms'] != ['grpo_v1', 'rloo_v1']
            or binding['reset_optimizer_for_task_stage'] is not True):
        raise ValueError('common full Stage1 source/reference required')
    from multimodal_jev.context_policy_artifacts import verify_native_adapter
    verify_native_adapter(root / 'adapter', authenticate(root / 'policy', binding))
    return manifest


def prepare(*, specification, output):
    specification, output = Path(specification), Path(output)
    spec = json.loads(specification.read_text())
    inputs = {str(specification.resolve()): sha(specification)}
    args = dict(spec)
    for name in ('experiment', 'training_proof', 'selection_proof'):
        path = Path(spec[name]);inputs[str(path.resolve())] = sha(path)
        args[name] = json.loads(path.read_text())
    # Re-execute binding, including actual policy/adapter tensor authentication.
    # A previously written 'passed' or binding JSON is not an admission token.
    binding = bind(**args)
    source = Path(binding['source_policy'])
    for root in (Path(args['training_root']), Path(args['selection_root'])):
        if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
            raise ValueError('source and output must be independent')
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('source policy symlinks forbidden')
    authenticate(source, binding)
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'stage2_source_package': output})
    output.mkdir(exist_ok=False)
    shutil.copytree(source, output / 'policy')
    # Re-derive provenance against the actual Stage1 policy identity. The
    # selection export may have another policy ID despite identical tensors.
    derive_adapter(output / 'policy', output / 'adapter', binding)
    (output / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    (output / 'evidence').mkdir()
    evidence = {}
    for i, (path, digest) in enumerate(sorted(inputs.items())):
        if sha(path) != digest:raise ValueError('source binding input changed')
        relative = f'evidence/{i:03d}.json'
        shutil.copyfile(path, output / relative)
        if sha(output / relative) != digest:raise ValueError('copied evidence differs')
        evidence[path] = dict(relative=relative, sha256=digest)
    manifest = dict(status='prepared_source_only_no_stage2_admission',
        training_dispatch_allowed=False, source_policy_id=binding['source_policy_id'],
        algorithms=['grpo_v1', 'rloo_v1'], policy='policy', adapter='adapter',
        reset_optimizer_for_task_stage=True, source_inputs=evidence,
        files={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()})
    (output / 'PACKAGE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return verify(output, sha(output / 'PACKAGE.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = prepare(specification=a.spec, output=a.output)
    print(json.dumps(dict(status=result['status'], source_policy_id=result['source_policy_id'],
                         manifest_sha256=sha(a.output / 'PACKAGE.json')), indent=2))
