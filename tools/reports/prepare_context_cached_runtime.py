"""Seal shared reference-build/GRPO sources before producing reference outputs.

This is a code-only preparation step. It grants neither reference-build nor
training dispatch and does not authenticate the inherited policy as Stage1.
Later policy/config binding must preserve this exact Python source manifest.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_context_replica_training_candidate import sha, rebase, IMAGE
from tools.reports.context_cached_reference_trainer import generate

ISOLATED_CHECK = '''import json,sys
from pathlib import Path
import tools.reports.context_cached_reference_trainer as trainer
import tools.reports.build_context_reference_dataset as builder
import tools.reports.audit_context_reference_dataset as auditor
import tools.reports.context_train_macro_parity as parity
import tools.reports.context_cached_tail_trainer as tail
root=Path(sys.argv[1]).resolve()
for module in (trainer,builder,auditor,parity,tail):
 if not Path(module.__file__).resolve().is_relative_to(root):
  raise ValueError('runtime imported from outside sealed package: '+module.__file__)
for formal in (False,True):
 compile(trainer.generate(root,formal=formal),'<isolated-cached-trainer>','exec')
compile(tail.generate(root),'<isolated-cached-tail>','exec')
from multimodal_jev.context_topology_batch_plan import derive
batch=dict(training_devices=4,tensor_parallel=1,pipeline_parallel=1,
 per_dp_replica_batch=64,microbatch=8,inference_instances=0,
 inference_batch_queries=32,source_queries=40513,epochs=2,efficiency_steps=100)
assert derive(batch)['total_updates']==318
bundle=dict(policy_id='endpoint-fixture',contract={'fixture':'contract'})
config=dict(stage1_endpoint=318,expected_population=40513,release_sha256='fixture-release',base_id='fixture-base')
checkpoint=dict(updates=318,joint_policy={'policy_id':bundle['policy_id']},contract=dict(
 batch_config=batch,stage='instruction',world_size=4,source_contract=bundle['contract'],
 objective={'kind':'candidate_infonce','probability_coefficient':0.},
 release_sha256=config['release_sha256'],base_id=config['base_id']))
builder.validate_endpoint(checkpoint,config,bundle)
print(json.dumps({'isolated_imports':True,'finite_and_formal_generation':True,'tail_generation':True,
 'stage1_zero_inference_endpoint_contract':True}))
'''

ROOTS = (
    'tools.reports.context_cached_reference_trainer',
    'tools.reports.context_cached_tail_trainer',
    'tools.reports.context_macro_trainer_candidate',
    'tools.reports.context_replica_trainer_candidate',
    'tools.reports.context_replica_formal_candidate',
    'tools.reports.context_cached_reference_session',
    'tools.reports.build_context_reference_dataset',
    'tools.reports.audit_context_reference_dataset',
    'tools.reports.context_train_macro_parity',
    'tools.reports.context_train_parity_inputs',
    'multimodal_jev.readonly_encoded_audio_cache',
    'multimodal_jev.context_stage_lora',
    'multimodal_jev.context_topology_batch_plan',
    'tools.reports.context_replica_runtime',
    'tools.reports.context_bounded_rollout_candidate',
    'tools.reports.context_bounded_rollout_worker',
    'tools.reports.context_bounded_attention_worker',
    'multimodal_jev.bounded_prefill_attention',
)


def module_path(module):
    if module.startswith('multimodal_jev.'):
        return Path('src') / (module.replace('.', '/') + '.py')
    if module.startswith(('tools.reports.', 'tools.jev.')):
        return Path(module.replace('.', '/') + '.py')
    return None


def imports(source, module):
    """Internal static imports, including relative and from-package imports."""
    result = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                parent = module.split('.')[:-node.level]
                prefix = '.'.join(parent + ([prefix] if prefix else []))
            if prefix in ('multimodal_jev', 'tools.reports', 'tools.jev'):
                result.update(prefix + '.' + a.name for a in node.names)
            else:
                result.add(prefix)
    return {name for name in result if module_path(name) is not None}


def close_sources(project, assembled):
    """Overlay declared new entries; preserve inherited dependencies bytewise.

    Missing imports are recursively copied, including imports in the generated
    trainer. Frozen inherited modules still undergo the generator's hash checks.
    """
    project, assembled = Path(project), Path(assembled)
    pending = list(ROOTS); seen = set(); copied = []
    for formal in (False, True):
        pending.extend(imports(generate(project, formal=formal),
                               'tools.reports.context_cached_reference_trainer'))
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        relative = module_path(module); dest = assembled / relative
        if module in ROOTS or not dest.is_file():
            origin = project / relative
            if not origin.is_file():
                # Dynamically synthesized private modules have no on-disk source.
                if module.startswith('multimodal_jev._'):
                    continue
                raise ValueError('runtime import has no source: ' + module)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, dest); copied.append(str(relative))
        pending.extend(imports(dest.read_text(), module))
    for formal in (False, True):
        compile(generate(assembled, formal=formal), '<sealed-cached-trainer>', 'exec')
    return sorted(copied)


def prepare(c, provenance):
    parent, output, project = map(Path, (c['parent'], c['output'], c['project']))
    manifest = parent / 'PREPARED.json'
    if parent.is_symlink() or sha(manifest) != c['parent_sha256']:
        raise ValueError('pinned regular parent required')
    receipt = json.loads(manifest.read_text())
    files = {str(p.relative_to(parent)): sha(p) for p in parent.rglob('*')
             if p.is_file() and p != manifest}
    if (any(p.is_symlink() for p in parent.rglob('*')) or files != receipt['files']
            or receipt['image'] != IMAGE
            or receipt['status'] not in ('replica_integrated_prepared_only', 'replica_formal_prepared_only')):
        raise ValueError('immutable replica runtime template required')
    old, remote = receipt['remote_root'], c['remote_root']
    from tools.reports.context_replica_training_operations import path
    path(remote)
    if (output.exists() or output.resolve().is_relative_to(parent.resolve())
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('fresh disjoint runtime paths required')
    # Validate source compatibility before creating any candidate files.
    generate(project, formal=False); generate(project, formal=True)
    check_paths({'cached_runtime_package': output})
    shutil.copytree(parent, output)
    (output / 'PREPARED.json').unlink()
    copied = close_sources(project, output / 'source')
    runtime = {str(p.relative_to(output / 'source')): sha(p)
               for p in (output / 'source').rglob('*.py') if p.is_file()}
    (output / 'RUNTIME-SOURCES.json').write_text(json.dumps(runtime, sort_keys=True, indent=2)+'\n')
    config = rebase(json.loads((parent / 'config.json').read_text()), old, remote)
    identity = dict(config['dual_runtime_identity'],
                    source_manifest_sha256=sha(output / 'RUNTIME-SOURCES.json'))
    config.update(dual_runtime_identity=identity, training_dispatch_allowed=False,
                  runtime_preparation_only=True)
    config['macro_client']['runtime_identity'] = identity
    config.pop('reference_cache', None)
    config.pop('reference_dataset', None)
    # An inherited run.py must never dispatch the historical trainer.
    (output / 'run.py').write_text(
        "raise SystemExit('code-only runtime: bind audited Stage1/reference and numerical admission before dispatch')\n")
    names = ['config.json', *receipt['layout']['services'].values()]
    external = {p:h for p,h in config['pins'].items() if not p.startswith(remote+'/')}
    pins = dict(external, **{remote+'/'+str(p.relative_to(output)):sha(p)
        for p in output.rglob('*') if p.is_file() and str(p.relative_to(output)) not in names})
    for name in names:
        value = config if name == 'config.json' else rebase(
            json.loads((parent / name).read_text()), old, remote)
        value.update(dual_runtime_identity=identity, pins=pins)
        (output / name).write_text(json.dumps(value, indent=2)+'\n')
    result = dict(receipt, status='cached_reference_runtime_prepared_only',
        remote_root=remote, dual_runtime_identity=identity, config_provenance=provenance,
        parent_sha256=c['parent_sha256'], training_dispatch_allowed=False,
        finite_npu_dispatch_allowed=False, copied_sources=copied,
        policy_provenance_admitted=False, numerical_admission=False,
        pending=['bind actual InfoNCE endpoint and native adapter', 'batch32 numerical admission',
                 'build and independently audit full TRAIN reference',
                 'bind readonly dataset without changing runtime sources', 'DP2+2 NPU acceptance'])
    result['files'] = {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    (output / 'PREPARED.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def audit(c, provenance):
    """Verify sealed bytes and import from a clean process outside the repo."""
    package = Path(c['package']).resolve()
    if package.is_symlink() or sha(package/'PREPARED.json') != c['manifest_sha256']:
        raise ValueError('cached runtime manifest differs')
    receipt = json.loads((package/'PREPARED.json').read_text())
    actual = {str(p.relative_to(package)):sha(p) for p in package.rglob('*')
              if p.is_file() and p != package/'PREPARED.json'}
    if (any(p.is_symlink() for p in package.rglob('*')) or actual != receipt['files']
            or receipt['status'] != 'cached_reference_runtime_prepared_only'):
        raise ValueError('immutable cached runtime required')
    runtime = json.loads((package/'RUNTIME-SOURCES.json').read_text())
    if (runtime != {k.removeprefix('source/'):v for k,v in actual.items()
                    if k.startswith('source/') and k.endswith('.py')}
            or sha(package/'RUNTIME-SOURCES.json') != receipt['dual_runtime_identity']['source_manifest_sha256']):
        raise ValueError('complete runtime identity required')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join(str(package/'source'/p) for p in ('','src','tools/jev')))
    result = subprocess.run([sys.executable,'-c',ISOLATED_CHECK,str(package/'source')],
        cwd=package.parent, env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError('isolated runtime import/generation failed: '+result.stderr)
    checks = json.loads(result.stdout)
    return dict(status='cached_runtime_cpu_artifacts_audited',manifest_sha256=c['manifest_sha256'],
        runtime_identity=receipt['dual_runtime_identity'],checks=checks,files=len(actual),
        npu_executed=False,training_dispatch_allowed=False,provenance=provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    parser.add_argument('--action',choices=['prepare','audit'],default='prepare')
    args = parser.parse_args(); config, provenance = load_config(args.config, args.local)
    if args.action=='prepare':
        receipt=prepare(config, provenance)
        result=dict(status=receipt['status'],runtime_identity=receipt['dual_runtime_identity'])
    else:
        result=audit(config,provenance)
        from multimodal_jev.local_checkpoint import commit_json
        commit_json(config['audit_output'],result)
    print(json.dumps(result, indent=2))
