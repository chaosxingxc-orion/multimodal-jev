"""KV-only matched100 candidate using the existing acoustic service lifecycle."""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_acoustic_training_operations as acoustic
from tools.reports import context_macro_training_operations as shared
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.run_context_batch_probe_candidate import verify

command = acoustic.command
audit_outputs = acoustic.audit_outputs
MUTABLE = {'FILES.json', 'training.json', 'service-0.json', 'service-1.json'}


def read(path):
    return json.loads(Path(path).read_text())


def check_delta(original, candidate, *, service, kv_bytes):
    """Compare relocated configs without weakening identities or external pins."""
    expected = json.loads(json.dumps(original))
    if service:
        if expected['policy']['kv_cache_memory_bytes'] != 2**30:
            raise ValueError('KV1GiB acoustic parent required')
        expected['policy']['kv_cache_memory_bytes'] = kv_bytes
    if candidate != expected:
        raise ValueError('only service KV capacity may change')


def validate(c):
    parent = c['parent_operations']
    acoustic.validate(parent)
    receipt = shared.validate(c, command_builder=command)
    source = Path(parent['prepared'])/'package'
    package = Path(c['prepared'])/'package'
    old, new = parent['remote_root'], c['remote_root']
    if Path(new).is_relative_to(old) or Path(old).is_relative_to(new):
        raise ValueError('independent KV training output required')
    if type(c['kv_cache_memory_bytes']) is not int or c['kv_cache_memory_bytes'] != 16 * 2**30:
        raise ValueError('measured KV16GiB candidate required')
    if receipt.get('kv_parent_manifest_sha256') != parent['manifest_sha256']:
        raise ValueError('exact acoustic parent manifest required')
    source_files = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    candidate_files = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if source_files != candidate_files:
        raise ValueError('inherited package file set changed')
    for rel in source_files - MUTABLE:
        if file_hash(source/rel) != file_hash(package/rel):
            raise ValueError('inherited runtime or asset changed: '+rel)
    for name in ('training.json', 'service-0.json', 'service-1.json'):
        check_delta(relocate(read(source/name), old, new), read(package/name),
                    service=name.startswith('service-'), kv_bytes=c['kv_cache_memory_bytes'])
    t = read(package/'training.json')
    if (t['start_update'], t['stop_update'], t['microbatch'], t['global_batch']) != (0, 100, 8, 128):
        raise ValueError('fixed initial100 micro8 global128 required')
    return receipt


def prepare(c, provenance):
    parent = read(c['parent_operations'])
    receipt = acoustic.validate(parent)
    source = Path(parent['prepared'])/'package'
    dest = Path(c['output'])
    check_paths({'kv_training_candidate': dest})
    if type(c['kv_cache_memory_bytes']) is not int or c['kv_cache_memory_bytes'] != 16 * 2**30:
        raise ValueError('measured KV16GiB candidate required')
    old, new = parent['remote_root'], c['remote_root']
    if Path(new).is_relative_to(old) or Path(old).is_relative_to(new):
        raise ValueError('independent KV training output required')
    if dest.resolve().is_relative_to(Path(parent['prepared']).resolve()):
        raise ValueError('candidate must not be inside parent package')
    dest.mkdir(parents=True, exist_ok=False)
    package = dest/'package'
    shutil.copytree(source, package)
    (package/'FILES.json').unlink()
    for name in ('training.json', 'service-0.json', 'service-1.json'):
        value = relocate(read(source/name), old, new)
        if name.startswith('service-'):
            value['policy']['kv_cache_memory_bytes'] = c['kv_cache_memory_bytes']
        (package/name).write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
    write_new(package/'FILES.json', {str(p.relative_to(package)): file_hash(p)
                                    for p in package.rglob('*') if p.is_file()})
    result = relocate(receipt, old, new)
    result.update(files_sha256=verify(package), kv_parent_manifest_sha256=parent['manifest_sha256'],
                  config_provenance=provenance)
    write_new(dest/'PREPARED.json', result)
    ops = dict(parent, parent_operations=parent, prepared=str(dest), remote_root=new,
               tag=c['tag'], operations_root=c['operations_root'],
               manifest_sha256=result['files_sha256'], kv_cache_memory_bytes=c['kv_cache_memory_bytes'],
               finite_training_dispatch_allowed=False)
    validate(ops)
    write_new(dest/'operations.json', ops)
    return ops


def run(c, provenance):
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('explicit finite dispatch required')
    return shared.run_services(dict(c, finite_inference_dispatch_allowed=True), provenance,
                               validator=validate, command_builder=command, required_devices=(4,5,6,7))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['prepare', 'validate', 'run', 'collect'], required=True)
    a = p.parse_args(); c, provenance = load_config(a.config, a.local)
    if a.action == 'prepare': result = prepare(c, provenance)
    elif a.action == 'validate': result = validate(c)
    elif a.action == 'run': result = run(c, provenance)
    else:
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result = collect_and_audit(c, validator=validate, command_builder=command,
                                  output_auditor=audit_outputs,
                                  result_dirs=('run','journal','service-0','service-1'))
    print(json.dumps(result, indent=2))
