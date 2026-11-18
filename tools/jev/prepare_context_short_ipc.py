"""Version a shared-container package with short, local per-role IPC paths.

Only the supervisor runner's cache environment changes. Training sources,
numerics and the two-update admission guard remain unchanged.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(runner):
    old = " cache=r/(role+'-cache')\n cache.mkdir(exist_ok=True)\n"
    new = (" cache=Path('/jev-tmp')/role\n"
           " cache.mkdir(exist_ok=True)\n"
           " env['VLLM_RPC_BASE_PATH']=str(cache)\n")
    if runner.count(old) != 1 or 'VLLM_RPC_BASE_PATH' in runner:
        raise ValueError('expected original per-role cache environment')
    result = runner.replace(old, new)
    ast.parse(result)
    return result


def prepare(*, source, source_manifest_sha256, output, remote_root):
    source, output = Path(source), Path(output)
    if sha(source/'PREPARED.json') != source_manifest_sha256:
        raise ValueError('source manifest changed')
    manifest = json.loads((source/'PREPARED.json').read_text())
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('source symlink forbidden')
    actual = {str(p.relative_to(source)): sha(p) for p in source.rglob('*')
              if p.is_file() and p != source/'PREPARED.json'}
    if actual != manifest['files']:
        raise ValueError('source package population or hashes changed')
    old_root = manifest['remote_root']
    parent = Path('/data/xingchao/multimodal-jev/simple-final-v1')
    destination = Path(remote_root)
    if (destination.parent != parent or '..' in destination.parts
            or str(destination) != remote_root or remote_root == old_root):
        raise ValueError('distinct direct local-storage root required')
    if (output.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(output.resolve())):
        raise ValueError('independent output required')
    runner = transform((source/'run.py').read_text())

    def relocate(value):
        if isinstance(value, str):
            return remote_root+value[len(old_root):] if value == old_root or value.startswith(old_root+'/') else value
        if isinstance(value, list):
            return [relocate(x) for x in value]
        if isinstance(value, dict):
            return {relocate(k): relocate(v) for k, v in value.items()}
        return value

    configs = {name: relocate(json.loads((source/name).read_text()))
               for name in ('config.json', 'service-config.json')}
    c = configs['config.json']
    if (c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False
            or c['start_update'] != 0 or c['stop_update'] != 2):
        raise ValueError('original bounded two-update candidate required')
    shutil.copytree(source, output)
    (output/'run.py').write_text(runner)
    for name, value in configs.items():
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    result = dict(status='prepared_only_no_dispatch', remote_root=remote_root,
        source_package=str(source), source_manifest_sha256=source_manifest_sha256,
        changes=['runner_role_cache_to_existing_jev_tmp_bind', 'explicit_short_VLLM_RPC_BASE_PATH',
                 'relocate_configuration_paths'],
        unchanged=['all_model_and_training_sources', 'loss', 'precision', 'schedule',
                   'source_policy', 'bounded_two_update_guard'],
        required_mount='remote_root/cache:/jev-tmp',
        files={str(p.relative_to(output)): sha(p) for p in output.rglob('*')
               if p.is_file() and p != output/'PREPARED.json'})
    (output/'PREPARED.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source', 'source-manifest-sha256', 'output', 'remote-root'):
        parser.add_argument('--'+key, required=True)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], files=len(result['files']),
                         remote_root=result['remote_root']), indent=2))
