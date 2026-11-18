"""Prepare one full GRPO/RLOO package from collected sources and NPU evidence.

No remote calls or launch. Original engineering packages and baseline lock
remain immutable. Publication requires full source binding and acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from tools.jev.context_formal_acceptance import verify as accept, IMAGE
from tools.jev.context_rl_evaluation_plan import bound, sha
from tools.jev.context_stage2_source import bind
from tools.jev.prepare_context_stage2_source import verify as verify_source
from tools.jev.prepare_context_formal_runtime import transform
from tools.jev.verify_context_baseline_lock import verify as verify_lock

PREFIX = '/data/xingchao/multimodal-jev/simple-final-v1/'
BASE_LOCK = 'configs/releases/context-prob-v1/LOCK.json'


def write(path, value):path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(package, digest):
    root = Path(package)
    if root.is_symlink() or sha(root / 'PREPARED.json') != digest:
        raise ValueError('formal package manifest differs')
    manifest = json.loads((root / 'PREPARED.json').read_text())
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('formal package symlink forbidden')
        if path.is_file() and path != root / 'PREPARED.json':actual[str(path.relative_to(root))] = sha(path)
    if actual != manifest['files']:raise ValueError('formal package population or bytes differ')
    c = json.loads((root / 'config.json').read_text())
    release = json.loads((root / 'RELEASE.json').read_text())
    service = json.loads((root / 'service-config.json').read_text())
    payload = {k:v for k,v in release.items() if k != 'release_id'}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (manifest['image'] != IMAGE or manifest['status'] != 'prepared_formal_package_no_dispatch'
            or manifest['algorithm'] != c['algorithm'] or manifest['source_policy_id'] != c['source_policy_id']
            or manifest['release_id'] != release['release_id'] or identity != release['release_id']
            or c['formal_release_sha256'] != sha(root / 'RELEASE.json')
            or release['config_contract'] != {k:v for k,v in c.items() if k not in ('pins','formal_release','formal_release_sha256')}):
        raise ValueError('formal package/release/config identity differs')
    remote = manifest['remote_root']
    if (c['formal_release'] != remote+'/RELEASE.json' or c['pins'].get(c['formal_release']) != sha(root/'RELEASE.json')
            or service['pins'] != c['pins'] or service['policy']['snapshot'] != c['source_policy']
            or service['policy']['policy_id'] != c['source_policy_id']
            or service['policy']['contract'] != c['source_contract']
            or service['policy']['adapter'] != c['initial_adapter']):
        raise ValueError('actor and service package bindings differ')
    for name, digest in release['files'].items():
        if sha(bound(root, release, name)) != digest or c['pins'].get(remote+'/'+name) != digest:
            raise ValueError('released source asset missing from runtime pins')
    return manifest


def runner(original):
    guard = """if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:
 raise ValueError('two-update engineering candidate only')
if c['start_update']!=0 or c['stop_update']!=2:
 raise ValueError('initial bounded candidate only; resume requires separate binding')
"""
    if original.count(guard) != 1:raise ValueError('known shared supervisor runner required')
    return original.replace(guard, 'from context_formal_runtime import validate\nvalidate(c)\n').replace(
        'train_context_online_candidate.py', 'train_context_formal.py')


def prepare(*, source_package, source_manifest_sha256, source_specification, acceptance_specification,
            algorithm, output, remote_root):
    if algorithm not in ('grpo_v1', 'rloo_v1'):
        raise ValueError('registered executable algorithm required')
    if not re.fullmatch(re.escape(PREFIX) + r'context-rl-(grpo|rloo)-[a-z0-9-]+', remote_root):
        raise ValueError('isolated formal hot-storage root required')
    baseline = verify_lock(BASE_LOCK)
    source_package, output = Path(source_package), Path(output)
    source_manifest = verify_source(source_package, source_manifest_sha256)
    spec = json.loads(Path(source_specification).read_text());source_args = dict(spec)
    for name in ('experiment','training_proof','selection_proof'):
        source_args[name] = json.loads(Path(spec[name]).read_text())
    if source_args['experiment'] != json.loads((Path(BASE_LOCK).parent / 'experiment.json').read_text()):
        raise ValueError('frozen experiment required')
    binding = bind(**source_args)
    if binding != json.loads((source_package / 'binding.json').read_text()):
        raise ValueError('packaged source differs from revalidated full Stage1')
    if source_manifest['source_policy_id'] != binding['source_policy_id']:
        raise ValueError('source package identity differs')
    acceptance_spec = json.loads(Path(acceptance_specification).read_text())
    evidence = accept(**acceptance_spec)
    candidate = Path(acceptance_spec['candidate_package'])
    manifest = json.loads((candidate / 'PREPARED.json').read_text())
    old_root = manifest['remote_root']
    if remote_root == old_root:raise ValueError('preserve engineering run')
    for root in (candidate, source_package, Path(spec['training_root']), Path(spec['selection_root'])):
        if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
            raise ValueError('new independent package required')
    formal = transform((candidate / 'source/tools/jev/train_context_online_candidate.py').read_text())
    run = runner((candidate / 'run.py').read_text())
    def relocate(value):
        if isinstance(value, str):return value.replace(old_root, remote_root)
        if isinstance(value, list):return [relocate(v) for v in value]
        if isinstance(value, dict):return {relocate(k):relocate(v) for k,v in value.items()}
        return value
    c = relocate(json.loads((candidate / 'config.json').read_text()))
    service = relocate(json.loads((candidate / 'service-config.json').read_text()))
    training = json.loads(bound(Path(spec['training_root']), source_args['training_proof'], 'config.json').read_text())
    if (c['release_sha256'] != source_args['experiment']['release_sha256']
            or c['source_contract'] != binding['source_contract']
            or c['base_id'] != training['base_id'] or c['schedule_sha256'] != training['schedule_sha256']):
        raise ValueError('accepted runtime differs from full Stage1 contract/data/schedule')
    c.update(start_update=0, stop_update=634, engineering_prefix=False, training_dispatch_allowed=True,
        checkpoint_interval=64, algorithm=algorithm, source_policy=remote_root+'/initial/policy',
        source_policy_id=binding['source_policy_id'], source_contract=binding['source_contract'],
        initial_adapter=remote_root+'/initial/adapter', reference_adapter=remote_root+'/initial/adapter',
        encoded_audio_cache=training['encoded_audio_cache'], audio_mirror_root=training['audio_mirror_root'])
    c.pop('diagnostic_scope', None)
    # Pin the same read-only model/data and completed learning cache publications.
    external = {p:h for p,h in training['pins'].items() if Path(p).is_relative_to(Path(c['model'])) or p == c['tasks']}
    for key in ('audio_mirror_root', 'encoded_audio_cache'):
        directory = Path(c[key]) if key == 'audio_mirror_root' else Path(c[key]).parent
        path = str(directory/'MANIFEST.json')
        external[path] = training['pins'][path]
    if c['model']+'/model.safetensors' not in external or c['tasks'] not in external:
        raise ValueError('model and task pins required')
    service['policy'].update(snapshot=c['source_policy'], policy_id=c['source_policy_id'],
                             contract=c['source_contract'], adapter=c['initial_adapter'])
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'formal_package':output})
    output.mkdir(exist_ok=False)
    shutil.copytree(candidate / 'source', output / 'source')
    shutil.copytree(source_package, output / 'initial')
    verify_source(output / 'initial', source_manifest_sha256)
    (output / 'source/tools/jev/train_context_formal.py').write_text(formal)
    shutil.copyfile('tools/jev/context_formal_runtime.py', output / 'source/tools/jev/context_formal_runtime.py')
    (output / 'run.py').write_text(run)
    (output / 'evidence').mkdir()
    write(output / 'evidence/acceptance.json', evidence)
    acceptance_files = {}
    for role in ('continuous','fresh_replay','checkpoint_frequency'):
        for kind in ('report','launch'):
            p = Path(acceptance_spec[role][kind]);relative = f'evidence/{role}-{kind}.json'
            digest = evidence['bindings'][role][kind+'_sha256']
            if sha(p) != digest:raise ValueError('acceptance evidence changed during publication')
            shutil.copyfile(p, output / relative)
            if sha(output / relative) != digest:raise ValueError('copied acceptance evidence differs')
            if kind == 'report':acceptance_files[role] = relative
    for name in ('experiment','training_proof','selection_proof'):
        write(output / f'evidence/{name}.json', source_args[name])
    write(output / 'evidence/baseline.json', baseline)
    release = dict(schema='jev-context-formal-rl-release-v1', status='released', image=IMAGE,
        config_contract={k:v for k,v in c.items() if k != 'pins'}, acceptance=acceptance_files,
        source=dict(source_step=634, reset_optimizer_for_task_stage=True,
                    policy_id=binding['source_policy_id'], reference_policy_id=binding['reference_policy_id'],
                    contract=binding['source_contract'], binding_file='initial/binding.json', binding=binding),
        candidate_manifest_sha256=acceptance_spec['candidate_manifest_sha256'],
        source_package_manifest_sha256=source_manifest_sha256,
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    release['release_id'] = hashlib.sha256(json.dumps(release, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    write(output / 'RELEASE.json', release)
    c.update(formal_release=remote_root+'/RELEASE.json', formal_release_sha256=sha(output/'RELEASE.json'))
    c['pins'] = {**external, **{remote_root+'/'+str(p.relative_to(output)):sha(p)
                             for p in output.rglob('*') if p.is_file()}}
    service['pins'] = dict(c['pins'])
    write(output / 'config.json', c);write(output / 'service-config.json', service)
    result = dict(status='prepared_formal_package_no_dispatch', image=IMAGE, remote_root=remote_root,
        algorithm=algorithm, release_id=release['release_id'], source_policy_id=binding['source_policy_id'],
        mount_source_container=json.loads(Path(acceptance_spec['checkpoint_frequency']['launch']).read_text())['container_id'],
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    write(output / 'PREPARED.json', result)
    return verify(output, sha(output / 'PREPARED.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('source-package','source-manifest-sha256','source-specification','acceptance-specification',
                 'algorithm','output','remote-root'):p.add_argument('--'+name, required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
