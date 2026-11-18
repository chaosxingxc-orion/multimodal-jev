"""Publish repaired full-run packages from re-audited actual NPU evidence.

Preserve the original formal package and 364-file lock. Both algorithms share
the exact repaired actor/trainer, full Stage1 source and unchanged experiment.
No remote calls. Formal training starts at update zero with a fresh optimizer.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

from tools.jev.prepare_context_formal_package import verify as verify_package, IMAGE, PREFIX, write
from tools.jev.prepare_context_formal_runtime import GUARD, CANDIDATE_SHA
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.context_stage2_source import bind
from tools.jev.verify_context_baseline_lock import verify as verify_lock
from tools.reports.context_arithmetic_acceptance import verify as accept
from tools.reports.prepare_context_arithmetic_candidate import transform_trainer, transform_actor
from tools.reports.context_probability_policy_v2 import PROFILE


def formal_trainer(candidate, original):
    if hashlib.sha256(original.encode()).hexdigest() != CANDIDATE_SHA:
        raise ValueError('exact accepted checkpoint trainer required')
    if candidate != transform_trainer(original) or candidate.count(GUARD) != 1:
        raise ValueError('only accepted arithmetic transform allowed')
    result = candidate.replace(GUARD,
        '        from context_arithmetic_formal_runtime import validate\n        validate(c)\n')
    ast.parse(result)
    return result


def relocate(value, old, new):
    if isinstance(value, str): return value.replace(old, new)
    if isinstance(value, list): return [relocate(x, old, new) for x in value]
    if isinstance(value, dict): return {relocate(k, old, new): relocate(v, old, new) for k,v in value.items()}
    return value


def prepare(*, original_package, original_manifest_sha256, acceptance_specification,
            source_specification, algorithm, output, remote_root):
    if algorithm not in ('grpo_v1', 'rloo_v1'):
        raise ValueError('registered algorithm required')
    family = 'grpo' if algorithm == 'grpo_v1' else 'rloo'
    if not re.fullmatch(re.escape(PREFIX) + 'context-rl-' + family + r'-[a-z0-9-]+', remote_root):
        raise ValueError('isolated matching formal root required')
    original, output = Path(original_package), Path(output)
    baseline = verify_lock('configs/releases/context-prob-v1/LOCK.json')
    prior = verify_package(original, original_manifest_sha256)
    old_release = json.loads((original/'RELEASE.json').read_text())
    spec = json.loads(Path(source_specification).read_text())
    source_args = dict(spec)
    for key in ('experiment','training_proof','selection_proof'):
        source_args[key] = json.loads(Path(spec[key]).read_text())
        if source_args[key] != json.loads((original/f'evidence/{key}.json').read_text()):
            raise ValueError('original full Stage1 evidence changed')
    if source_args['experiment'] != json.loads(Path('configs/releases/context-prob-v1/experiment.json').read_text()):
        raise ValueError('frozen experiment required')
    binding = bind(**source_args)
    if binding != old_release['source']['binding']:
        raise ValueError('revalidated full Stage1 differs')
    a = json.loads(Path(acceptance_specification).read_text())
    evidence = accept(**a)
    if evidence['source_policy_id'] != binding['source_policy_id']:
        raise ValueError('arithmetic acceptance must use the full Stage1 source')
    old_checkpoint = evidence['reused_checkpoint_frequency']['evidence']['bindings']['checkpoint_frequency']
    if sha(original/old_release['acceptance']['checkpoint_frequency']) != old_checkpoint['report_sha256']:
        raise ValueError('reused checkpoint evidence differs from original release')
    candidate = Path(a['continuous']['package'])
    for root in (original, candidate, Path(spec['training_root']), Path(spec['selection_root'])):
        if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
            raise ValueError('independent package required')
    rel = 'source/tools/jev/train_context_online_candidate.py'
    trainer = formal_trainer((candidate/rel).read_text(), (original/rel).read_text())
    actor_rel = 'source/src/multimodal_jev/context_native_actor.py'
    actor = transform_actor((original/actor_rel).read_text())
    if actor != (candidate/actor_rel).read_text():
        raise ValueError('accepted actor required')
    run = (original/'run.py').read_text()
    old_import = 'from context_formal_runtime import validate'
    if run.count(old_import) != 1:
        raise ValueError('original formal supervisor required')
    run = run.replace(old_import, 'from context_arithmetic_formal_runtime import validate')
    ast.parse(run)
    c = relocate(json.loads((original/'config.json').read_text()), prior['remote_root'], remote_root)
    service = relocate(json.loads((original/'service-config.json').read_text()), prior['remote_root'], remote_root)
    c.update(algorithm=algorithm, binary_policy_arithmetic=PROFILE)
    if c['start_update'] != 0 or c['stop_update'] != 634 or c.get('resume_root'):
        raise ValueError('fresh full task stage required')
    external = {p:h for p,h in c['pins'].items() if not p.startswith(remote_root+'/')}
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'formal_package':output})
    output.mkdir(exist_ok=False)
    for folder in ('source','initial','evidence'):
        shutil.copytree(original/folder, output/folder)
    (output/actor_rel).write_text(actor)
    (output/rel).write_text((candidate/rel).read_text())
    (output/'source/tools/jev/train_context_formal.py').write_text(trainer)
    for name in ('context_probability_policy_v2.py','context_arithmetic_formal_runtime.py'):
        shutil.copyfile(Path('tools/reports')/name, output/'source/tools/jev'/name)
    (output/'run.py').write_text(run)
    write(output/'evidence/baseline.json', baseline)
    write(output/'evidence/arithmetic-acceptance.json', evidence)
    write(output/'evidence/acceptance.json', evidence)
    for role in ('continuous','fresh_replay'):
        for kind in ('report','launch'):
            p = Path(a[role][kind]); expected = evidence['bindings'][role][kind]['sha256']
            if sha(p) != expected: raise ValueError('acceptance artifact changed')
            shutil.copyfile(p, output/f'evidence/{role}-{kind}.json')
    # checkpoint_frequency retains its original report and explicitly qualified
    # reuse in arithmetic-acceptance.json; it is not a new-policy trajectory.
    release = dict(old_release)
    release.pop('release_id')
    release.update(config_contract={k:v for k,v in c.items() if k not in ('pins','formal_release','formal_release_sha256')},
        candidate_manifest_sha256=sha(candidate/'PREPARED.json'),
        numerical_repair=dict(profile=PROFILE, acceptance_file='evidence/arithmetic-acceptance.json',
            original_release_id=old_release['release_id'], original_manifest_sha256=original_manifest_sha256),
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    release['release_id'] = hashlib.sha256(json.dumps(release,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    write(output/'RELEASE.json', release)
    c.update(formal_release=remote_root+'/RELEASE.json', formal_release_sha256=sha(output/'RELEASE.json'))
    c['pins'] = {**external, **{remote_root+'/'+str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}}
    service['pins'] = dict(c['pins'])
    write(output/'config.json',c); write(output/'service-config.json',service)
    result = dict(status='prepared_formal_package_no_dispatch', image=IMAGE, remote_root=remote_root,
        algorithm=algorithm, release_id=release['release_id'], source_policy_id=binding['source_policy_id'],
        binary_policy_arithmetic=PROFILE,
        mount_source_container=json.loads(Path(a['fresh_replay']['launch']).read_text())['container_id'],
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    write(output/'PREPARED.json', result)
    return verify_package(output,sha(output/'PREPARED.json'))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('original-package','original-manifest-sha256','acceptance-specification',
                 'source-specification','algorithm','output','remote-root'):
        p.add_argument('--'+name, required=True)
    result=prepare(**vars(p.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
