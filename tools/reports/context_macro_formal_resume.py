"""Resume the formal source trainer at a committed sparse checkpoint.

This is an execution entry, not a deployment or completion auditor. A resumed
segment still needs an authenticated prefix and independent whole-run audit.
The fresh formal contract and running efficiency packages remain unchanged.
"""
import argparse
import ast
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from tools.reports.context_macro_formal_candidate import contract, generate as fresh_source, STEPS
from tools.reports.context_macro_trainer_candidate import topology


def expected_contract(c):
    # Runtime package intentionally excludes offline audit/deployment modules.
    result = {k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
        'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds')}
    result.update(dual_objective=dict(algorithm=c['algorithm'],arm=c['dual_arm'],ranking_coefficient=.1,
        probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id']),
        dual_runtime_identity=c['dual_runtime_identity'],optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        objective=dict(group_size=16,reward='component_binary_correctness',kl=.01,auxiliary='constant_pair0.1_nll0.3'),
        grad_clip=1.,world_size=2,macro_layout=topology(c,2,['davinci4','davinci5']))
    if 'schedule_release_sha256' in c:result['schedule_release_sha256']=c['schedule_release_sha256']
    return result


def resume_contract(config):
    step = config.get('start_update')
    root = Path(config.get('source_root', ''))
    resume = root/'resume'
    if (type(step) is not int or step not in STEPS[1:-1]
            or not root.is_absolute() or '..' in root.parts
            or config.get('resume_root') != str(resume)
            or config.get('initial_adapter') != str(resume/f'adapter-{step:04d}')):
        raise ValueError('formal recovery requires a committed nonterminal sparse cursor and sealed resume paths')
    fresh = dict(config, start_update=0)
    fresh.pop('resume_root')
    # Returning the identical plan preserves checkpoint experiment identity.
    return contract(fresh)


def validate_resume(config):
    plan = resume_contract(config)
    step = config['start_update']; root = Path(config['resume_root'])
    commit_path = root/f'checkpoint-{step:04d}-committed.json'
    def pinned(path):
        if path.is_symlink() or not path.is_file() or config['pins'].get(str(path)) != file_hash(path):
            raise ValueError('sealed recovery asset changed: '+str(path))
    pinned(commit_path)
    commit = json.loads(commit_path.read_text())
    experiment = expected_contract(config)
    experiment['formal_experiment'] = plan
    names = {f'rank-{rank}/checkpoint-{step:04d}.pt' for rank in range(2)}
    if (commit['updates'] != step or commit['contract'] != experiment
            or set(commit['rank_shards']) != names
            or commit['joint_policy_directory'] != f'policy-{step:04d}'):
        raise ValueError('formal recovery topology, cursor or objective differs')
    for name in sorted(names):
        path = root/name; pinned(path)
        if file_hash(path) != commit['rank_shards'][name]:
            raise ValueError('recovery rank shard differs from commit')
    policy = root/commit['joint_policy_directory']
    pinned(policy/'MANIFEST.json')
    if file_hash(policy/'MANIFEST.json') != commit['joint_policy_manifest_sha256']:
        raise ValueError('recovery policy differs from commit')
    bundle = read_execution_bundle(policy, expected_policy_id=commit['joint_policy']['policy_id'],
        expected_contract=config['source_contract'], expected_base_id=config['base_id'])
    verify_native_adapter(config['initial_adapter'], bundle)
    return plan


def generate(source_root):
    source = fresh_source(source_root)
    before = ('        from tools.reports.context_macro_formal_candidate import contract as formal_contract,validate_schedule\n'
              '        formal=formal_contract(c)')
    after = ('        from tools.reports.context_macro_formal_candidate import validate_schedule\n'
             '        from tools.reports.context_macro_formal_resume import validate_resume\n'
             '        formal=validate_resume(c)')
    if source.count(before) != 1:
        raise ValueError('unique formal recovery integration anchor required')
    source = source.replace(before, after)
    ast.parse(source)
    return source


def run(config, output):
    resume_contract(config)
    namespace = {'__name__': __name__+'.derived'}
    exec(compile(generate(config['source_root']), '<formal-macro-dp2-resume>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(json.loads(args.config.read_text()), args.output)
