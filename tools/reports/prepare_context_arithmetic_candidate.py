"""Prepare a bounded numerical repair from the exact failed formal package.

Keep its full Stage1 source/reference and all data/hyperparameters. Reuse its
already accepted checkpoint trainer and supervisor, with a versioned actor
arithmetic import and checkpoint identity. No remote calls or formal release.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from tools.jev.prepare_context_formal_package import verify as verify_formal
from tools.jev.verify_context_baseline_lock import verify as verify_baseline
from tools.reports.context_probability_policy_v2 import PROFILE


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transform_actor(text):
    old = 'from .context_binary_policy import binary_logprobs'
    if text.count(old) != 1:
        raise ValueError('expected original actor arithmetic import')
    value = text.replace(old, 'from context_probability_policy_v2 import binary_logprobs')
    ast.parse(value)
    return value


def transform_trainer(text):
    old = "        contract.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),"
    if ('binary_policy_arithmetic' in text or text.count(old) != 1 or
            "if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:" not in text):
        raise ValueError('expected bounded original checkpoint trainer')
    new = """        from context_probability_policy_v2 import PROFILE as ARITHMETIC
        if c.get('binary_policy_arithmetic') != ARITHMETIC:
            raise ValueError('explicit versioned probability arithmetic required')
        contract['binary_policy_arithmetic'] = ARITHMETIC
""" + old
    value = text.replace(old, new)
    ast.parse(value)
    return value


def prepare(source, output, remote_root):
    source, output = Path(source), Path(output)
    baseline = verify_baseline('configs/releases/context-prob-v1/LOCK.json')
    manifest = verify_formal(source, sha(source/'PREPARED.json'))
    old_root = manifest['remote_root']
    if (not remote_root.startswith('/data/xingchao/multimodal-jev/simple-final-v1/context-arithmetic-')
            or '..' in Path(remote_root).parts or output.exists()):
        raise ValueError('new isolated bounded candidate root required')
    c = json.loads((source/'config.json').read_text())
    service = json.loads((source/'service-config.json').read_text())
    if c['algorithm'] != 'grpo_v1' or c['start_update'] != 0 or c['stop_update'] != 634:
        raise ValueError('exact failed formal GRPO source required')
    preserved = {k:c[k] for k in ('source_policy','source_policy_id','source_contract','initial_adapter','reference_adapter')}
    output.mkdir()
    shutil.copytree(source/'source', output/'source')
    actor = output/'source/src/multimodal_jev/context_native_actor.py'
    actor.write_text(transform_actor(actor.read_text()))
    trainer = output/'source/tools/jev/train_context_online_candidate.py'
    trainer.write_text(transform_trainer(trainer.read_text()))
    arithmetic = output/'source/tools/jev/context_probability_policy_v2.py'
    shutil.copyfile('tools/reports/context_probability_policy_v2.py', arithmetic)
    # Exact already accepted engineering runner, independent of formal release.
    runner_source = Path('.local/decision-objective-research/context-checkpoint-shared-npu-v3-package')
    runner_manifest = json.loads((runner_source/'PREPARED.json').read_text())
    if sha(runner_source/'run.py') != runner_manifest['files']['run.py']:
        raise ValueError('original engineering supervisor changed')
    shutil.copyfile(runner_source/'run.py', output/'run.py')
    c.update(start_update=0, stop_update=2, engineering_prefix=True, training_dispatch_allowed=False,
             checkpoint_interval=1, binary_policy_arithmetic=PROFILE,
             schedule=remote_root+'/source/schedule', reference_cache=remote_root+'/reference-cache')
    c.pop('formal_release');c.pop('formal_release_sha256')
    # Keep source/reference at their immutable original mounts, allowing replay
    # to relocate its own source tree without changing the reference identity.
    external = {p:h for p,h in c['pins'].items() if not p.startswith(old_root+'/')
                or p.startswith(old_root+'/initial/')}
    c['pins'] = {**external, **{remote_root+'/'+str(p.relative_to(output)):sha(p)
                              for p in (output/'source').rglob('*') if p.is_file()}}
    service['pins'] = dict(c['pins'])
    if any(c[k] != v for k,v in preserved.items()):
        raise ValueError('full Stage1 source/reference drift')
    for name, value in [('config.json',c),('service-config.json',service)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    result = dict(status='prepared_only_no_dispatch',remote_root=remote_root,
        source_package=str(source),source_manifest_sha256=sha(source/'PREPARED.json'),
        baseline=baseline,binary_policy_arithmetic=PROFILE,
        source_policy_id=c['source_policy_id'],image=manifest['image'],
        changes=['versioned_actor_probability_log_path','checkpoint_arithmetic_identity',
                 'bounded_two_update_engineering_config','checkpoint_each_update_for_replay'],
        unchanged=['full_Stage1_source_reference','model_forward','LoRA_head','reward_advantage_surrogate',
                   'data_order','global128_micro8_DP4','BF16_execution_FP32_loss','parity_thresholds'],
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','remote-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=prepare(a.source,a.output,a.remote_root)
    print(json.dumps(dict(files=len(result['files']),remote_root=result['remote_root'],status=result['status'])))
