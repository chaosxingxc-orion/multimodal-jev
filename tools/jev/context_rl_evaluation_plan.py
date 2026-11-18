"""Build an offline complete RL evaluation plan from collected checkpoints.

No dispatch or score selection. Checkpoint binding complements, and never
replaces, independent tensor/runtime audits and full-population scoring.
"""
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bound(root,proof,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative artifact path required')
    p=Path(root)/relative
    if p.is_symlink() or not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('artifact escapes collection')
    if proof['files'].get(name)!=sha(p):raise ValueError('unbound or changed artifact: '+name)
    return p


def build(*,experiment,stage1_root,stage1_proof,rl_root,rl_proof,run_id,algorithm):
    if not run_id or algorithm not in ('grpo_v1','rloo_v1'):
        raise ValueError('explicit run and registered algorithm required')
    for p in (stage1_proof,rl_proof):
        state=p['state']
        if state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']:
            raise ValueError('successful terminal collection required')
    if rl_proof.get('passed') is not True:
        raise ValueError('independently admitted complete RL collection required')
    c=json.loads(bound(rl_root,rl_proof,'config.json').read_text())
    if (c['stage']!='task' or c['algorithm']!=algorithm or c['engineering_prefix'] is not False
            or c['training_dispatch_allowed'] is not True or c['start_update']!=0
            or c['stop_update']!=experiment['task_stage']['updates']
            or c['release_sha256']!=experiment['release_sha256']):
        raise ValueError('full formal RL run, algorithm and population must match')
    steps=experiment['evaluation']['checkpoints']
    if steps!=[0,*range(64,634,64),634]:raise ValueError('complete fixed 11-point curve required')
    def checkpoint(root,proof,step):
        name=f'run/checkpoint-{step:04d}-committed.json'
        commit=json.loads(bound(root,proof,name).read_text())
        if commit['updates']!=step:raise ValueError('checkpoint step differs')
        directory='run/'+commit['joint_policy_directory']
        manifest_path=bound(root,proof,directory+'/MANIFEST.json')
        if sha(manifest_path)!=commit['joint_policy_manifest_sha256']:
            raise ValueError('checkpoint manifest differs from commit')
        manifest=json.loads(manifest_path.read_text())
        if manifest['policy_id']!=commit['joint_policy']['policy_id']:
            raise ValueError('checkpoint policy differs from commit')
        for asset,digest in manifest['files'].items():
            if sha(bound(root,proof,directory+'/'+asset))!=digest:
                raise ValueError('policy manifest asset differs')
        return commit,manifest,dict(collection=str(root),policy_relative_path=directory,
                                  policy_id=manifest['policy_id'],commit_sha256=sha(Path(root)/name))
    source_config=json.loads(bound(stage1_root,stage1_proof,'config.json').read_text())
    if (source_config['stage']!='instruction' or source_config['engineering_prefix'] is not False
            or source_config['stop_update']!=experiment['instruction_stage']['updates']
            or source_config['release_sha256']!=experiment['release_sha256']):
        raise ValueError('completed full Stage1 source required')
    initial_commit,initial,entry=checkpoint(stage1_root,stage1_proof,experiment['instruction_stage']['updates'])
    if initial_commit['contract']['stage']!='instruction':
        raise ValueError('step0 source is not instruction-stage checkpoint')
    if (initial['contract'].get('token',{}).get('registration_status')!='admitted'
            or initial['contract'].get('request_version')!='jev-context-exclusive_event-v1'):
        raise ValueError('admitted exclusive-event policy required')
    if initial['policy_id']!=c['source_policy_id'] or initial['contract']!=c['source_contract']:
        raise ValueError('RL step0 must be completed Stage1 reference')
    points=[dict(step=0,source_stage='instruction',source_step=634,**entry)]
    for step in steps[1:]:
        commit,manifest,entry=checkpoint(rl_root,rl_proof,step)
        contract=commit['contract']
        if (contract['stage']!='task' or contract['algorithm']!=algorithm
                or contract['source_policy_id']!=initial['policy_id']
                or contract['release_sha256']!=experiment['release_sha256']
                or manifest['contract']!=initial['contract']):
            raise ValueError('RL checkpoint algorithm/reference/contract differs')
        points.append(dict(step=step,source_stage='task',source_step=step,**entry))
    return dict(run_id=run_id,algorithm=algorithm,stage='task',initial_policy_id=initial['policy_id'],
        checkpoints=points,selection_recordings=experiment['selection_recordings'],
        threshold=experiment['evaluation']['set_threshold'],
        status='bound_plan_only_no_inference',
        scope='Artifact-bound full curve plan; tensor equivalence, runtime admission and actual full selection remain separate')
