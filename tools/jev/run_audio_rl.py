#!/usr/bin/env python3
"""Sequential sample → actor → export schedule on explicitly allocated hardware.

Each subprocess exits before the next starts; vLLM weights cannot remain stale
in an engine across updates. No resource discovery or preemption is performed.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.rl_preflight import validate_schedule


def completed(path):
    terminal=path/'terminal.json'
    if not terminal.exists():return False
    state=json.loads(terminal.read_text())
    if state['exit_code']!=0:raise RuntimeError(f'failed stage retained: {path}; inspect before retry')
    if not (path/'result.json').exists():raise RuntimeError(f'success without result: {path}')
    return True


def execute(tool,config,output):
    config_path=output.parent/(output.name+'-config.json')
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('stage config changed')
    atomic_json(config_path,config)
    if completed(output):return
    # A disappeared host process may leave an incomplete output. Never infer that
    # a child is dead from a missing terminal; require inspection before resuming.
    if output.exists():raise RuntimeError(f'incomplete stage requires process inspection: {output}')
    with (output.parent/(output.name+'.log')).open('x') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name(tool)),
                        '--config',str(config_path),'--output',str(output)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    if not completed(output):raise RuntimeError('stage did not write terminal receipt')


def export_actor(actor,export_root,stage):
    if completed(stage):
        result=json.loads((stage/'result.json').read_text())
        if checkpoint_identity(result['checkpoint'])[0]!=result['policy_id']:
            raise ValueError('published policy changed')
        return result['checkpoint']
    if stage.exists():raise RuntimeError('incomplete export requires inspection')
    stage.mkdir()
    try:
        with (stage/'export.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('export_audio_policy.py')),
                            '--run',str(actor),'--output-root',str(export_root)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        matches=[]
        for receipt in export_root.glob('*/EXPORT.json'):
            if receipt.parent.name.startswith('.'):continue
            report=json.loads(receipt.read_text())
            if Path(report['training_run']).resolve()==actor.resolve() and report['status']=='passed':
                matches.append(receipt.parent)
        if len(matches)!=1:raise ValueError('expected one successful export of this actor')
        checkpoint=str(matches[0]);policy,_=checkpoint_identity(checkpoint)
        atomic_json(stage/'result.json',{'checkpoint':checkpoint,'policy_id':policy})
        atomic_json(stage/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
        return checkpoint
    except BaseException as exc:
        atomic_json(stage/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise


def run(config,root):
    source=Path(__file__).parents[2]
    preflight=validate_schedule(config)
    identity={'config':config,**preflight,
              'source_sha256':{str(p.relative_to(source)):file_hash(p) for directory in
                               (source/'src/multimodal_jev',source/'tools/jev') for p in sorted(directory.glob('*.py'))}}
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())!=identity:
        raise ValueError('schedule identity changed')
    atomic_json(root/'identity.json',identity)
    checkpoint=config['initial_checkpoint'];parent=None;total_updates=0
    for i,example_ids in enumerate(config['batches']):
        stage=root/f'iteration-{i:04d}';stage.mkdir(exist_ok=True)
        atomic_json(root/'progress.json',{'iteration':i,'phase':'rollout','heartbeat_unix':time.time()})
        rollout_config={'checkpoint':checkpoint,'manifest':config['manifest'],'example_ids':example_ids,
                        'sampling':{**config['sampling'],'seed':config['seed']+i},'engine':config['engine']}
        if 'audio_input_mode' in config:rollout_config['audio_input_mode']=config['audio_input_mode']
        execute('sample_audio_policy.py',rollout_config,stage/'rollout')
        # The actor's immutable base is explicit in the actor section.  The
        # rollout checkpoint may be a merged export from the previous
        # iteration, while model_path must remain the original base so LoRA
        # coordinates and optimizer state stay compatible.
        actor_config={**config['actor'],'rollout_run':str(stage/'rollout'),
                      'parent_run':str(parent) if parent else None,
                      'initial_adapter':config.get('initial_adapter') if parent is None else None,
                      'seed':config['seed']+i,'reward_mode':config['reward_mode'],
                      'pair_bonus_weight':config['pair_bonus_weight']}
        atomic_json(root/'progress.json',{'iteration':i,'phase':'actor','heartbeat_unix':time.time()})
        execute('update_audio_actor.py',actor_config,stage/'actor')
        actor_result=json.loads((stage/'actor/result.json').read_text())
        total_updates=actor_result['updates'];parent=stage/'actor'
        atomic_json(root/'progress.json',{'iteration':i,'phase':'export','heartbeat_unix':time.time()})
        # Equal-reward no-op iterations can export identical weights. Separate
        # iteration roots retain the distinct lineage without overwriting receipts.
        checkpoint=export_actor(parent,Path(config['export_root'])/f'iteration-{i:04d}',stage/'export')
    atomic_json(root/'result.json',{'status':'passed','iterations':len(config['batches']),
                'optimizer_updates':total_updates,'checkpoint':checkpoint,'last_actor_run':str(parent),
                'scope':'completed on-policy iteration schedule; independent evaluation still required'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'schedule.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'result.json').exists():raise FileExistsError('completed schedule')
        try:run(json.loads(a.config.read_text()),a.output)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(a.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
