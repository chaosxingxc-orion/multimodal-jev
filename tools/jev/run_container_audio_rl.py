#!/usr/bin/env python3
"""CPU rollout/actor/export schedule using distinct pinned images and hosts."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.container_executor import ContainerExecutor
from multimodal_jev.policy_identity import checkpoint_identity,validate_adapter_export


def run(config,runtime,root):
    source=Path(__file__).resolve().parents[2]
    root=Path(root).resolve()
    if config['actor']['device']!='cpu' or config['actor']['dtype']!='float32':
        raise ValueError('this registered transport supports CPU float32 only')
    source.relative_to(root) # caller must execute a retained snapshot under the run
    executor=ContainerExecutor(root,source,runtime)
    runtime_path=root/'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text())!=runtime:
        raise ValueError('runtime changed')
    atomic_json(runtime_path,runtime)
    spec=importlib.util.spec_from_file_location('local_schedule',Path(__file__).with_name('run_audio_rl.py'))
    schedule=importlib.util.module_from_spec(spec);spec.loader.exec_module(schedule)

    def execute(tool,child,output):
        path=output.parent/(output.name+'-config.json')
        if path.exists() and json.loads(path.read_text())!=child:raise ValueError('stage config changed')
        atomic_json(path,child)
        role='rollout' if tool=='sample_audio_policy.py' else 'actor'
        executor.execute(role,output,['-B',str(source/'tools/jev'/tool),'--config',str(path),'--output',str(output)])
        if not schedule.completed(output):raise RuntimeError('successful Docker exit without application result')

    def export_actor(actor,export_root,stage):
        if not (actor/'terminal.json').exists():raise ValueError('completed actor required')
        export_root.resolve().relative_to(Path(runtime['shared_root']).resolve())
        export_root.mkdir(parents=True,exist_ok=True)
        wrapper=stage.parent/'export-stage.py'
        code=('from pathlib import Path\nimport runpy\n'
              'm=runpy.run_path('+repr(str(source/'tools/jev/run_audio_rl.py'))+')\n'
              "m['export_actor'](Path("+repr(str(actor))+'),Path('+repr(str(export_root))+'),Path('+repr(str(stage))+'))\n')
        if wrapper.exists() and wrapper.read_text()!=code:raise ValueError('export wrapper changed')
        wrapper.write_text(code)
        executor.execute('export',stage,['-B',str(wrapper)],
                         [str(export_root)+':'+str(export_root)+':rw'])
        if not schedule.completed(stage):raise RuntimeError('export missing terminal/result')
        report=json.loads((stage/'result.json').read_text());checkpoint=report['checkpoint']
        if checkpoint_identity(checkpoint)[0]!=report['policy_id']:raise ValueError('published policy changed')
        validate_adapter_export(config['actor']['model_path'],checkpoint,actor/'adapter',expected_training_run=actor)
        return checkpoint

    schedule.execute=execute;schedule.export_actor=export_actor
    schedule.run(config,root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'container-schedule.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'result.json').exists():raise FileExistsError('completed schedule')
        try:run(json.loads(a.config.read_text()),json.loads(a.runtime.read_text()),a.output)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(a.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
