"""One ordinary runner for all declared Stage2 training windows.

Stage-specific numerical admission remains external. This module validates
the declared window, preflights the actor before loading any inference model,
and delegates ownership and cleanup to the shared supervisor.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def window(config):
    kind=config['experiment_kind']
    if kind=='formal_source':
        from tools.reports.context_replica_formal_candidate import contract
        contract(config)
    elif kind=='cached_reference_tail_admission':
        from multimodal_jev.context_cached_tail_window import contract
        contract(config)
    else:
        from tools.reports.context_macro_trainer_candidate import experiment_window
        experiment_window(config)
        if config.get('engineering_prefix') is not True or config.get('training_dispatch_allowed') is not False:
            raise ValueError('finite window requires finite dispatch flags')
        if bool(config.get('resume_root')) != (config['start_update']==1):
            raise ValueError('step2 replay requires explicit resume; fresh windows cannot resume')
    return kind


def environment(role,visible,cache_root,base=None):
    env=dict(os.environ if base is None else base)
    env.pop('ASCEND_RT_VISIBLE_DEVICES',None)
    cache=Path(cache_root)/role;cache.mkdir(exist_ok=False)
    env['VLLM_RPC_BASE_PATH']=str(cache)
    for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR',
                'TRITON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[key]=str(cache)
    env['ASCEND_RT_VISIBLE_DEVICES']=visible
    if role in ('actor','preflight'):env['PYTHONSAFEPATH']='1'
    return env


def plan(root,config,manifest):
    # This helper is pinned beside run.py: the inherited model source does
    # not contain it and must remain byte-identical to the parity baseline.
    helper=Path(root)/'device-visibility.py'
    if helper.is_file():
        spec=importlib.util.spec_from_file_location('pinned_device_visibility',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        visibility=module.visibility
    else:
        from tools.reports.context_device_visibility import visibility
    root=Path(root);kind=window(config)
    train=manifest['training_devices'];services=manifest['inference_services']
    if len(train)!=2 or len(services)!=2 or config['physical_devices']!=train:
        raise ValueError('four-card DP2/two-instance manifest required')
    train_visible,service_visible=visibility(manifest)
    if str(root)!=config['objective_extension']['root'].rsplit('/',1)[0]:
        raise ValueError('runner must execute inside its packaged root')
    entry=root/('objective-tail-entry.py' if kind=='cached_reference_tail_admission'
                else 'objective-extension/bootstrap.py')
    return dict(train_visible=train_visible,services=service_visible,
        actor_command=['torchrun','--nnodes=1','--node-rank=0','--master-addr=127.0.0.1',
                       '--master-port=29617','--nproc-per-node='+str(len(train)),str(entry),
                       '--config',str(root/'config.json'),'--output',str(root/'run')],
        preflight_command=[sys.executable,str(root/'runtime-startup.py'),'--config',str(root/'config.json')])


def run(root,*,cache_root='/jev-tmp',command_runner=subprocess.run,supervisor=None):
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    import time
    root=Path(root).resolve();require_training_storage(root)
    config=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'PREPARED.json').read_text())
    launch=plan(root,config,manifest)
    if supervisor is None:
        spec=importlib.util.spec_from_file_location('pinned_runtime_startup',root/'runtime-startup.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        supervisor=module.supervise_services
    started=time.monotonic()
    try:
        command_runner(launch['preflight_command'],env=environment('preflight','',cache_root),check=True)
    except BaseException as error:
        commit_json(root/'startup-preflight.json',dict(status='failed',primary_error=repr(error),
                    seconds=time.monotonic()-started,services_started=False,npu_executed=False))
        raise
    commit_json(root/'startup-preflight.json',dict(status='verified_cpu',
                seconds=time.monotonic()-started,services_started=False,npu_executed=False))
    services=[]
    for name,visible in launch['services']:
        service_config=json.loads((root/manifest['layout']['services'][name]).read_text())
        services.append(dict(name=name,command=[sys.executable,str(root/'source/tools/reports/context_replica_runtime.py'),
            '--config',str(root/manifest['layout']['services'][name]),'--output',str(root/('service-'+name))],
            env=environment('service-'+name,visible,cache_root),ready_path=root/('service-'+name)/'ready.json',
            socket_path=Path(service_config['socket_path'])))
    return supervisor(services=services,actor_command=launch['actor_command'],
                      actor_env=environment('actor',launch['train_visible'],cache_root),output=root/'supervision')


if __name__=='__main__':
    report=run(Path(__file__).parent)
    raise SystemExit(0 if report['passed'] else 1)
