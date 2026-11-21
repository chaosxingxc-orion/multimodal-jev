"""Ordinary configured selection runner for committed Stage1/Stage2 policies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def run(root):
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from context_process_supervisor import stop_owned
    root=Path(root).resolve();require_training_storage(root)
    c=json.loads((root/'parity-hf.json').read_text());m=json.loads((root/'PREPARED.json').read_text())
    if c['expected_population']!=9226 or c['source_step'] not in [0,*range(64,634,64),634]:
        raise ValueError('registered full GRPO selection point required')
    t=c['evaluation_topology'];devices=sorted(t['devices'])
    if m['training_devices'] or [s['device'] for s in m['inference_services']]!=t['devices']:
        raise ValueError('inference-only declared topology required')
    def env(role,visible):
        e=dict(os.environ);e.pop('ASCEND_RT_VISIBLE_DEVICES',None)
        cache=Path('/jev-tmp')/role;cache.mkdir(exist_ok=False)
        for k in ('VLLM_RPC_BASE_PATH','TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):e[k]=str(cache)
        e['ASCEND_RT_VISIBLE_DEVICES']=visible
        return e
    spec=importlib.util.spec_from_file_location('selection_supervisor',root/'runtime-startup.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    control=root/'hf-process';control.mkdir()
    with (control/'process.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(root/'selection-client.py'),'--config',str(root/'parity-template.json'),
            '--local',str(root/'parity-hf-local.json'),'--phase','hf'],env=env('parity-hf',c['process_device_scope']['visible_ordinal']),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait()
        finally:stop_owned(child,30.)
    commit_json(control/'terminal.json',dict(exit_code=code,pid=child.pid))
    if code:raise SystemExit(code)
    services=[]
    for s in m['inference_services']:
        name=s['name'];service=json.loads((root/m['layout']['services'][name]).read_text())
        services.append(dict(name=name,command=[sys.executable,str(root/'source/tools/reports/context_replica_runtime.py'),
            '--config',str(root/m['layout']['services'][name]),'--output',str(root/('service-'+name))],
            env=env('service-'+name,str(devices.index(s['device']))),ready_path=root/('service-'+name)/'ready.json',socket_path=Path(service['socket_path'])))
    actor="import json,sys,runpy;from pathlib import Path;from multimodal_jev.baseline import file_hash;run=runpy.run_path(str(Path(sys.argv[1]).parent/'selection-client.py'))['run'];c=json.loads(Path(sys.argv[1]).read_text());c['hf_reference']={'path':sys.argv[2],'sha256':file_hash(sys.argv[2])};run(c,{'source':'committed_selection'},'macro')"
    report=helper.supervise_services(services=services,actor_command=[sys.executable,'-c',actor,str(root/'parity-macro.json'),str(root/'hf/reference.json')],actor_env=env('selection-client',''),output=root/'supervision')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':run(Path(__file__).parent)
