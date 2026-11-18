#!/usr/bin/env python3
"""Prepare-only same-container candidate from authenticated failed-v1 sources.

No remote calls, no launch permission, no changes to the frozen trainer. The
only existing engine source change is its explicit process-scope validator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


RUNNER = '''import json,os,sys
from pathlib import Path
from context_process_supervisor import supervise
from multimodal_jev.local_runtime_storage import require_training_storage
r=Path(__file__).resolve().parent
require_training_storage(r)
c=json.loads((r/'config.json').read_text())
if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:
 raise ValueError('two-update engineering candidate only')
if c['start_update']!=0 or c['stop_update']!=2:
 raise ValueError('initial bounded candidate only; resume requires separate binding')
envs={}
for role in ('actor','service'):
 env=dict(os.environ);env.pop('ASCEND_RT_VISIBLE_DEVICES',None)
 cache=r/(role+'-cache')
 cache.mkdir(exist_ok=True)
 for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR',
             'TRITON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
  env[key]=str(cache)
 if role=='service':env['ASCEND_RT_VISIBLE_DEVICES']='3'
 envs[role]=env
report=supervise(service_command=[sys.executable,str(r/'source/tools/jev/serve_context_rollout.py'),
 '--config',str(r/'service-config.json'),'--output',str(r/'service')],
 actor_command=['torchrun','--nnodes=1','--node-rank=0','--master-addr=127.0.0.1',
 '--master-port=29617','--nproc-per-node=4',str(r/'source/tools/jev/train_context_online_candidate.py'),
 '--config',str(r/'config.json'),'--output',str(r/'run')],
 service_env=envs['service'],actor_env=envs['actor'],output=r/'supervision',
 ready_path=r/'service/ready.json',socket_path=c['socket_path'])
raise SystemExit(0 if report['passed'] else 1)
'''


def prepare(output, remote_root):
    local=Path('.local/decision-objective-research')
    old=json.loads((local/'context-online-npu-v1-prepared.json').read_text())
    pins=json.loads((local/'context-online-acceptance-v1-pins.json').read_text())
    for name,digest in pins.items():
        if sha(name)!=digest:
            raise ValueError('original candidate changed: '+name)
    if not remote_root.startswith('/data/xingchao/multimodal-jev/simple-final-v1/'):
        raise ValueError('explicit local hot-storage root required')
    prior='/data/xingchao/multimodal-jev/simple-final-v1/context-online-npu-v1'
    if remote_root==prior:
        raise ValueError('failed v1 must remain immutable')
    output=Path(output);output.mkdir(exist_ok=False)
    copied={}
    for name,digest in old['source_files'].items():
        if sha(name)!=digest:raise ValueError('source changed: '+name)
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target);copied[name]=digest
    # The numerical engine path stays byte-identical apart from this import.
    engine=output/'source/tools/jev/context_switch_rollout.py'
    before=engine.read_text()
    source='from multimodal_jev.span_device_scope import validate_tp1'
    replacement='from context_shared_device_scope import validate_tp1'
    if before.count(source)!=1:raise ValueError('unexpected engine initialization source')
    engine.write_text(before.replace(source,replacement))
    extras=['tools/jev/context_shared_device_scope.py','tools/jev/context_process_supervisor.py']
    for name in extras:
        shutil.copyfile(name,output/'source'/name)
    shutil.copytree(local/'context-full-learning-schedule-v1',output/'source/schedule')
    config=json.loads((local/'context-online-npu-v1-config.json').read_text())
    service=json.loads((local/'context-online-npu-v1-service-config.json').read_text())
    def relocate(value):
        if isinstance(value,str):return value.replace(prior,remote_root)
        if isinstance(value,list):return [relocate(x) for x in value]
        if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
        return value
    config=relocate(config);service=relocate(service)
    for path in (output/'source').rglob('*'):
        if path.is_file():config['pins'][remote_root+'/'+str(path.relative_to(output))]=sha(path)
    service['pins']=config['pins']
    service['policy']['process_device_scope']=dict(kind='same_container_runtime_visibility_v1',
        mounted_physical=[4,5,6,7],visible_ordinal='3')
    config['diagnostic_scope']='Two-update same-container engineering candidate; not released Stage2'
    for name,value in [('config.json',config),('service-config.json',service)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'run.py').write_text(RUNNER)
    manifest=dict(status='prepared_only_no_dispatch',remote_root=remote_root,
        original_source_files=copied,original_acceptance_pins_sha256=sha(local/'context-online-acceptance-v1-pins.json'),
        source_change=dict(path='tools/jev/context_switch_rollout.py',before=source,after=replacement),
        new_modules=extras,
        required_before_launch=['independently_collected_device_sharing_probe_passed',
                               'exact_probe_physical_mapping_7_and_4_5_6_7',
                               'shared_device_lease_and_free_devices',
                               'versioned_launcher_and_independent_collector'],
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'PREPARED.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--remote-root',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.remote_root),indent=2))
