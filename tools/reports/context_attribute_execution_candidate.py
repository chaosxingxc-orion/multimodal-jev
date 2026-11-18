"""Bounded attribute actor/service configuration using the existing supervisor.

This module has no SSH, Docker or device acquisition. The caller must own a
fixed-image DP4 lease. Source changes are rejected before subprocess creation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

from multimodal_jev.baseline import file_hash
from tools.reports.context_attribute_actor_candidate import REFERENCE_ID

PREFIX='/data/xingchao/multimodal-jev/simple-final-v1/'
DEPENDENCIES={
    'tools/jev/context_process_supervisor.py':'f02c8d89536f1c7e8ea50a851a6f41f2e905d512c976f98c064fc1ba123aeaa2',
    'tools/jev/serve_context_rollout.py':'25ca84ed3e04e7d8b5910489b54b1d8d3076f77608812c027f0d086fa41c1fc9',
    'tools/reports/context_attribute_training_candidate.py':'61fd944531af68a59eb9c08b78a849979dae5b59b8c515294e2eca2358246124',
}


def authenticate(source_root):
    for name,h in DEPENDENCIES.items():
        if file_hash(Path(source_root)/name)!=h:
            raise ValueError('Shared execution dependency changed: '+name)


def configurations(base,service,*,root,source_root,dataset,schedule,schedule_sha256,arm,algorithm,pins):
    """Build an initial two-update candidate; no remote or filesystem writes."""
    if (not re.fullmatch(re.escape(PREFIX)+r'context-attribute-[a-z0-9-]+',root)
            or arm not in ('direct','q_input','q_credit') or algorithm not in ('grpo_v1','rloo_v1')
            or base['source_policy_id']!=REFERENCE_ID
            or base['physical_devices']!=[4,5,6,7] or base['global_batch']!=128 or base['microbatch']!=8
            or base['binary_policy_arithmetic']!='context-bernoulli-probability-cpu-log-v2'
            or len(schedule_sha256)!=64 or any(c not in '0123456789abcdef' for c in schedule_sha256)):
        raise ValueError('Frozen bounded attribute deployment required')
    if (service['policy']['policy_id']!=REFERENCE_ID or service['policy']['snapshot']!=base['source_policy']
            or service['policy']['adapter']!=base['initial_adapter']
            or service['policy']['contract']!=base['source_contract']):
        raise ValueError('Same fixed initial policy required')
    c=copy.deepcopy(base);s=copy.deepcopy(service)
    for k in ('formal_release','formal_release_sha256','resume_root'):c.pop(k,None)
    c.update(start_update=0,stop_update=2,checkpoint_interval=1,engineering_prefix=True,
        training_dispatch_allowed=False,algorithm=algorithm,attribute_arm=arm,
        attribute_source_root=source_root,attribute_dataset=dataset,attribute_schedule=schedule,
        attribute_schedule_sha256=schedule_sha256,encoded_audio_cache=root+'/encoded-cache',
        reference_cache=root+'/reference-cache',socket_path='/jev-exchange/engine.sock',
        exchange='/jev-exchange',pins=copy.deepcopy(pins))
    s.update(socket_path=c['socket_path'],exchange=c['exchange'],pins=copy.deepcopy(pins))
    s['policy'].update(max_decisions=60 if arm=='direct' else 124,max_model_len=4096,
        physical_devices=[7],tensor_parallel_size=1,
        process_device_scope=dict(kind='same_container_runtime_visibility_v1',mounted_physical=[4,5,6,7],visible_ordinal='3'))
    validate_pair(c,s)
    return c,s


def validate_pair(c,s):
    policy=s['policy']
    if (c.get('engineering_prefix') is not True or c.get('training_dispatch_allowed') is not False
            or c.get('attribute_arm') not in ('direct','q_input','q_credit')
            or c['algorithm'] not in ('grpo_v1','rloo_v1')
            or type(c['start_update']) is not int or type(c['stop_update']) is not int
            or (c['start_update'],c['stop_update']) not in ((0,2),(1,2))
            or c['checkpoint_interval']!=1 or c['source_policy_id']!=REFERENCE_ID
            or c['global_batch']!=128 or c['microbatch']!=8 or c['physical_devices']!=[4,5,6,7]
            or c['binary_policy_arithmetic']!='context-bernoulli-probability-cpu-log-v2'
            or s['pins']!=c['pins'] or s['socket_path']!=c['socket_path'] or s['exchange']!=c['exchange']
            or c['socket_path']!='/jev-exchange/engine.sock' or c['exchange']!='/jev-exchange'
            or policy['contract']!=c['source_contract'] or policy['base_id']!=c['base_id']
            or policy['model']!=c['model'] or policy['max_model_len']!=4096
            or policy['max_decisions']!=(60 if c['attribute_arm']=='direct' else 124)
            or policy['physical_devices']!=[7] or policy['tensor_parallel_size']!=1
            or policy['process_device_scope']!=dict(kind='same_container_runtime_visibility_v1',mounted_physical=[4,5,6,7],visible_ordinal='3')):
        raise ValueError('Actor/service bounded configuration differs')
    if c['start_update']==0:
        if (c.get('resume_root') or policy['policy_id']!=REFERENCE_ID
                or policy['snapshot']!=c['source_policy'] or policy['adapter']!=c['initial_adapter']):
            raise ValueError('Fresh initial strategy required')
    else:
        resume=Path(c['resume_root']);commit_path=resume/'checkpoint-0001-committed.json'
        if c['pins'].get(str(commit_path))!=file_hash(commit_path):
            raise ValueError('Pinned committed update1 required')
        commit=json.loads(commit_path.read_text())
        joint=resume/commit['joint_policy_directory']
        if (commit['updates']!=1 or len(commit['rank_shards'])!=4
                or file_hash(joint/'MANIFEST.json')!=commit['joint_policy_manifest_sha256']
                or policy['snapshot']!=str(joint) or policy['adapter']!=c['initial_adapter']
                or policy['policy_id']!=json.loads((joint/'MANIFEST.json').read_text())['policy_id']):
            raise ValueError('Resume service must start at committed current policy')
        for path,h in commit['rank_shards'].items():
            if file_hash(resume/path)!=h:raise ValueError('Resume shard changed')
    return True


def run(package):
    from multimodal_jev.local_runtime_storage import require_training_storage
    from tools.reports.context_attribute_training_candidate import AttributeTrainingRuntime
    from context_process_supervisor import supervise
    root=Path(package).resolve();require_training_storage(root)
    c=json.loads((root/'config.json').read_text());s=json.loads((root/'service-config.json').read_text())
    validate_pair(c,s)
    authenticate(c['attribute_source_root'])
    runtime=AttributeTrainingRuntime(c)
    if c['start_update']:
        commit=json.loads((Path(c['resume_root'])/'checkpoint-0001-committed.json').read_text())
        if commit['contract']!=runtime.contract:raise ValueError('Resume arm/objective contract differs')
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4','davinci5','davinci6','davinci7']:
        raise ValueError('Exclusively leased DP4 device scope required')
    source=Path(c['attribute_source_root']);envs={}
    for role in ('actor','service'):
        env=dict(os.environ);env.pop('ASCEND_RT_VISIBLE_DEVICES',None)
        cache=Path('/jev-tmp')/role;cache.mkdir(exist_ok=True)
        env['VLLM_RPC_BASE_PATH']=str(cache)
        for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
                    'ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):env[key]=str(cache)
        if role=='service':env['ASCEND_RT_VISIBLE_DEVICES']='3'
        envs[role]=env
    return supervise(service_command=[sys.executable,str(source/'tools/jev/serve_context_rollout.py'),
        '--config',str(root/'service-config.json'),'--output',str(root/'service')],
        actor_command=['torchrun','--nnodes=1','--node-rank=0','--master-addr=127.0.0.1',
            '--master-port=29617','--nproc-per-node=4',str(source/'tools/reports/context_attribute_training_candidate.py'),
            '--config',str(root/'config.json'),'--output',str(root/'run')],
        service_env=envs['service'],actor_env=envs['actor'],output=root/'supervision',
        ready_path=root/'service/ready.json',socket_path=c['socket_path'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);a=p.parse_args()
    raise SystemExit(0 if run(a.package)['passed'] else 1)
