"""Seal a finite actual-audio serial/batch probe; local preparation only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tools.reports.context_batched_owner_candidate import PROFILE

IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
ROOT='/data/xingchao/multimodal-jev/simple-final-v1/context-batch-probe-npu-v1'
REFERENCE='9164eddffd9e3fdb21c060a72827b1a4324f3d45e60eb0d491b589b194db0e66'
TOOLS=['context_segmented_attention_candidate.py','context_segmented_audio_candidate.py',
       'context_segmented_worker_candidate.py','context_batched_request_candidate.py',
       'context_batched_owner_candidate.py','context_batched_worker_extension_candidate.py',
       'context_batched_service_candidate.py','context_bulk_rpc_candidate.py',
       'context_batch_measurement_candidate.py','run_context_batch_probe_candidate.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(model):
    argv=['docker','create','--name','context-batch-probe-npu-v1','--pull=never','--network=none',
          '--cpus=8','--memory=64g','--shm-size=2g']
    for name in ('davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv+=['--device','/dev/'+name+':/dev/'+name]
    for mount in (ROOT+':'+ROOT,ROOT+'/package:'+ROOT+'/package:ro',ROOT+'/home:/root',
                  ROOT+'/cache:/jev-tmp',model+':'+model+':ro',
                  '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):
        argv+=['-v',mount]
    env=dict(PYTHONPATH=ROOT+'/package:'+ROOT+'/package/src:'+ROOT+'/package/tools/jev',
             PYTHONDONTWRITEBYTECODE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',
             TMPDIR='/jev-tmp',VLLM_RPC_BASE_PATH='/jev-tmp',HCCL_DETERMINISTIC='true')
    for name in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[name]='/jev-tmp'
    for k,v in env.items():argv+=['-e',k+'='+v]
    return argv+['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',
        ROOT+'/package/tools/reports/run_context_batch_probe_candidate.py',
        '--package',ROOT+'/package','--output',ROOT+'/evaluation']


def prepare(source, formal, inputs, output):
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    source,formal,inputs,output=map(lambda p:Path(p).resolve(),(source,formal,inputs,output))
    policy=json.loads((formal/'service-config.json').read_text())['policy']
    if policy['policy_id']!=REFERENCE:raise ValueError('fixed Stage1 source required')
    bundle=read_execution_bundle(formal/'initial/policy',expected_policy_id=REFERENCE,
        expected_contract=policy['contract'],expected_base_id=policy['base_id'])
    verify_native_adapter(formal/'initial/adapter',bundle)
    manifest=json.loads((inputs/'MANIFEST.json').read_text())
    if manifest['rows']!=128 or sha(inputs/'inputs.pt')!=manifest['input_sha256']:
        raise ValueError('complete authenticated TRAIN128 inputs required')
    output.mkdir(exist_ok=False);package=output/'package';package.mkdir()
    for folder in ('src/multimodal_jev','tools/jev'):
        for path in sorted((source/folder).glob('*.py')):
            dest=package/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in TOOLS:
        path=Path('tools/reports')/name;dest=package/path;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/path,dest)
    shutil.copytree(formal/'initial/policy',package/'policy')
    shutil.copytree(formal/'initial/adapter',package/'adapter')
    shutil.copy2(inputs/'inputs.pt',package/'inputs.pt');shutil.copy2(inputs/'MANIFEST.json',package/'INPUTS.json')
    policy=dict(policy,snapshot=ROOT+'/package/policy',adapter=ROOT+'/package/adapter',
                physical_devices=[7],candidate_runtime=PROFILE,max_batch_queries=4,max_batch_tokens=16384)
    policy.pop('process_device_scope',None)
    config=dict(policy=policy,source_root=ROOT+'/package',inputs=ROOT+'/package/inputs.pt',
                input_sha256=manifest['input_sha256'],audio_token_id=151676,
                pins={ROOT+'/package/'+str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()})
    (package/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    files={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()}
    (package/'FILES.json').write_text(json.dumps(files,indent=2)+'\n')
    receipt=dict(status='sealed_not_launched',image=IMAGE,remote_root=ROOT,source_files=len(files),
        files_sha256=sha(package/'FILES.json'),argv=command(policy['model']),
        predecessor='existing simple training/evaluation queue and exclusive device7 lease',
        prerequisites=['fixed image present','actual device7 idle','exclusive lease',
                       'remote payload hashes match','bounded source/model/policy identities verified'],
        training_dispatch_allowed=False,scope='serial and batch4 fresh processes, 128 TRAIN rows, four passes; no optimizer')
    (output/'PREPARED.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','formal','inputs','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.formal,a.inputs,a.output),indent=2))
