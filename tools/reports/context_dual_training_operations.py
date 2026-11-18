"""Finite DP4 prefix operations using the existing lease and collector.

Validation is local by default. An idle leased resource window replaces the
historical requirement to finish the entire research queue. This tool never
stops an existing job, retries a launch, or admits full training.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_context_dual_throughput_pair import sha,write
from tools.reports.launch_context_batch_probe_candidate import wait_exact
from tools.jev.collect_context_rl_training import collect

IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def path(value):
    if (type(value)!=str or not re.fullmatch(r'/[a-zA-Z0-9_./-]+',value) or
            '..' in Path(value).parts or str(Path(value))!=value or value=='/'):
        raise ValueError('explicit canonical absolute runtime path required')
    return value


def validate(c):
    package=Path(c['package']);m=json.loads((package/'PREPARED.json').read_text())
    if sha(package/'PREPARED.json')!=c['manifest_sha256']:
        raise ValueError('prepared manifest identity differs')
    actual={str(p.relative_to(package)):sha(p) for p in package.rglob('*')
            if p.is_file() and p!=package/'PREPARED.json'}
    if any(p.is_symlink() for p in package.rglob('*')) or actual!=m['files']:
        raise ValueError('complete immutable package required')
    remote=path(m['remote_root']);config=json.loads((package/'config.json').read_text())
    service=json.loads((package/'service-config.json').read_text())
    replay=m['status']=='integrated_replay_prepared_only'
    if (m['status'] not in ('integrated_prepared_only','integrated_replay_prepared_only') or m['image']!=IMAGE or
            config['engineering_prefix'] is not True or config['training_dispatch_allowed'] is not False or
            config['start_update']!=(1 if replay else 0) or config['stop_update']!=2 or
            (not replay and config.get('resume_root')) or
            config['checkpoint_interval']!=1 or config['global_batch']!=128 or config['microbatch']!=8 or
            config['physical_devices']!=[4,5,6,7] or config['algorithm']!='dual_credit_rloo_clipped_v1' or
            config['dual_arm']!='joint' or config['source_root']!=remote+'/source' or
            config['reference_cache']!=remote+'/reference-cache' or
            service['policy']['max_batch_queries'] not in (1,4) or service['policy']['max_batch_tokens']!=16384 or
            service['policy']['physical_devices']!=[7] or service['policy']['process_device_scope']!=dict(
                kind='same_container_runtime_visibility_v1',mounted_physical=[4,5,6,7],visible_ordinal='3')):
        raise ValueError('isolated matched initial DP4 prefix required')
    if (service['pins']!=config['pins'] or
            service['dual_runtime_identity']!=config['dual_runtime_identity'] or
            config['dual_runtime_identity']!=m['dual_runtime_identity'] or
            config['dual_runtime_manifest']!=remote+'/RUNTIME-SOURCES.json' or
            sha(package/'RUNTIME-SOURCES.json')!=m['dual_runtime_identity']['source_manifest_sha256']):
        raise ValueError('common authenticated runtime required')
    runtime=json.loads((package/'RUNTIME-SOURCES.json').read_text())
    if runtime!={k.removeprefix('source/'):v for k,v in actual.items() if k.startswith('source/') and k.endswith('.py')}:
        raise ValueError('runtime source coverage differs')
    for name,digest in config['pins'].items():
        path(name)
        if name.startswith(remote+'/') and actual.get(name[len(remote)+1:])!=digest:
            raise ValueError('internal dependency pin differs')
    if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['host']):raise ValueError('safe host alias required')
    if not re.fullmatch('[a-z0-9][a-z0-9-]*',c['tag']):raise ValueError('safe unique tag required')
    if replay:
        binding=m['replay'];parent=path(binding['parent_remote_root'])
        proof=json.loads((package/'PARENT-AUDIT.json').read_text())
        if (Path(parent).is_relative_to(remote) or Path(remote).is_relative_to(parent) or
                sha(package/'PARENT-AUDIT.json')!=binding['parent_audit_sha256'] or
                proof['passed'] is not True or proof['schema']!='context-dual-prefix-audit-v1' or
                proof['container_id']!=binding['parent_container'] or
                proof['manifest_sha256']!=binding['parent_manifest_sha256'] or
                config['resume_root']!=parent+'/run' or config['initial_adapter']!=parent+'/run/adapter-0001' or
                service['policy']['snapshot']!=parent+'/run/policy-0001' or
                service['policy']['adapter']!=config['initial_adapter'] or
                service['policy']['policy_id']!=binding['initial_behavior_policy_id']):
            raise ValueError('explicit audited first-step replay binding required')
        required=binding['required_files']
        essentials={'run/checkpoint-0001-committed.json','run/policy-0001/MANIFEST.json',
                    *[f'run/rank-{i}/checkpoint-0001.pt' for i in range(4)]}
        if (not essentials<=required.keys() or
                not any(n.startswith('run/adapter-0001/') for n in required) or
                any(Path(n).is_absolute() or '..' in Path(n).parts or
                    proof['files'].get(n)!=h or config['pins'].get(parent+'/'+n)!=h for n,h in required.items())):
            raise ValueError('complete pinned parent checkpoint and adapter required')
    return m,config,service


def command(c,m,config):
    remote=m['remote_root'];shared=path(c['shared_root']);driver=path(c['driver_root'])
    initial=str(Path(path(config['source_policy'])).parent)
    cache=path(config['encoded_audio_cache']);mirror=path(config['audio_mirror_root'])
    mounts=[(remote,remote,False),(remote+'/exchange','/jev-exchange',False),
        (remote+'/home','/root',False),(remote+'/cache','/jev-tmp',False),
        (shared,shared,True),(driver,'/usr/local/Ascend/driver',True),
        (initial,initial,True),(str(Path(cache).parent),str(Path(cache).parent),True),
        (cache,cache,False),(mirror,mirror,True)]
    if config.get('resume_root'):
        parent=path(m['replay']['parent_remote_root'])
        if Path(remote).is_relative_to(parent) or Path(parent).is_relative_to(remote):
            raise ValueError('replay and parent must be disjoint')
        mounts.append((parent,parent,True))
    for name in ('source','run.py','config.json','service-config.json','RUNTIME-SOURCES.json','PREPARED.json'):
        mounts.append((remote+'/'+name,remote+'/'+name,True))
    if config.get('resume_root'):mounts.append((remote+'/PARENT-AUDIT.json',remote+'/PARENT-AUDIT.json',True))
    for name in config['pins']:
        if not any(Path(name).is_relative_to(dst) for _,dst,_ in mounts):
            raise ValueError('dependency outside explicit mounts: '+name)
    # Prevent a writable run root from encompassing shared input mounts.
    if any(Path(src).is_relative_to(remote) or Path(remote).is_relative_to(src)
           for src in (shared,initial,str(Path(cache).parent),mirror,driver)):
        raise ValueError('runtime and reusable inputs must be disjoint')
    argv=['docker','create','--pull=never','--network=none','--name',c['tag'],
        '--cpus=24','--memory=128g','--shm-size=8g','--log-driver=json-file',
        '--log-opt','max-size=20m','--log-opt','max-file=2']
    for name in ('davinci4','davinci5','davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv+=['--device','/dev/'+name+':/dev/'+name]
    for src,dst,ro in mounts:argv+=['-v',src+':'+dst+(':ro' if ro else '')]
    env=dict(PYTHONPATH=remote+'/source:'+remote+'/source/src:'+remote+'/source/tools/jev',
        PYTHONDONTWRITEBYTECODE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',TMPDIR='/jev-tmp')
    for key in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[key]='/jev-tmp'
    for key,value in env.items():argv+=['-e',key+'='+value]
    return argv+['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',remote+'/run.py']


def inference_admission(c):
    # Re-run the independent actual-audio audit, not merely a passed Boolean.
    from tools.reports.context_dual_probe_operations import audit
    probe,_=load_config(c['probe_operations_config'],c['probe_operations_local'])
    result=audit(probe,c['probe_collection'],c['probe_handle'])
    if (result['exact_actual_logits'] is not True or result['exact_probabilities_and_G16'] is not True or
            result['status']!='finite_fixed_policy_inference_audited'):
        raise ValueError('actual audio serial/batch logits acceptance required')
    return result


def launch(c,provenance):
    m,config,_=validate(c);argv=command(c,m,config)
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('finite training dispatch not enabled')
    base=Path(c['operations_root']);base.mkdir(parents=True,exist_ok=True)
    with (base/(c['tag']+'.lock')).open('a') as owner,Path(c['lease']).open('r') as lease:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=base/(c['tag']+'-attempt.json');handle=base/(c['tag']+'-launch.json')
        if attempt.exists() or handle.exists():raise ValueError('existing attempt: reconcile exact handle')
        check_paths({'operations':base})
        accepted=inference_admission(c);validate(c)
        if config.get('resume_root'):
            from tools.reports.audit_context_dual_training import audit
            parent_config,_=load_config(c['parent_audit_config'],c['parent_audit_local'])
            proof=audit(parent_config);binding=m['replay']
            if (proof['container_id']!=binding['parent_container'] or
                    proof['manifest_sha256']!=binding['parent_manifest_sha256'] or
                    any(proof['files'].get(n)!=h for n,h in binding['required_files'].items())):
                raise ValueError('replay parent audit no longer matches')
        if accepted['policy_id']!=config['source_policy_id'] or accepted['image']!=IMAGE:
            raise ValueError('accepted inference policy or image differs')
        def remote(args,bounded=True):
            return subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
                capture_output=True,text=True,timeout=55 if bounded else None)
        def checked(args):
            r=remote(args)
            if r.returncode:raise RuntimeError(r.stderr)
            return r.stdout
        if json.loads(checked(['docker','image','inspect',IMAGE]))[0]['Id']!=IMAGE:
            raise ValueError('fixed image missing')
        usage=checked(['npu-smi','info'])
        if any(f'No running processes found in NPU {i}' not in usage for i in (4,5,6,7)):
            raise ValueError('leased DP4 devices occupied; retain existing work')
        root=m['remote_root'];package=Path(c['package'])
        storage=(package/'source/src/multimodal_jev/local_runtime_storage.py').read_text()
        check=storage+'\ncheck_paths('+repr({'run':root,'cache':config['encoded_audio_cache']})+')\n'
        check+='assert not Path('+repr(root)+').exists()\n'
        external={k:v for k,v in config['pins'].items() if not k.startswith(root+'/')}
        check+='import hashlib\nfor name,digest in '+repr(external)+'.items():\n assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name\n'
        checked(['python3','-c',check])
        archive=base/(c['tag']+'-package.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for p in sorted(package.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(package)),recursive=False)
        write(attempt,dict(config=c,provenance=provenance,argv=argv,archive_sha256=sha(archive),inference_admission=accepted))
        checked(['mkdir','-p',str(Path(root).parent)])
        checked(['mkdir',root])
        for name in ('exchange','cache','home'):checked(['mkdir',root+'/'+name])
        subprocess.run(['scp',str(archive),c['host']+':'+root+'/package.tar.gz'],check=True,capture_output=True,timeout=55)
        checked(['tar','-xzf',root+'/package.tar.gz','-C',root])
        script='from pathlib import Path\nimport hashlib,json\nr=Path('+repr(root)+')\n'
        script+='assert hashlib.sha256((r/"PREPARED.json").read_bytes()).hexdigest()=='+repr(c['manifest_sha256'])+'\n'
        script+='m=json.loads((r/"PREPARED.json").read_text())\nassert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m["files"].items())\n'
        checked(['python3','-c',script])
        cid=checked(argv).strip()
        if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('invalid container handle')
        record=dict(container_id=cid,image=IMAGE,remote_root=root,host=c['host'],argv=argv,
            manifest_sha256=c['manifest_sha256'],config_sha256=sha(package/'config.json'),
            service_config_sha256=sha(package/'service-config.json'))
        write(handle,record)
        try:
            response=remote(['docker','start',cid])
            write(base/(c['tag']+'-start-response.json'),dict(returncode=response.returncode,stdout=response.stdout,stderr=response.stderr))
        except subprocess.TimeoutExpired:
            write(base/(c['tag']+'-start-response.json'),dict(observation_timeout=True))
        code=wait_exact(remote,cid);write(base/(c['tag']+'-wait.json'),dict(container_id=cid,exit_code=code))
        inspected=json.loads(checked(['docker','inspect',cid]))[0]
        write(base/(c['tag']+'-inspect.json'),inspected)
        result=collect(record,base/(c['tag']+'-collected'),host=c['host'])
        write(base/(c['tag']+'-collection.json'),result)
        from tools.reports.context_dual_probe_operations import check_runtime
        check_runtime(dict(argv=argv,image=IMAGE),record,inspected)
        return dict(container_id=cid,exit_code=code,status='collected_independent_training_audit_required')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.launch:result=launch(c,provenance)
    else:
        m,config,_=validate(c);argv=command(c,m,config)
        result=dict(status='validated_local_only',argv=argv,finite_training_dispatch_allowed=c.get('finite_training_dispatch_allowed'))
    print(json.dumps(result,indent=2))
