"""Finite isolated two-service probe with exact handles and no implicit retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_context_batch_probe_candidate import IMAGE
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.launch_context_batch_probe_candidate import write_new, wait_exact


def validate(c):
    root=Path(c['prepared']);package=root/'package'
    receipt=json.loads((root/'PREPARED.json').read_text())
    if (receipt['status']!='macro_probe_prepared_only' or receipt['image']!=IMAGE
            or receipt['files_sha256']!=c['manifest_sha256'] or verify(package)!=c['manifest_sha256']
            or receipt['remote_root']!=c['remote_root'] or receipt['training_dispatch_allowed'] is not False
            or receipt['device_scope']!=[[6],[7]]):
        raise ValueError('authenticated finite macro payload required')
    for key in ('host','tag'):
        if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c[key]):raise ValueError('safe host/tag required')
    remote=Path(c['remote_root'])
    if (not remote.is_absolute() or '..' in remote.parts or str(remote) in ('/','/data','/mnt')
            or not Path(receipt['ipc_root']).is_relative_to(remote)):
        raise ValueError('isolated local root and contained IPC required')
    measurement=json.loads((package/'measurement.json').read_text())
    client=measurement['macro_client']
    if (client['runtime_identity']!=receipt['runtime_identity'] or client['max_queries']!=128
            or client['rollout_queries']!=32 or measurement['output']!=str(remote/'measurement')
            or client['journal_root']!=str(remote/'journal')):
        raise ValueError('macro128 output/runtime contract differs')
    for i in range(2):
        service=json.loads((package/f'service-{i}.json').read_text())
        endpoint=client['replicas'][i]
        if (service['policy']['physical_devices']!=[6+i]
                or service['policy']['model']!=c['model']
                or service['dual_runtime_identity']!=receipt['runtime_identity']
                or service['source_root']!=str(remote/'package')
                or service['socket_path']!=endpoint['socket'] or service['exchange']!=endpoint['exchange']
                or service['policy'].get('process_device_scope') is not None
                or not all(Path(v).is_relative_to(receipt['ipc_root']) for v in endpoint.values())):
            raise ValueError('isolated service/runtime/IPC differs')
    return receipt


def command(c, role):
    if role not in ('service-0','service-1','client','preflight'):raise ValueError('explicit probe role required')
    root=c['remote_root'];package=root+'/package'
    argv=['docker','create','--name',c['tag']+'-'+role,'--pull=never','--network=none',
          '--cpus=8','--memory=32g','--shm-size=2g']
    if role.startswith('service-'):
        for name in ('davinci'+str(6+int(role[-1])),'davinci_manager','devmm_svm','hisi_hdc'):
            argv+=['--device','/dev/'+name+':/dev/'+name]
    for mount in (root+':'+root,package+':'+package+':ro',root+'/'+role+'-home:/root',
            root+'/'+role+'-cache:/jev-tmp',c['model']+':'+c['model']+':ro',
            c['driver_root']+':/usr/local/Ascend/driver:ro'):
        argv+=['-v',mount]
    env=dict(PYTHONPATH=package+':'+package+'/src:'+package+'/tools/jev',PYTHONDONTWRITEBYTECODE='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',TMPDIR='/jev-tmp',VLLM_RPC_BASE_PATH='/jev-tmp')
    for key in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[key]='/jev-tmp'
    for key,value in env.items():argv+=['-e',key+'='+value]
    argv+=['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python','-B']
    if role=='preflight':
        code="import json; from tools.reports.context_macro_measurement import run; from tools.reports.context_dual_runtime_candidate import verify_runtime; from tools.reports.run_context_batch_probe_candidate import verify; verify("+repr(package)+"); [verify_runtime(json.load(open("+repr(package)+"+'/service-'+str(i)+'.json'))) for i in range(2)]; print('sealed client and runtime imports passed')"
        argv+=['-c',code]
    elif role=='client':
        code="import json; from tools.reports.context_macro_measurement import run; run(json.load(open("+repr(package+'/measurement.json')+")), {'source':'sealed_macro_payload'})"
        argv+=['-c',code]
    else:
        argv += ['-m','tools.reports.context_dual_runtime_candidate','--config',package+'/'+role+'.json',
                 '--output',root+'/'+role]
    return argv


def inspect_runtime(observed, handle, *, terminal=False):
    argv=handle['argv']
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    cfg=observed['Config'];hc=observed['HostConfig'];state=observed['State']
    if (observed['Id']!=handle['container_id'] or observed['Image']!=IMAGE
            or hc['Privileged'] or hc['NetworkMode']!='none'
            or sorted(hc['Binds'])!=sorted(values('-v')) or cfg['Entrypoint']!=values('--entrypoint')
            or cfg['Cmd']!=argv[argv.index(IMAGE)+1:]):
        raise ValueError('actual macro container scope differs')
    expected=sorted((*v.split(':'),'rwm') for v in values('--device'))
    actual=sorted((d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in hc['Devices'])
    env=dict(v.split('=',1) for v in cfg['Env'])
    if actual!=expected or any(env.get(k)!=v for k,v in (e.split('=',1) for e in values('-e'))):
        raise ValueError('actual devices or environment differs')
    if terminal and (state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']):
        raise ValueError('successful exact terminal required')


def run(c,provenance,*,validator=validate,command_builder=command,required_devices=(6,7)):
    receipt=validator(c)
    if c['finite_inference_dispatch_allowed'] is not True:raise ValueError('finite probe not admitted')
    base=Path(c['operations_root']);check_paths({'operations':base});base.mkdir(parents=True,exist_ok=True)
    with (base/'owner.lock').open('a') as owner, Path(c['lease']).open('r') as lease:
        for lock in (owner,lease):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'attempt.json').exists():raise ValueError('existing attempt; reconcile exact handles, never restart')
        def remote(args,bounded=True):
            return subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
                text=True,capture_output=True,timeout=55 if bounded else None)
        def checked(args):
            r=remote(args)
            if r.returncode:raise RuntimeError(r.stderr)
            return r.stdout
        if json.loads(checked(['docker','image','inspect',IMAGE]))[0]['Id']!=IMAGE:raise ValueError('image differs')
        observation=checked(['npu-smi','info'])
        if any(f'No running processes found in NPU {i}' not in observation for i in required_devices):
            raise ValueError('all requested physical devices must be idle')
        root=c['remote_root'];package=root+'/package'
        storage=(Path(c['prepared'])/'package/src/multimodal_jev/local_runtime_storage.py').read_text()
        checked(['python3','-B','-c',storage+'\ncheck_paths({"root":'+repr(str(Path(root).parent))+'})\nassert not Path('+repr(root)+').exists()'])
        write_new(base/'attempt.json',dict(config=c,provenance=provenance,receipt=receipt))
        handles={};failure=None
        try:
            archive=base/'package.tar.gz'
            with tarfile.open(archive,'x:gz') as tar:tar.add(Path(c['prepared'])/'package',arcname='package')
            checked(['mkdir','-p',str(Path(root).parent)])
            checked(['mkdir',root])
            checked(['mkdir',receipt['ipc_root'],root+'/journal',*[root+'/'+r+s for r in ('service-0','service-1','client','preflight') for s in ('-cache','-home')]])
            subprocess.run(['scp',str(archive),c['host']+':'+root+'/package.tar.gz'],check=True,capture_output=True,timeout=55)
            checked(['tar','-xzf',root+'/package.tar.gz','-C',root])
            code='import sys;sys.path.insert(0,'+repr(package)+');from tools.reports.run_context_batch_probe_candidate import verify;assert verify('+repr(package)+')=='+repr(receipt['files_sha256'])
            checked(['python3','-B','-c',code])
            def create(role):
                argv=command_builder(c,role)
                # Deterministic create intent survives an uncertain SSH response.
                write_new(base/(role+'-intent.json'),dict(argv=argv,name=c['tag']+'-'+role))
                cid=checked(argv).strip()
                if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('invalid exact container id')
                handle=dict(container_id=cid,argv=argv,role=role);handles[role]=handle
                write_new(base/(role+'-launch.json'),handle)
                obs=json.loads(checked(['docker','inspect',cid]))[0];inspect_runtime(obs,handle)
                checked(['docker','start',cid]);return handle
            create('preflight')
            preflight_code=wait_exact(remote,handles['preflight']['container_id'])
            write_new(base/'preflight-wait.json',dict(exit_code=preflight_code))
            if preflight_code:raise ValueError('fixed-image imports failed before NPU creation')
            for role in ('service-0','service-1'):create(role)
            deadline=time.monotonic()+900
            while True:
                ready=[]
                for role in ('service-0','service-1'):
                    obs=json.loads(checked(['docker','inspect',handles[role]['container_id']]))[0]
                    if not obs['State']['Running']:raise ValueError(role+' exited before connection')
                    code='from pathlib import Path;p=Path('+repr(root+'/'+role+'/ready.json')+');print(p.read_text() if p.exists() else "null")'
                    ready.append(json.loads(checked(['python3','-B','-c',code])))
                if all(ready):
                    for i,r in enumerate(ready):
                        if r['physical_devices']!=[6+i]:raise ValueError('service ready device differs')
                    write_new(base/'services-ready.json',ready);break
                if time.monotonic()>deadline:raise TimeoutError('finite startup deadline')
                time.sleep(5)  # Fixed readiness script, not periodic model analysis.
            create('client')
            # A failed client may never connect/close the listeners. Detect it
            # before waiting for services so exact-container cleanup can run.
            codes={'client':wait_exact(remote,handles['client']['container_id'])}
            write_new(base/'client-wait.json',codes)
            if codes['client']:raise ValueError('macro client failed')
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures={r:executor.submit(wait_exact,remote,handles[r]['container_id'])
                         for r in ('service-0','service-1')}
                codes.update({r:f.result() for r,f in futures.items()})
            write_new(base/'wait.json',codes)
            if any(codes.values()):raise ValueError('macro process failed: '+str(codes))
        except BaseException as exc:
            failure=repr(exc)
        finally:
            # Only stop exact containers this attempt created, after scope check.
            for role,handle in handles.items():
                try:
                    obs=json.loads(checked(['docker','inspect',handle['container_id']]))[0]
                    inspect_runtime(obs,handle)
                    if failure and obs['State']['Running']:
                        checked(['docker','stop','--time','30',handle['container_id']])
                        obs=json.loads(checked(['docker','inspect',handle['container_id']]))[0]
                    write_new(base/(role+'-inspect.json'),obs)
                    logs=remote(['docker','logs',handle['container_id']])
                    (base/(role+'.log')).write_text(logs.stdout+logs.stderr)
                    if not failure:inspect_runtime(obs,handle,terminal=True)
                except Exception as exc:
                    failure=failure or repr(exc)
            write_new(base/'terminal.json',dict(exit_code=int(failure is not None),error=failure,
                output_remote=root,collection_and_independent_audit_required=True))
        if failure:raise RuntimeError(failure)
        return dict(status='macro_processes_completed_collection_required',remote_root=root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['validate','run'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance) if a.action=='run' else validate(c),indent=2))
