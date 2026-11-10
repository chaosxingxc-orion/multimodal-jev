"""Sequential CPU stage execution across two immutable Docker images."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from .baseline import atomic_json


class ObservationUnavailable(RuntimeError):
    pass


def call(host,argv):
    if host!='local':
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',host):
            raise ValueError('explicit SSH host alias required')
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(argv)]
    return subprocess.run(argv,capture_output=True,text=True,timeout=30)


def verify_container(item,name,image):
    if item.get('Name')!='/'+name or item.get('Image')!=image:
        raise ValueError('container name/image identity changed')


class ContainerExecutor:
    def __init__(self,root,source,runtime):
        self.root=Path(root).resolve();self.source=Path(source).resolve();self.runtime=runtime
        for role in ('rollout','actor','export'):
            spec=runtime[role]
            if not re.fullmatch(r'sha256:[a-f0-9]{64}',spec['image']):
                raise ValueError('immutable image ID required')
            if type(spec['cpus']) is not int or spec['cpus']<1:
                raise ValueError('positive CPU allocation required')
            if type(spec['memory_gib']) is not int or spec['memory_gib']<1:
                raise ValueError('positive memory allocation required')
            if spec['host']!='local' and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',spec['host']):
                raise ValueError('explicit host alias required')

    def inspect(self,host,handle):
        try:
            result=call(host,['docker','inspect',handle])
            if result.returncode:raise ObservationUnavailable(result.stderr[:400])
            items=json.loads(result.stdout)
            if len(items)!=1:raise ObservationUnavailable('unexpected inspect result')
            return items[0]
        except (subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
            raise ObservationUnavailable(str(exc)) from exc

    def execute(self,role,output,python_args,extra_mounts=()):
        output=Path(output).resolve()
        output.relative_to(self.root)
        spec=self.runtime[role];host=spec['host'];image=spec['image']
        name='multimodal-jev-stage-'+hashlib.sha256(str(output).encode()).hexdigest()[:24]
        receipt=output.parent/(output.name+'-container.json')
        intent=output.parent/(output.name+'-launch-intent.json')
        argv=['docker','run','-d','--network=none','--name',name,
              '--cpus='+str(spec['cpus']),'--memory='+str(spec['memory_gib'])+'g',
              '-e','PYTHONPATH='+str(self.source/'src'),'-e','PYTHONDONTWRITEBYTECODE=1',
              '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1']
        if host!='local':argv.insert(3,'--pull=never')
        if role=='rollout':
            cpuset=spec.get('cpuset')
            if not isinstance(cpuset,str) or not re.fullmatch(r'[0-9,-]+',cpuset):
                raise ValueError('explicit rollout CPU set required')
            kv_cache_gib=spec.get('kv_cache_gib',1)
            if type(kv_cache_gib) is not int or not 1<=kv_cache_gib<spec['memory_gib']:
                raise ValueError('positive integer KV cache smaller than container memory required')
            argv+=['--cpuset-cpus='+cpuset,'--shm-size=2g','-e','OMP_NUM_THREADS='+str(spec['cpus']),
                   '-e','VLLM_CPU_OMP_THREADS_BIND='+cpuset,'-e','VLLM_CPU_KVCACHE_SPACE='+str(kv_cache_gib)]
        else:
            argv+=['-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
        shared=Path(self.runtime['shared_root']).resolve()
        self.root.relative_to(shared);self.source.relative_to(self.root)
        for mount in [str(shared)+':'+str(shared)+':ro',str(self.root)+':'+str(self.root)+':rw',
                      str(self.source)+':'+str(self.source)+':ro',*extra_mounts]:
            argv+=['-v',mount]
        # Ascend image entrypoint sets toolkit/driver library paths even for CPU.
        argv+=(['--entrypoint','python3',image] if role=='rollout' else [image,'python3'])
        argv+=python_args
        command={'host':host,'argv':argv,'name':name,'image':image}
        if intent.exists():
            if json.loads(intent.read_text())!=command:raise ValueError('container command changed')
        else:
            if output.exists() or receipt.exists():raise RuntimeError('untracked stage requires inspection')
            image_check=call(host,['docker','image','inspect',image])
            if image_check.returncode:raise RuntimeError('required pinned image unavailable')
            # A persisted intent is never submitted twice. Timeout recovery observes
            # the same named container; no heartbeat or missing receipt means death.
            atomic_json(intent,command)
            try:
                result=call(host,argv)
                atomic_json(output.parent/(output.name+'-launch-observation.json'),
                            {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                if result.returncode==0 and re.fullmatch(r'[a-f0-9]{64}',result.stdout.strip()):
                    atomic_json(receipt,{'container_id':result.stdout.strip(),**command})
                if result.returncode==125 and 'unknown flag:' in result.stderr:
                    raise RuntimeError('Docker rejected command before container creation: '+result.stderr[:300])
                if result.returncode==125 and 'no space left on device' in result.stderr:
                    raise RuntimeError('Docker allocation failed: '+result.stderr[:400]+'; inspect named container before retry')
            except subprocess.TimeoutExpired as exc:
                atomic_json(output.parent/(output.name+'-launch-observation.json'),{'timeout':str(exc)})
        cid=json.loads(receipt.read_text())['container_id'] if receipt.exists() else None
        while True:
            try:
                item=self.inspect(host,cid or name)
            except ObservationUnavailable as exc:
                atomic_json(self.root/'container-progress.json',{'stage':str(output),
                    'phase':'observation_retry','error':str(exc),'heartbeat_unix':time.time()})
                time.sleep(30);continue
            verify_container(item,name,image)
            if cid is not None and item['Id']!=cid:raise ValueError('container ID changed')
            if cid is None:
                cid=item['Id'];atomic_json(receipt,{'container_id':cid,**command})
            state=item['State']
            atomic_json(self.root/'container-progress.json',{'stage':str(output),'container_id':cid,
                'state':state,'heartbeat_unix':time.time()})
            if state['Status']=='exited':
                atomic_json(output.parent/(output.name+'-docker-terminal.json'),{'container_id':cid,'state':state})
                if state['ExitCode']!=0 or state.get('OOMKilled'):raise RuntimeError('container stage failed')
                return
            if state['Status'] in {'dead','removing'}:raise RuntimeError('container terminal failure')
            time.sleep(30)
