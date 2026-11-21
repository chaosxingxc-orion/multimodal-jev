"""Content-bound objective replay on one leased NPU, without loading a model.

Only recorded decision tensors and the exact objective are evaluated. This
does not replace the caller's source, transport, optimizer or model audit.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.audit_objective_traces import checked
from tools.reports.launch_context_batch_probe_candidate import wait_exact

IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def evidence(c):
    files={c['objective_plugin']['path']:c['objective_plugin']['sha256']}
    for update in c['updates']:
        for binding in [update['groups'],*update['traces']]:
            files[binding['path']]=binding['sha256']
    for name,digest in files.items():checked(dict(path=name,sha256=digest))
    return files


def remote_configuration(c,remote):
    """Explicit role mapping keeps input identities independent of path changes."""
    c=json.loads(json.dumps(c));files={}
    def bind(binding,name):
        files[name]=dict(path=binding['path'],sha256=binding['sha256'])
        binding['path']=remote+'/package/'+name
    bind(c['objective_plugin'],'evidence/objective.py')
    for u in c['updates']:
        bind(u['groups'],f'evidence/groups-{u["update"]:04d}.pt')
        for t in u['traces']:
            bind(t,f'evidence/trace-{u["update"]:04d}-rank-{t["rank"]}.json')
    c.update(replay_device='npu:0',output=remote+'/run/report.json')
    return c,files


def accepted(report,c,remote_c):
    if (report.get('status')!='finite_objective_traces_replayed' or report.get('replay_device')!='npu:0'
            or report.get('evidence')!=remote_c['updates']
            or report.get('objective')!=dict(api='jev-stage2-objective-v1',
                **{k:v for k,v in c['objective_plugin'].items() if k!='path'})
            or len(report.get('updates',[]))!=len(c['updates'])):
        raise ValueError('complete same-backend objective replay required')
    for result,u in zip(report['updates'],c['updates'],strict=True):
        if (result['update']!=u['update'] or result['ranks']!=c['world_size']
                or result['queries']!=c['world_size']*c['local_batch']
                or result['microbatches']!=c['world_size']*c['local_batch']//c['microbatch']):
            raise ValueError('backend replay population differs')
    return dict(report,evidence=c['updates'])


class TraceReplayEngine:
    def __init__(self,config):
        self.c=config;self.cache={}
        if (config['image']!=IMAGE or type(config['device'])is not int or config['device']not in (4,5,6,7)
                or not re.fullmatch(r'(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9_.-]*',config['host'])):
            raise ValueError('fixed-image authorized-device backend replay required')
        self.source=Path(config['source']);self.manifest=json.loads((self.source/'FILES.json').read_text())
        self.sources={n:h for n,h in self.manifest.items() if n.endswith('.py') and
                      (n.startswith('src/multimodal_jev/') or n.startswith('tools/reports/'))}
        self.verify_source()

    def verify_source(self):
        for n,h in self.sources.items():
            if file_hash(self.source/n)!=h:raise ValueError('backend auditor source changed: '+n)

    def audit(self,c,provenance):
        self.verify_source();inputs=evidence(c)
        key=hashlib.sha256(json.dumps(dict(config=c,sources=self.sources),sort_keys=True).encode()).hexdigest()
        if key not in self.cache:self.cache[key]=self.execute(c,key,inputs)
        entry=self.cache[key]
        if file_hash(entry['path'])!=entry['sha256']:raise ValueError('backend receipt changed')
        return dict(json.loads(Path(entry['path']).read_text()),provenance=provenance)

    def execute(self,c,key,inputs):
        cfg=self.c;root=Path(cfg['output'])/key[:16];remote=cfg['remote_root']+'/'+key[:16]
        check_paths({'backend_trace':root});root.mkdir(parents=True,exist_ok=False)
        package=root/'package';package.mkdir();remote_c,mapping=remote_configuration(c,remote)
        for name,h in self.sources.items():
            target=package/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((self.source/name).read_bytes())
        for name,b in mapping.items():
            target=package/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(checked(b).read_bytes())
        commit_json(package/'config.json',remote_c)
        files={str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()}
        commit_json(package/'FILES.json',files)
        archive=root/'package.tar';
        with tarfile.open(archive,'x') as tar:tar.add(package,arcname='package')
        def remote_run(args,bounded=True):
            return subprocess.run(['ssh','-o','ConnectTimeout=10',cfg['host'],shlex.join(args)],
                                  capture_output=True,text=True,timeout=55 if bounded else None)
        def run(args):
            r=remote_run(args)
            if r.returncode:raise RuntimeError(r.stderr)
            return r.stdout
        with Path(cfg['lease']).open('r') as lease:
            wait = cfg.get('wait_for_lease', False)
            if type(wait) is not bool:
                raise ValueError('explicit Boolean lease wait required')
            fcntl.flock(lease,fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            if f'No running processes found in NPU {cfg["device"]}' not in run(['npu-smi','info']):
                raise ValueError('backend replay device occupied')
            storage=(self.source/'src/multimodal_jev/local_runtime_storage.py').read_text()
            run(['python3','-c',storage+'\ncheck_paths({"output":'+repr(str(Path(remote).parent))+'})'])
            run(['mkdir','-p',str(Path(remote).parent)]);run(['mkdir',remote])
            subprocess.run(['scp','-q',str(archive),cfg['host']+':'+remote+'/package.tar'],check=True,capture_output=True,timeout=55)
            run(['tar','-xf',remote+'/package.tar','-C',remote]);run(['mkdir',remote+'/run',remote+'/cache',remote+'/home'])
            argv=['docker','create','--pull=never','--network=none','--name',cfg['tag']+'-'+key[:12],
                  '--cpus=4','--memory=8g','--log-driver=json-file','--log-opt','max-size=10m','--log-opt','max-file=2']
            for d in (f'davinci{cfg["device"]}','davinci_manager','devmm_svm','hisi_hdc'):
                argv+=['--device','/dev/'+d+':/dev/'+d]
            for src,dst,ro in [(remote+'/package',remote+'/package',True),(remote+'/run',remote+'/run',False),
                               (remote+'/cache','/jev-tmp',False),(remote+'/home','/root',False),
                               (cfg['driver_root'],'/usr/local/Ascend/driver',True)]:
                argv+=['-v',src+':'+dst+(':ro' if ro else '')]
            env=dict(PYTHONPATH=remote+'/package/src:'+remote+'/package',PYTHONDONTWRITEBYTECODE='1',
                     PYTHONSAFEPATH='1',OMP_NUM_THREADS='1',ASCEND_RT_VISIBLE_DEVICES='0',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR='/jev-tmp',ASCEND_WORK_PATH='/jev-tmp',
                     ASCEND_CACHE_PATH='/jev-tmp')
            for name,value in env.items():argv+=['-e',name+'='+value]
            argv+=['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python','-B','-m',
                   'tools.reports.context_backend_trace_replay','--worker-root',remote]
            waiter=wait_exact
            if 'lease_deadline_unix' in cfg:
                from functools import partial
                import time
                from tools.reports.context_infonce_formal import lease_wait
                deadline,grace=cfg['lease_deadline_unix'],cfg['lease_stop_grace_seconds']
                if (type(deadline) not in (int,float) or type(grace) is not int or grace<1
                        or deadline<=time.time()+grace+30):
                    raise ValueError('allocation expired before replay container creation')
                waiter=partial(lease_wait,deadline=deadline,stop_grace=grace)
            cid=run(argv).strip()
            if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('exact replay container required')
            commit_json(root/'launch.json',dict(container_id=cid,argv=argv,files=files,config=c,remote_config=remote_c))
            run(['docker','start',cid]);code=waiter(remote_run,cid)
            inspection=json.loads(run(['docker','inspect',cid]))[0];commit_json(root/'inspect.json',inspection)
            log=remote_run(['docker','logs',cid]);(root/'container.log').write_text(log.stdout+log.stderr)
            if (code or inspection['Id']!=cid or inspection['Image']!=IMAGE or inspection['State']['Running']
                    or inspection['State']['OOMKilled'] or inspection['State']['ExitCode']):
                raise ValueError('same-backend replay failed; see '+str(root/'container.log'))
            for name in ('report.json','terminal.json'):
                subprocess.run(['scp','-q',cfg['host']+':'+remote+'/run/'+name,str(root/name)],check=True,capture_output=True,timeout=55)
        terminal=json.loads((root/'terminal.json').read_text())
        if terminal!=dict(exit_code=0,files=files,config=remote_c,model_calls=0,npu_executed=True):
            raise ValueError('actual replay inputs or executable identity differs')
        report=accepted(json.loads((root/'report.json').read_text()),c,remote_c)
        evidence(c);self.verify_source()
        commit_json(root/'accepted.json',report)
        return dict(path=str(root/'accepted.json'),sha256=file_hash(root/'accepted.json'))


def worker(root):
    import torch
    from tools.reports.audit_objective_traces import audit
    root=Path(root);p=root/'package';files=json.loads((p/'FILES.json').read_text())
    for n,h in files.items():
        if file_hash(p/n)!=h:raise ValueError('replay payload changed')
    torch.set_num_threads(1);c=json.loads((p/'config.json').read_text())
    report=audit(c,{'source':'same_backend_independent_replay'})
    commit_json(root/'run/report.json',report)
    commit_json(root/'run/terminal.json',dict(exit_code=0,files=files,config=c,model_calls=0,npu_executed=True))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--worker-root',required=True);a=p.parse_args();worker(a.worker_root)
