"""Immutable four-device performance candidate, compact collection, no scoring.

All old model/runtime files stay read-only. Only candidate orchestration is new.
Ordinary monitoring remains 900 seconds; docker wait observes exact completion.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze


def run(c,provenance):
    root=Path(c['output']);check_paths({'campaign':root});root.mkdir(exist_ok=False)
    started=time.monotonic();stages={}
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    def remote(args,timeout=55):
        return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
                              check=True,text=True,capture_output=True,timeout=timeout).stdout
    try:
        with Path(c['lease']).open('r') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            usage=remote(['npu-smi','info'])
            if any(f'No running processes found in NPU {d}' not in usage for d in c['devices']):
                raise ValueError('configured evaluation devices occupied')
            snapshot=freeze(c['source'],root/'source',patterns=['tools/reports/evaluation_resident_*.py',
                                'src/multimodal_jev/evaluation_campaign.py'],io_workers=4)
            commit_json(root/'source-receipt.json',snapshot)
            new=c['remote_root'];old=c['frozen_run'];runtime=old+'/source'
            cfg=dict(c['experiment'],devices=c['devices'],output=new+'/run',runtime_cache=new+'/cache',
                     service_configs=[old+f'/service-eval{i}.json' for i in range(len(c['devices']))])
            # Bind actual completed old predictions without copying models/data.
            for point in cfg['points']:
                raw=remote(['sha256sum',point['predictions']['path']]);point['predictions']['sha256']=raw.split()[0]
            commit_json(root/'experiment.json',cfg)
            with tarfile.open(root/'source.tar','x') as tar:tar.add(root/'source',arcname='source')
            storage=(Path(c['source'])/'src/multimodal_jev/local_runtime_storage.py').read_text()
            remote(['python3','-c',storage+'\ncheck_paths({"run":'+repr(new)+'})'])
            remote(['mkdir',new]);remote(['mkdir',new+'/cache',new+'/home'])
            subprocess.run(['scp',str(root/'source.tar'),str(root/'experiment.json'),c['host']+':'+new+'/'],check=True,capture_output=True)
            remote(['tar','-xf',new+'/source.tar','-C',new])
            args=['docker','create','--pull=never','--name',c['tag'],'--network=none','--cpus=24','--memory=128g','--shm-size=8g']
            for d in c['devices']:args+=['--device',f'/dev/davinci{d}']
            for d in ('davinci_manager','devmm_svm','hisi_hdc'):args+=['--device','/dev/'+d]
            for src,dst,ro in [(new,new,False),(new+'/home','/root',False),(new+'/cache','/jev-tmp',False),
                               (new+'/source',new+'/source',True),*[(p,p,True) for p in c['readonly_roots']],
                               (c['driver_root'],'/usr/local/Ascend/driver',True)]:
                args+=['-v',src+':'+dst+(':ro' if ro else '')]
            env=dict(PYTHONPATH=runtime+':'+runtime+'/src:'+runtime+'/tools/jev',PYTHONDONTWRITEBYTECODE='1',
                     VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     TMPDIR='/jev-tmp',HF_HOME='/jev-tmp',XDG_CACHE_HOME='/jev-tmp',
                     TORCHINDUCTOR_CACHE_DIR='/jev-tmp',TRITON_CACHE_DIR='/jev-tmp',
                     ASCEND_WORK_PATH='/jev-tmp',ASCEND_CACHE_PATH='/jev-tmp',VLLM_RPC_BASE_PATH='/jev-tmp')
            for k,v in env.items():args+=['-e',k+'='+v]
            args+=['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',c['image'],'python','-B',
                   new+'/source/tools/reports/evaluation_resident_runner.py','--config',new+'/experiment.json']
            if json.loads(remote(['docker','image','inspect',c['image']]))[0]['Id']!=c['image']:raise ValueError('fixed image missing')
            commit_json(root/'attempt.json',dict(argv=args,source=snapshot))
            cid=remote(args).strip();commit_json(root/'launch.json',dict(container_id=cid,argv=args))
            stages['preparation']=time.monotonic()-started;begin=time.monotonic()
            remote(['docker','start',cid]);code=int(remote(['docker','wait',cid],timeout=c['timeout_seconds']).strip())
            stages['container_wait']=time.monotonic()-begin
            state=json.loads(remote(['docker','inspect',cid]))[0];commit_json(root/'inspect.json',state)
            begin=time.monotonic()
            subprocess.run(['scp','-r',c['host']+':'+new+'/run',str(root/'collected')],check=True,capture_output=True)
            stages['collection']=time.monotonic()-begin
            commit_json(root/'terminal.json',dict(exit_code=code,wall_seconds=time.monotonic()-started,
                stages=stages,formal_admitted=False,status='collected_independent_campaign_audit_required'))
            return dict(exit_code=code,container_id=cid,stages=stages)
    except BaseException as exc:
        commit_json(root/'failure.json',dict(error=repr(exc),automatic_retry=False,preserve_container=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);print(json.dumps(run(c,provenance),indent=2))
