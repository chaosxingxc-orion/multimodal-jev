"""Configured CPU-only verification of an existing selection cache on its host."""
import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths

IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def audit_receipts(binding_root,collection,binding_sha256):
    import torch
    from multimodal_jev.span_request_identity import tensor_identity
    binding_root=Path(binding_root);collection=Path(collection)
    def read(root,name):return json.loads((root/name).read_text())
    if file_hash(binding_root/'binding.json')!=binding_sha256:raise ValueError('binding changed')
    binding=read(binding_root,'binding.json')
    for name,h in binding['artifacts'].items():
        if name not in ('population.json','candidate-ids.json','cache-receipts.json') or file_hash(binding_root/name)!=h:
            raise ValueError('original input evidence changed')
    population=read(binding_root,'population.json');prior=read(binding_root,'cache-receipts.json')
    candidates=read(binding_root,'candidate-ids.json');receipts=read(collection,'input-receipts.json')
    result=read(collection,'result.json')
    if (read(collection,'terminal.json')!={'exit_code':0} or len(receipts)!=9226 or len(population)!=9226
            or result['recordings']!=9226 or result['decisions']!=60
            or result['binding_sha256']!=binding_sha256
            or result['input_receipts_sha256']!=file_hash(collection/'input-receipts.json')):
        raise ValueError('complete successful full-cache result required')
    for asset,old,new in zip(population,prior,receipts,strict=True):
        if (any(new[k]!=v or old[k]!=v for k,v in asset.items()) or new['candidate_ids']!=candidates
                or new['inputs']!={k:v for k,v in old['tensors'].items() if k!='positions'}
                or len(new['positions'])!=60
                or tensor_identity(torch.tensor([new['positions']],dtype=torch.long))!=old['tensors']['positions']):
            raise ValueError('exported receipts differ from original independent tensor evidence')
    return dict(status='full_selection_cache_receipts_independently_bound',recordings=9226,decisions=60,
        binding_sha256=binding_sha256,input_receipts_sha256=result['input_receipts_sha256'],
        scope='Actual CPU cache verification and independent original tensor-hash comparison; no NPU or task score')


def command(c):
    root=c['remote_root']
    for key in ('host','tag'):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',c[key]):raise ValueError('safe host/tag required')
    for p in (root,c['cache_root'],c['driver_root']):
        if not Path(p).is_absolute() or any(x in p for x in (',',':','\n')) or '..' in Path(p).parts:
            raise ValueError('absolute safe mount paths required')
    if (len(Path(root).parts)<4 or Path(c['cache_root']).is_relative_to(root)
            or Path(root).is_relative_to(c['cache_root'])):raise ValueError('isolated output/cache required')
    argv=['docker','create','--name',c['tag'],'--pull=never','--network=none','--cpus=2','--memory=8g',
          '--log-driver=json-file','--log-opt','max-size=20m','--log-opt','max-file=2']
    for source,target,ro in [(root+'/package',root+'/package',True),(root+'/output',root+'/output',False),
        (root+'/home','/root',False),(root+'/tmp','/jev-tmp',False),(c['cache_root'],c['cache_root'],True),
        (c['driver_root'],'/usr/local/Ascend/driver',True)]:
        argv+=['--mount',f'type=bind,src={source},dst={target}'+(',readonly' if ro else '')]
    env=dict(PYTHONPATH=root+'/package/src:'+root+'/package',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for name in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
        env[name]='/jev-tmp'
    for key,value in env.items():argv+=['-e',key+'='+value]
    return argv+['--entrypoint','python',IMAGE,root+'/package/run.py']


def operate(c,provenance,action):
    argv=command(c);base=Path(c['operations_root']);check_paths({'operations':base})
    def ssh(args):
        return subprocess.run(['ssh','-o','ConnectTimeout=10',c['host'],shlex.join(args)],
            capture_output=True,text=True,check=True,timeout=55).stdout
    root=c['remote_root']
    if action=='run':
        base.mkdir(parents=True,exist_ok=False);package=base/'package';package.mkdir()
        binding=Path(c['binding_root'])
        if file_hash(binding/'binding.json')!=c['binding_sha256']:raise ValueError('existing binding identity differs')
        shutil.copytree(binding,package/'binding')
        source=Path(c['project'])
        shutil.copytree(source/'src/multimodal_jev',package/'src/multimodal_jev',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        script='tools/reports/verify_selection_macro_cache.py';(package/script).parent.mkdir(parents=True)
        shutil.copy2(source/script,package/script)
        config=dict(binding_root=root+'/package/binding',cache_root=c['cache_root'],output=root+'/output/run',binding_sha256=c['binding_sha256'])
        (package/'config.json').write_text(json.dumps(config))
        (package/'run.py').write_text("""import hashlib,json,runpy,sys
from pathlib import Path
r=Path(__file__).parent
files=json.loads((r/'FILES.json').read_text())
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in files.items())
sys.argv=[str(r/'tools/reports/verify_selection_macro_cache.py'),'--config',str(r/'config.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
""")
        files={str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()}
        (package/'FILES.json').write_text(json.dumps(files,sort_keys=True))
        storage=(source/'src/multimodal_jev/local_runtime_storage.py').read_text()
        ssh(['python3','-c',storage+'\n'+f"check_paths({{'parent':{str(Path(root).parent)!r}}})\nr=Path({root!r});r.mkdir(exist_ok=False)\n[(r/n).mkdir() for n in ('package','output','home','tmp')]\nassert Path({c['cache_root']!r}).is_dir()"])
        archive=base/'package.tar'
        subprocess.run(['tar','-C',str(package),'-cf',str(archive),'.'],check=True)
        with archive.open('rb') as f:subprocess.run(['ssh',c['host'],shlex.join(['tar','-C',root+'/package','-xf','-'])],stdin=f,check=True,timeout=55)
        cid=ssh(argv).strip()
        launch=dict(container_id=cid,argv=argv,image=IMAGE,config_provenance=provenance,
                    files=files,source_manifest_sha256=file_hash(package/'FILES.json'))
        commit_json(base/'launch.json',launch)
        ssh(['docker','start',cid]);obs=json.loads(ssh(['docker','inspect',cid]))[0]
        commit_json(base/'start.json',obs)
        return dict(container_id=cid,state=obs['State'])
    launch=json.loads((base/'launch.json').read_text());cid=launch['container_id']
    if launch['argv']!=argv:raise ValueError('launch config changed')
    obs=json.loads(ssh(['docker','inspect',cid]))[0];state=obs['State']
    if (obs['Id']!=cid or obs['Image']!=IMAGE or obs['HostConfig']['Devices']
            or obs['HostConfig']['NetworkMode']!='none' or obs['Config']['Entrypoint']!=['python']
            or obs['Config']['Cmd']!=argv[argv.index(IMAGE)+1:]):raise ValueError('CPU runtime differs')
    expected_mounts={}
    for i,arg in enumerate(argv):
        if arg=='--mount':
            parts=argv[i+1].split(',');spec=dict(p.split('=',1) for p in parts if '=' in p)
            expected_mounts[spec['dst']]=(spec['src'],'readonly' not in parts)
    actual_mounts={m['Destination']:(m['Source'],m['RW']) for m in obs['Mounts'] if m['Type']=='bind'}
    if actual_mounts!=expected_mounts:raise ValueError('CPU input/output mount scope differs')
    if state['Status']!='exited':raise ValueError('retain exact live container handle')
    if state['ExitCode'] or state['OOMKilled']:raise ValueError('verification failed; preserve container logs')
    code="""from pathlib import Path
import hashlib,json
r=Path(ROOT);files=json.loads((r/'package/FILES.json').read_text())
assert hashlib.sha256((r/'package/FILES.json').read_bytes()).hexdigest()==MANIFEST
assert all(hashlib.sha256((r/'package'/p).read_bytes()).hexdigest()==h for p,h in files.items())
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'output/run').iterdir() if p.is_file()}))
""".replace('ROOT',repr(root)).replace('MANIFEST',repr(launch['source_manifest_sha256']))
    hashes=json.loads(ssh(['python3','-c',code]));out=base/'collected';out.mkdir(exist_ok=False)
    for name,h in hashes.items():
        if Path(name).name!=name:raise ValueError('invalid collection name')
        subprocess.run(['scp','-q',c['host']+':'+root+'/output/run/'+name,str(out/name)],check=True,timeout=55)
        if file_hash(out/name)!=h:raise ValueError('collection bytes differ')
    commit_json(base/'transport.json',dict(container_id=cid,inspection=obs,files=hashes,source_manifest_sha256=launch['source_manifest_sha256']))
    audit=audit_receipts(c['binding_root'],out,c['binding_sha256'])
    commit_json(base/'audit.json',dict(audit,container_id=cid,image=IMAGE,state=state,files=hashes,collection=str(out)))
    return dict(container_id=cid,state=state,files=hashes,collection=str(out),audit=audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['run','collect'],required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    print(json.dumps(operate(config,provenance,args.action),indent=2))
