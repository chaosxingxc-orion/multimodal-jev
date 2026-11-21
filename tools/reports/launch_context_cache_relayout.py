"""Configured immutable CPU relayout job, detached exact Docker handle."""
import argparse,json,shlex,subprocess,tarfile,time
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths

def launch(c,provenance):
    out=Path(c['output']);check_paths({'output':out});out.mkdir(parents=True,exist_ok=False)
    pkg=out/'package';pkg.mkdir();project=Path(c['project'])
    import shutil
    for p in (project/'src/multimodal_jev').glob('*.py'):
        dest=pkg/'src/multimodal_jev'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(project/'tools/jev/relayout_context_learning_cache.py',pkg/'run.py')
    commit_json(pkg/'config.json',c['cache'])
    files={str(p.relative_to(pkg)):file_hash(p) for p in pkg.rglob('*') if p.is_file()}
    commit_json(pkg/'FILES.json',files)
    root=c['remote_root']
    def ssh(args):return subprocess.check_output(['ssh',c['host'],shlex.join(args)],text=True,timeout=55)
    storage=(project/'src/multimodal_jev/local_runtime_storage.py').read_text()
    ssh(['python3','-c',storage+'\nfrom pathlib import Path\ncheck_paths({"job":'+repr(root)+'})\nr=Path('+repr(root)+');r.mkdir();[(r/n).mkdir() for n in ("package","home","tmp")]'])
    archive=out/'package.tar'
    with tarfile.open(archive,'w') as t:t.add(pkg,arcname='package')
    subprocess.run(['scp','-q',str(archive),c['host']+':'+root+'/package.tar'],check=True)
    ssh(['tar','-xf',root+'/package.tar','-C',root])
    ssh(['python3','-c','from pathlib import Path;import json,hashlib;r=Path('+repr(root+'/package')+');m=json.loads((r/"FILES.json").read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m.items())'])
    argv=['docker','create','--name',c['tag'],'--network=none','--cpus='+str(c['cache']['workers']),'--memory=32g']
    for src,dst,ro in [(root,root,False),(root+'/home','/root',False),(root+'/tmp','/jev-tmp',False),(c['driver_root'],'/usr/local/Ascend/driver',True),*[(x,x,True) for x in c['readonly_mounts']],*[(x['root'],x['destination'],True) for x in c.get('relocated_mounts',[])]]:
        argv+=['--mount','type=bind,src='+src+',dst='+dst+(',readonly' if ro else '')]
    for k,v in dict(PYTHONPATH=root+'/package/src',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TMPDIR='/jev-tmp',HF_HOME=root+'/tmp',XDG_CACHE_HOME=root+'/tmp',TRITON_CACHE_DIR=root+'/tmp',TORCHINDUCTOR_CACHE_DIR=root+'/tmp',ASCEND_WORK_PATH=root+'/tmp',ASCEND_CACHE_PATH=root+'/tmp').items():argv+=['-e',k+'='+v]
    argv+=['--entrypoint','python',c['image'],root+'/package/run.py','--config',root+'/package/config.json','--output',root+'/run']
    cid=ssh(argv).strip();record=dict(container_id=cid,argv=argv,source_files=files,config=c,provenance=provenance,created_unix=time.time())
    commit_json(out/'launch.json',record);ssh(['docker','start',cid]);return dict(container_id=cid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True);a=p.parse_args();c,pr=load_config(a.config,a.local);print(json.dumps(launch(c,pr)))
