"""Configured CPU-only test snapshot; persist exact handle before starting."""
from pathlib import Path
import json,hashlib,shutil,subprocess,shlex,time
import argparse
from multimodal_jev.local_config import load_config
parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
a=parser.parse_args();config,provenance=load_config(a.config,a.local)
base=Path(config['output']);base.mkdir(parents=True,exist_ok=True);tag=config['tag']
if not __import__('re').fullmatch('[a-z0-9-]+',tag):raise ValueError('invalid tag')
pkg=base/(tag+'-package');pkg.mkdir(exist_ok=False)
root=config['remote_root']
image=config['image']
if image!='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c':raise ValueError('fixed image required')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pinned=Path(config['source'])
for p in (pinned/'src/multimodal_jev').glob('*.py'):
 dest=pkg/p.relative_to(pinned);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for name in sorted(set(config['tests'] + config.get('support_files', []))):
 if Path(name).is_absolute() or '..' in Path(name).parts:
  raise ValueError('package support paths must be relative source files')
 p=pinned/name;dest=pkg/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(pkg/'tests/__init__.py').write_text('')
runner="""from pathlib import Path
import json,time,subprocess,sys,hashlib
r=Path(__file__).parent.parent
package=r/'package'
files=json.loads((package/'FILES.json').read_text())
if any(hashlib.sha256((package/p).read_bytes()).hexdigest()!=h for p,h in files.items()):
 raise ValueError('CPU source snapshot changed')
assert not list(Path('/dev').glob('davinci[0-9]*'))
import torch
torch.set_num_threads(1)
from multimodal_jev.local_runtime_storage import check_paths
check_paths({'run':r/'run','home':Path('/root'),'cache':r/'cache'})
argv=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--junitxml='+str(r/'run/junit.xml')]
argv+=json.loads((package/'TESTS.json').read_text())
start=time.time()
with (r/'run/pytest.log').open('x') as f:
 p=subprocess.run(argv,cwd=package,stdout=f,stderr=subprocess.STDOUT)
terminal=dict(exit_code=p.returncode,started_unix=start,finished_unix=time.time(),
 source_manifest_sha256=hashlib.sha256((package/'FILES.json').read_bytes()).hexdigest(),argv=argv,torch_version=torch.__version__,python_version=sys.version)
from multimodal_jev.local_checkpoint import commit_json
commit_json(r/'run/terminal.json',terminal)
raise SystemExit(p.returncode)
"""
(pkg/'run.py').write_text(runner)
(pkg/'TESTS.json').write_text(json.dumps(config['tests']))
files={str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file()}
(pkg/'FILES.json').write_text(json.dumps(files,indent=2)+'\n')
def ssh(args):return subprocess.run(['ssh','-o','ConnectTimeout=10',config['host'],shlex.join(args)],capture_output=True,text=True,check=True,timeout=55).stdout
storage_source=Path('src/multimodal_jev/local_runtime_storage.py').read_text()
print(ssh(['python3','-c',storage_source+"\ncheck_paths({'root':"+repr(str(Path(root).parent))+"})\n"+"from pathlib import Path\nimport shutil,subprocess,json\nd=shutil.disk_usage("+repr(str(Path(root).parent))+")\nassert d.free/d.total>.10 and d.free>20*1024**3\nsubprocess.run(['docker','image','inspect',"+repr(image)+"],stdout=subprocess.DEVNULL,check=True)\nr=Path("+repr(root)+")\nr.mkdir(exist_ok=False)\n[(r/n).mkdir() for n in ('package','run','home','cache')]\nprint(json.dumps(dict(free=d.free,fraction=d.free/d.total)))"]),flush=True)
prod=subprocess.Popen(['tar','-C',str(pkg),'-cf','-','.'],stdout=subprocess.PIPE)
p=subprocess.run(['ssh',config['host'],shlex.join(['tar','-C',root+'/package','-xf','-'])],stdin=prod.stdout,capture_output=True,text=True,timeout=55)
prod.stdout.close()
if prod.wait()!=0 or p.returncode:raise ValueError('CPU package transfer failed')
manifest_sha=sha(pkg/'FILES.json')
check="""from pathlib import Path
import json,hashlib
r=Path(ROOT)
assert hashlib.sha256((r/'FILES.json').read_bytes()).hexdigest()==DIGEST
files=json.loads((r/'FILES.json').read_text())
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in files.items())
print(json.dumps(dict(verified=len(files))))
""".replace('ROOT',repr(root+'/package')).replace('DIGEST',repr(manifest_sha))
print(ssh(['python3','-c',check]),flush=True)
resources=config.get('resources',dict(cpus=2,memory_gib=8))
if (set(resources)!={'cpus','memory_gib'} or
    any(type(v) is not int or not 1<=v<=64 for v in resources.values())):
 raise ValueError('bounded integer CPU/memory test resources required')
argv=['docker','create','--name',tag,'--pull=never','--network=none',
 '--cpus='+str(resources['cpus']),'--memory='+str(resources['memory_gib'])+'g',
 '--log-driver=json-file','--log-opt','max-size=100m','--log-opt','max-file=3']
for src,dst,ro in [(root+'/package',root+'/package',True),(root+'/run',root+'/run',False),
 (root+'/home','/root',False),(root+'/cache',root+'/cache',False),(root+'/cache','/jev-tmp',False),
 (config['driver_root'],'/usr/local/Ascend/driver',True)]:
 argv+=['--mount','type=bind,src='+src+',dst='+dst+(',readonly' if ro else '')]
env=dict(PYTHONPATH=root+'/package/src:'+root+'/package/tools/jev:'+root+'/package/tests:'+root+'/package',
 PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for n in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):env[n]=root+'/cache'
env['TMPDIR']='/jev-tmp'
for n,v in env.items():argv+=['-e',n+'='+v]
argv+=['--entrypoint','python',image,root+'/package/run.py']
cid=ssh(argv).strip()
record=dict(container_id=cid,image=image,remote_root=root,argv=argv,
 source_files=files,source_manifest_sha256=manifest_sha,runner_sha256=sha(pkg/'run.py'),created_unix=time.time(),config_provenance=provenance)
with (base/(tag+'-launch.json')).open('x') as f:json.dump(record,f,indent=2)
ssh(['docker','start',cid])
state=json.loads(ssh(['docker','inspect',cid]))[0]['State']
with (base/(tag+'-start.json')).open('x') as f:json.dump(dict(container_id=cid,state=state,observed_unix=time.time()),f,indent=2)
print(json.dumps(dict(container_id=cid,state=state,files=len(files))),flush=True)
