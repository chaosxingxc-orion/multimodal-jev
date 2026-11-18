"""Independent collection of a configured CPU-only test container."""
from pathlib import Path
import subprocess,shlex,json,hashlib,xml.etree.ElementTree as ET
import argparse
from multimodal_jev.local_config import load_config
parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
a=parser.parse_args();config,provenance=load_config(a.config,a.local)
base=Path(config['output']);tag=config['tag']
launch=json.loads((base/(tag+'-launch.json')).read_text());root=launch['remote_root'];cid=launch['container_id']
def ssh(args):return subprocess.run(['ssh','-o','ConnectTimeout=10',config['host'],shlex.join(args)],capture_output=True,text=True,check=True,timeout=55).stdout
inspection=json.loads(ssh(['docker','inspect',cid]))[0];state=inspection['State']
if state['Status']!='exited':raise ValueError('Not terminal; retain exact handle')
if inspection['Image']!=launch['image'] or inspection['Id']!=cid:raise ValueError('Runtime identity changed')
if inspection['HostConfig']['Devices']:raise ValueError('CPU check received devices')
expected=launch['argv'];image_index=expected.index(launch['image'])
if inspection['Config']['Cmd']!=expected[image_index+1:] or inspection['Config']['Entrypoint']!=['python']:raise ValueError('Runtime command changed')
if inspection['HostConfig']['NetworkMode']!='none':raise ValueError('Network scope changed')
code="""from pathlib import Path
import hashlib,json
r=Path(ROOT);m=json.loads((r/'package/FILES.json').read_text())
assert all(hashlib.sha256((r/'package'/p).read_bytes()).hexdigest()==h for p,h in m.items())
names=['run/terminal.json','run/junit.xml','run/pytest.log','package/FILES.json','package/run.py']
print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names if (r/n).is_file()}))
""".replace('ROOT',repr(root))
hashes=json.loads(ssh(['python3','-c',code]))
out=base/(tag+'-collected');out.mkdir(exist_ok=False)
for n,h in hashes.items():
 p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['scp','-q',config['host']+':'+root+'/'+n,str(p)],capture_output=True,text=True,check=True,timeout=55)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Collection hash mismatch')
(out/'inspect.json').write_text(json.dumps(inspection,indent=2)+'\n')
terminal=json.loads((out/'run/terminal.json').read_text())
if hashes['package/FILES.json']!=launch['source_manifest_sha256'] or hashes['package/run.py']!=launch['runner_sha256']:raise ValueError('Actual source differs')
junit=ET.parse(out/'run/junit.xml').getroot()
suites=[junit] if junit.tag=='testsuite' else list(junit.iter('testsuite'))
counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
report=dict(schema='configured-fixed-image-cpu-tests-v1',container_id=cid,image=inspection['Image'],state=state,
 source_files=len(launch['source_files']),source_manifest_sha256=hashes['package/FILES.json'],
 collection=str(out.resolve()),files=hashes,counts=counts,terminal=terminal,
 scope=config['scope'])
with Path(config['report']).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(state=state,counts=counts,log=str(out/'run/pytest.log')),indent=2))
if counts['tests']!=config['expected_tests'] or state['ExitCode'] or terminal['exit_code'] or counts['errors'] or counts['failures'] or counts['skipped']:
 raise ValueError('Fixed image CPU checks incomplete or failed')
