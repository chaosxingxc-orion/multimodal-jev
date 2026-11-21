"""Collect an explicit subset of a sealed transport manifest, without rescanning.

The original full collector remains independent. A subset is content evidence,
never a complete training audit. No devices are granted to the reader container.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.collect_context_rl_training import unpack
from tools.jev.collect_context_rl_selection import check_handle
from tools.reports.prepare_objective_training_candidate import read_bound


def selected_files(files, names, prefixes):
    def safe(name):
        p=Path(name)
        if not name or p.is_absolute() or '..' in p.parts or str(p)!=name.rstrip('/'):
            raise ValueError('canonical relative collection member required')
    for n in names+prefixes:safe(n)
    result={n:h for n,h in files.items() if n in names or any(n.startswith(p.rstrip('/')+'/') for p in prefixes)}
    if set(names)-result.keys() or any(not any(n.startswith(p.rstrip('/')+'/') for n in result) for p in prefixes):
        raise ValueError('requested collection member absent')
    for n in result:safe(n)
    if not result:raise ValueError('nonempty explicit subset required')
    return result


READER='''import json,sys,tarfile,hashlib
from pathlib import Path
c=json.loads(sys.stdin.read());r=Path(c['root'])
for name,digest in c['files'].items():
 p=r/name
 if not p.is_file() or any(q.is_symlink() for q in [p,*p.parents]):raise ValueError('regular file required')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('source changed: '+name)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for name in sorted(c['files']):tar.add(r/name,arcname=name,recursive=False)
'''


def collect(c,provenance):
    root=Path(c['output']);check_paths({'subset':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        launch=json.loads(read_bound(c['launch']));transport=json.loads(read_bound(c['transport']))
        if transport['container_id']!=launch['container_id'] or transport['image']!=launch['image']:
            raise ValueError('exact source container required')
        inspected=json.loads(subprocess.check_output(['ssh',c['host'],shlex.join(['docker','inspect',launch['container_id']])],text=True))[0]
        check_handle(launch,inspected)
        if inspected['State']['Running'] or inspected['State']['ExitCode'] or inspected['State']['OOMKilled']:
            raise ValueError('successfully closed source required')
        files=selected_files(transport['files'],c['names'],c['prefixes'])
        if len(files)>c['max_files']:raise ValueError('subset file budget exceeded')
        archive=root/'subset.tar';out=root/'files';out.mkdir()
        argv=['docker','run','--rm','-i','--network=none','--volumes-from',launch['container_id']+':ro','--entrypoint','python',launch['image'],'-c',READER]
        with archive.open('xb') as stream:
            subprocess.run(['ssh',c['host'],shlex.join(argv)],input=json.dumps(dict(root=launch['remote_root'],files=files)).encode(),stdout=stream,check=True)
        if archive.stat().st_size>c['max_bytes']:raise ValueError('subset byte budget exceeded')
        unpack(archive,out,files)
        report=dict(status='sealed_subset_collected_training_audit_pending',container_id=launch['container_id'],
                    files=files,root=str(out),bytes=archive.stat().st_size,archive_sha256=file_hash(archive),training_audited=False)
        commit_json(root/'report.json',report);commit_json(root/'terminal.json',dict(exit_code=0,status=report['status']))
        return report
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True);a=p.parse_args()
    c,prov=load_config(a.config,a.local);r=collect(c,prov);print(json.dumps({k:v for k,v in r.items() if k!='files'}))
