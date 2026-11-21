"""Cancel one explicitly bound experiment; retain metadata, never delete artifacts."""
import argparse,json,os,signal,subprocess,shlex,time
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json


def cancel(c):
    out=Path(c['receipt'])
    if out.exists():return json.loads(out.read_text())
    launch=json.loads(Path(c['launch']).read_text());cid=launch['container_id']
    if launch['remote_root']!=c['remote_root']:raise ValueError('cancellation root mismatch')
    def remote(args):return subprocess.check_output(['ssh',c['host'],shlex.join(args)],text=True)
    before=json.loads(remote(['docker','inspect',cid]))[0]
    if before['Id']!=cid or not any(m['Source']==c['remote_root'] and m['Destination']==c['remote_root'] and m['RW'] for m in before['Mounts']):raise ValueError('exact owned training mount required')
    h=json.loads(Path(c['driver_handle']).read_text());pid=h['pid'];stat=Path(f'/proc/{pid}/stat')
    if stat.exists():
        fd=os.pidfd_open(pid)
        try:
            if stat.read_text().split(') ',1)[1].split()[19]!=h['process_start_ticks']:raise ValueError('driver PID reused')
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
            if [x.decode() for x in cmd]!=h['argv']:raise ValueError('driver argv changed')
            signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally:os.close(fd)
    if before['State']['Running']:remote(['docker','stop','--time','30',cid])
    after=json.loads(remote(['docker','inspect',cid]))[0]
    if after['State']['Running']:raise ValueError('training still running')
    result=dict(status='cancelled_superseded_objective',reason=c['reason'],container_id=cid,
        before=before['State'],after=after['State'],remote_root=c['remote_root'],
        use_as_stage2_reference=False,use_as_formal_result=False,artifacts_deleted=False,unix=time.time())
    commit_json(out,result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);x=a.parse_args();c,_=load_config(x.config,x.local);print(json.dumps(cancel(c),indent=2))
