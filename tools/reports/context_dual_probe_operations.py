"""Configured finite inference launch/collection; reuse existing transport/audit.

Default validation is local only. Launch owns the original exclusive lease,
requires the recovered research queue to finish, and persists an exact handle
before starting. A partial attempt is never treated as permission to restart.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_context_batch_probe_candidate import IMAGE, ROOT, command, sha
from tools.reports.prepare_context_dual_probe_candidate import rebind_argv
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.launch_context_batch_probe_candidate import (
    archive_package, check_device_observation, wait_exact, write_new)
from tools.reports.context_dual_rollout_candidate import RUNTIME
from multimodal_jev.context_logits_wire import WIRE

AUDITOR_SHA='6be9097c1229c0dbf65ed6a45ceb77c6956c60df8e180bd4fd4b188da6e82ec2'


def validate(c):
    prepared=Path(c['prepared']);package=prepared/'package'
    receipt=json.loads((prepared/'PREPARED.json').read_text())
    config=json.loads((package/'config.json').read_text());policy=config['policy']
    remote=c['remote_root'];tag=c['tag']
    if (not re.fullmatch('[a-z0-9-]+',tag) or not re.fullmatch('[a-zA-Z0-9_.-]+',c['host'])
            or c['host'].startswith('-') or not re.fullmatch('/[a-zA-Z0-9_./-]+',remote)
            or '..' in Path(remote).parts or remote==ROOT):
        raise ValueError('explicit safe remote bindings required')
    expected=rebind_argv(command(c['model']),ROOT,remote,c['model'],c['model'],
                         c['driver_root'],7,tag)
    if (receipt['status']!='sealed_not_launched' or receipt['image']!=IMAGE
            or receipt['remote_root']!=remote or receipt['tag']!=tag or receipt['host']!=c['host']
            or receipt['training_dispatch_allowed'] is not False
            or receipt['files_sha256']!=verify(package) or receipt['argv']!=expected
            or receipt['files_sha256']!=c['package_manifest_sha256']
            or config['source_root']!=remote+'/package' or config['inputs']!=remote+'/package/inputs.pt'
            or policy['model']!=c['model'] or policy['snapshot']!=remote+'/package/policy'
            or policy['adapter']!=remote+'/package/adapter' or policy['physical_devices']!=[7]
            or policy['max_batch_queries']!=4 or policy['max_batch_tokens']!=16384
            or policy.get('process_device_scope') is not None
            or policy['candidate_runtime']!=RUNTIME or policy['output_protocol']!=WIRE):
        raise ValueError('sealed dual inference package or command differs')
    files=json.loads((package/'FILES.json').read_text())
    if (config['pins']!={remote+'/package/'+n:h for n,h in files.items() if n!='config.json'}
            or sha(package/'inputs.pt')!=config['input_sha256']):
        raise ValueError('complete payload/input pins required')
    return receipt,config


def check_priority(c):
    mode=c.get('priority_mode','completed_research_chain')
    if mode=='idle_resource_window':
        # User allows early replacement after measured gains. The caller still
        # holds the original exclusive lease and checks actual device idleness
        # before any launch; no existing job is stopped by this mode.
        return
    if mode!='completed_research_chain':raise ValueError('unknown resource priority mode')
    root=Path(c['priority_root'])
    plan=json.loads((root/'input.json').read_text())
    if not plan.get('owner_locks'):raise ValueError('predecessor ownership locks required')
    with ExitStack() as stack:
        for name in plan['owner_locks']:
            owner=stack.enter_context(Path(name).open('r'))
            fcntl.flock(owner,fcntl.LOCK_SH|fcntl.LOCK_NB)
        status=json.loads((root/'status.json').read_text())
        expected=[s['name'] for s in plan['stages']]
        if (status['phase']!='complete_predeclared_chain' or not expected
                or status.get('completed')!=expected):
            raise ValueError('existing recovered training/evaluation chain has priority')
        for stage in plan['stages']:
            evidence=json.loads((root/(stage['name']+'-complete.json')).read_text())
            if evidence['outputs']!={r['path']:sha(r['path']) for r in stage['results']}:
                raise ValueError('completed predecessor evidence changed')
            for result in stage['results']:
                data=json.loads(Path(result['path']).read_text())
                if any(data.get(k)!=v for k,v in result['equals'].items()):
                    raise ValueError('predecessor terminal result differs')


def auditor_namespace(c):
    path=Path(c['source'])/'tools/reports/audit_context_batch_probe_candidate.py'
    if sha(path)!=AUDITOR_SHA:raise ValueError('original independent auditor changed')
    namespace={'__name__':__name__+'.audit'}
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    from tools.reports.context_dual_measurement_candidate import audit_dual_measurements
    namespace['validate']=lambda prepared:validate(dict(c,prepared=str(prepared)))
    namespace['audit_measurements']=audit_dual_measurements
    return namespace


def check_runtime(receipt,handle,observed):
    argv=receipt['argv']
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    hc=observed['HostConfig'];cfg=observed['Config'];state=observed['State']
    if (handle['argv']!=argv or observed['Id']!=handle['container_id']
            or observed['Image']!=receipt['image'] or state['Status']!='exited'
            or state['ExitCode']!=0 or state['OOMKilled'] or hc['Privileged']
            or hc['NetworkMode']!='none' or sorted(hc['Binds'])!=sorted(values('-v'))
            or cfg['Entrypoint']!=values('--entrypoint')
            or cfg['Cmd']!=argv[argv.index(receipt['image'])+1:]):
        raise ValueError('actual isolated terminal runtime differs')
    actual=dict(v.split('=',1) for v in cfg['Env'])
    if any(actual.get(k)!=v for k,v in (e.split('=',1) for e in values('-e'))):
        raise ValueError('actual runtime environment differs')
    expected=sorted((*v.split(':'),'rwm') for v in values('--device'))
    if sorted((d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in hc['Devices'])!=expected:
        raise ValueError('actual physical device scope differs')


def audit(c,collection,handle):
    receipt,_=validate(c)
    launch_record=json.loads(Path(handle).read_text())
    inspected=Path(c['operations_root'])/(c['tag']+'-inspect.json')
    check_runtime(receipt,launch_record,json.loads(inspected.read_text()))
    result=auditor_namespace(c)['audit'](c['prepared'],collection,handle)
    comparison=json.loads((Path(collection)/'evaluation/comparison.json').read_text())
    if comparison.get('exact_actual_logits') is not True:
        raise ValueError('actual logits comparison not recorded')
    return dict(result,runtime_inspect_sha256=sha(inspected))


def launch(c,provenance):
    receipt,_=validate(c)
    if c.get('finite_inference_dispatch_allowed') is not True:
        raise ValueError('finite inference operations release not admitted')
    # No remote calls before priority, local reserve, and both leases pass.
    check_priority(c)
    base=Path(c['operations_root']);base.mkdir(parents=True,exist_ok=True)
    check_paths({'operations':base})
    tag=c['tag'];remote_root=c['remote_root'];host=c['host']
    with (base/(tag+'.lock')).open('a') as owner,Path(c['lease']).open('r') as lease:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt,handle=[base/(tag+s) for s in ('-attempt.json','-launch.json')]
        if attempt.exists() or handle.exists():
            raise ValueError('existing attempt/handle: preserve original, never recreate')
        check_priority(c);validate(c)
        def remote(args,bounded=True):
            return subprocess.run(['ssh','-o','ConnectTimeout=10',host,shlex.join(args)],
                text=True,capture_output=True,timeout=55 if bounded else None)
        def checked(args):
            r=remote(args)
            if r.returncode:raise RuntimeError(r.stderr)
            return r.stdout
        if json.loads(checked(['docker','image','inspect',IMAGE]))[0]['Id']!=IMAGE:
            raise ValueError('fixed image unavailable')
        check_device_observation(checked(['npu-smi','info']))
        storage=(Path(c['prepared'])/'package/src/multimodal_jev/local_runtime_storage.py').read_text()
        script=storage+'\ncheck_paths({"root":'+repr(str(Path(remote_root).parent))+'})\n'
        script+='assert not Path('+repr(remote_root)+').exists()\n'
        checked(['python3','-B','-c',script])
        archive=base/(tag+'-package.tar.gz');archive_package(c['prepared'],archive)
        write_new(attempt,dict(config=c,provenance=provenance,prepared_sha256=sha(Path(c['prepared'])/'PREPARED.json'),
                               files_sha256=receipt['files_sha256'],archive_sha256=sha(archive)))
        checked(['mkdir',remote_root])
        checked(['mkdir',remote_root+'/home',remote_root+'/cache'])
        subprocess.run(['scp',str(archive),host+':'+remote_root+'/package.tar.gz'],
                       check=True,capture_output=True,timeout=55)
        checked(['tar','-xzf',remote_root+'/package.tar.gz','-C',remote_root])
        script='import sys;sys.path.insert(0,'+repr(remote_root+'/package')+');'
        script+='from tools.reports.run_context_batch_probe_candidate import verify;'
        script+='assert verify('+repr(remote_root+'/package')+')=='+repr(receipt['files_sha256'])
        # Importing the verifier must not add bytecode to its immutable payload.
        checked(['python3','-B','-c',script])
        # An ambiguous create leaves the attempt receipt and deterministic name;
        # an operator must reconcile it, never issue another create automatically.
        cid=checked(receipt['argv']).strip()
        if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('invalid exact container handle')
        record=dict(container_id=cid,image=IMAGE,host=host,remote_root=remote_root,argv=receipt['argv'],
            config_sha256=sha(Path(c['prepared'])/'package/config.json'),files_sha256=receipt['files_sha256'])
        write_new(handle,record)
        try:
            response=remote(['docker','start',cid])
            write_new(base/(tag+'-start-response.json'),dict(returncode=response.returncode,
                stdout=response.stdout,stderr=response.stderr))
        except subprocess.TimeoutExpired:
            write_new(base/(tag+'-start-response.json'),dict(observation_timeout=True))
        code=wait_exact(remote,cid)
        write_new(base/(tag+'-wait.json'),dict(container_id=cid,exit_code=code))
        from tools.jev.collect_context_rl_selection import collect
        collection=base/(tag+'-collected');collect(record,collection,host=host)
        write_new(base/(tag+'-inspect.json'),json.loads(checked(['docker','inspect',cid]))[0])
        result=audit(c,collection,handle)
        write_new(base/(tag+'-audit.json'),result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','launch','audit'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='launch':result=launch(c,provenance)
    elif a.action=='audit':
        base=Path(c['operations_root']);tag=c['tag']
        result=audit(c,base/(tag+'-collected'),base/(tag+'-launch.json'))
        write_new(base/(tag+'-audit.json'),result)
    else:
        receipt,_=validate(c)
        result=dict(status='validated_only_no_remote_calls',files_sha256=receipt['files_sha256'],
                    finite_inference_dispatch_allowed=c['finite_inference_dispatch_allowed'],training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
