"""Readonly storage plan for reference production and DP2 training; no launch entry.

Reuse the shared Docker command builder without modifying frozen launchers.
Actual deployment still requires source/reference/parity admission and exact
Docker inspect comparison; an argv plan does not prove mounted permissions.
"""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports import context_replica_training_operations as shared


def overlaps(a,b):
    return Path(a).is_relative_to(b) or Path(b).is_relative_to(a)


def mounts(argv):
    result=[]
    for i,value in enumerate(argv):
        if value!='-v':continue
        parts=argv[i+1].split(':')
        if len(parts) not in (2,3) or (len(parts)==3 and parts[2]!='ro'):
            raise ValueError('explicit bind mount access required')
        result.append((shared.path(parts[0]),shared.path(parts[1]),len(parts)==3))
    if len({dst for _,dst,_ in result})!=len(result):
        raise ValueError('duplicate mount destination')
    return result


def audit_mounts(argv,config,*,producer=False):
    bindings=mounts(argv)
    protected=[config['source_root'],str(Path(config['source_policy']).parent),
        str(Path(config['encoded_audio_cache']).parent),config['audio_mirror_root']]
    if producer:
        if 'reference_dataset' in config or 'reference_cache' in config:
            raise ValueError('reference producer cannot consume a reference cache')
    else:protected.append(config['reference_dataset']['root'])
    protected.extend(config['pins'])
    if config.get('resume_root'):protected.append(config['resume_root'])
    for name in protected:
        shared.path(name)
        parents=[m for m in bindings if Path(name).is_relative_to(m[1])]
        if not parents or not max(parents,key=lambda m:len(Path(m[1]).parts))[2]:
            raise ValueError('dependency not effectively readonly: '+name)
        if any(not ro and Path(dst).is_relative_to(name) for _,dst,ro in bindings):
            raise ValueError('writable nested mount shadows readonly dependency: '+name)
    return dict(protected_paths=len(protected),readonly_mounts=sum(m[2] for m in bindings),
                actual_container_inspect_pending=True)


def command(c,m,config,*,producer=False):
    remote=shared.path(m['remote_root'])
    reference=None if producer else shared.path(config['reference_dataset']['root'])
    initial=str(Path(shared.path(config['source_policy'])).parent)
    if initial!=remote+'/initial':
        raise ValueError('packaged current endpoint under run/initial required')
    if reference is not None and overlaps(reference,remote):
        raise ValueError('reference publication must be disjoint from writable runtime')
    train=m['training_devices'];services=m['inference_services']
    devices=[*train,*[s['device'] for s in services]]
    shape=(len(train),len(services))
    if 'batch_config' in config and train:
        batch=config['batch_config']
        expected=(batch['training_devices'],batch['inference_instances'])
        shape_ok=shape==expected and train==config['physical_devices']
    elif producer and not train:
        shape_ok=len(services)==len(config['macro_client']['replicas']) and bool(services)
    else:
        shape_ok=shape==(2,2)
    if (not shape_ok
            or len(set(devices))!=len(devices)
            or any(type(d)is not int or d<0 for d in devices)):
        raise ValueError('supported disjoint cached training or TRAIN scoring devices required')
    # Transform only storage decisions. The original launcher remains frozen;
    # common image, runtime environment, resource limits and argv stay shared.
    source=inspect.getsource(shared.command)
    edits=[
        ('(cache, cache, False), (mirror, mirror, True)',
         '(cache, cache, True), (mirror, mirror, True)'+('' if producer else ',\n'
         "              (config['reference_dataset']['root'], config['reference_dataset']['root'], True)")),
        ('(shared, initial, str(Path(cache).parent), mirror, driver)',
         '(shared, str(Path(cache).parent), mirror, driver)' if producer else
         "(shared, str(Path(cache).parent), mirror, driver, config['reference_dataset']['root'])"),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('shared mount builder changed: '+before)
        source=source.replace(before,after)
    ns=dict(shared.command.__globals__)
    exec(compile(source,'<cached-storage-command>','exec'),ns)
    argv=ns['command'](c,m,config)
    if config.get('resume_root'):
        resume=shared.path(config['resume_root'])
        if producer or resume!=remote+'/resume' or config.get('start_update')!=1 or config.get('stop_update')!=2:
            raise ValueError('packaged first-step recovery input required')
        for name in ('checkpoint-0001-committed.json',*(f'rank-{i}/checkpoint-0001.pt' for i in range(len(train))),'policy-0001/MANIFEST.json'):
            if resume+'/'+name not in config['pins']:raise ValueError('recovery asset pin missing: '+name)
        i=argv.index('--entrypoint');argv[i:i]=['-v',resume+':'+resume+':ro']
    # Source binding/audit receipts at the package top are dependencies too;
    # protect each otherwise writable pinned file under the run-root mount.
    for name in config['pins']:
        parents=[b for b in mounts(argv) if Path(name).is_relative_to(b[1])]
        if parents and not max(parents,key=lambda b:len(Path(b[1]).parts))[2]:
            i=argv.index('--entrypoint')
            argv[i:i]=['-v',name+':'+name+':ro']
    audit_mounts(argv,config,producer=producer)
    return argv


def storage_preflight(config,remote,*,producer=False):
    """Remote Python preflight: only the new run root needs writable storage.

    Caller prepends the sealed local_runtime_storage module. Shared inputs are
    existence/hash reads only: no mkdir, touch, chmod or fsync in those roots.
    """
    remote=shared.path(remote)
    readonly=[shared.path(config['encoded_audio_cache']),shared.path(config['audio_mirror_root'])]
    if producer:
        if 'reference_dataset' in config or 'reference_cache' in config:
            raise ValueError('reference producer cannot consume a reference cache')
    else:readonly.append(shared.path(config['reference_dataset']['root']))
    if any(overlaps(p,remote) for p in readonly):
        raise ValueError('readonly inputs and new runtime must be disjoint')
    external={shared.path(p):h for p,h in config['pins'].items() if not Path(p).is_relative_to(remote)}
    if not producer and external.get(readonly[-1]+'/MANIFEST.json')!=config['reference_dataset']['manifest_sha256']:
        raise ValueError('reference manifest must be pinned externally')
    for digest in external.values():
        if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('explicit SHA256 dependency pins required')
    return ('from pathlib import Path\nimport hashlib\n'
        'check_paths('+repr({'run':remote})+')\n'
        'assert not Path('+repr(remote)+').exists(), "runtime already exists"\n'
        'for directory in '+repr(readonly)+':\n'
        ' assert Path(directory).is_dir(), directory\n'
        'for name,digest in '+repr(external)+'.items():\n'
        ' assert not Path(name).is_symlink(), name\n'
        ' assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name\n')


def audit_container(argv,config,handle,observed,*,producer=False,expected_exit=0):
    """Require actual terminal inspect to agree with every source/access bind."""
    from tools.reports.context_dual_probe_operations import check_runtime
    check_runtime(dict(argv=argv,image=shared.IMAGE),handle,observed,expected_exit=expected_exit)
    declared=mounts(argv);actual=observed.get('Mounts')
    if not isinstance(actual,list):raise ValueError('actual container mounts required')
    expected={dst:(src,not ro) for src,dst,ro in declared}
    if len({m.get('Destination') for m in actual})!=len(actual):
        raise ValueError('duplicate actual container mount')
    for item in actual:
        dst=item.get('Destination')
        if dst not in expected or item.get('Type')!='bind':
            raise ValueError('undeclared actual container mount')
        src,rw=expected[dst]
        if item.get('Source')!=src or type(item.get('RW')) is not bool or item['RW']!=rw:
            raise ValueError('actual mount source/access differs: '+str(dst))
    if {m['Destination'] for m in actual}!=set(expected):
        raise ValueError('missing actual container mount')
    report=audit_mounts(argv,config,producer=producer)
    return dict(report,actual_container_inspect_pending=False,
                container_id=observed['Id'],actual_readonly_mounts_verified=True)


def reference_command(c,m,config):
    if (m.get('status')!='full_train_reference_job_prepared_only'
            or m.get('actor_updates')!=0 or m.get('expected_population')!=config.get('reference_source',{'population':40513})['population']):
        raise ValueError('full TRAIN reference-only package required')
    return command(c,m,config,producer=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    package=Path(c['package']);m=json.loads((package/'PREPARED.json').read_text())
    if shared.sha(package/'PREPARED.json')!=c['manifest_sha256']:
        raise ValueError('storage plan package changed')
    actual={str(f.relative_to(package)):shared.sha(f) for f in package.rglob('*')
            if f.is_file() and f!=package/'PREPARED.json'}
    if any(f.is_symlink() for f in package.rglob('*')) or actual!=m['files']:
        raise ValueError('storage plan needs complete sealed package')
    config=json.loads((package/'config.json').read_text())
    kind=c.get('kind','cached_training')
    if kind not in ('reference_build','cached_training'):raise ValueError('known storage operation required')
    producer=kind=='reference_build'
    argv=reference_command(c,m,config) if producer else command(c,m,config)
    print(json.dumps(dict(status='storage_plan_only_not_dispatch_admission',argv=argv,
        mounts=audit_mounts(argv,config,producer=producer),
        preflight=storage_preflight(config,m['remote_root'],producer=producer),
        provenance=provenance),indent=2))
