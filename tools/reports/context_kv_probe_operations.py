"""Seal and independently audit a KV-only comparison after acoustic admission."""
import argparse
import json
from pathlib import Path
import re
import shutil

from multimodal_jev.local_config import load_config
from tools.reports.prepare_context_batch_probe_candidate import sha,ROOT,command,IMAGE
from tools.reports.prepare_context_dual_probe_candidate import rebind_argv
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.context_acoustic_probe_operations import namespace as parent_namespace
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.launch_context_batch_probe_candidate import write_new

ADDITIONS=('run_context_kv_probe.py',)


def argv(c):
    args=rebind_argv(command(c['model']),ROOT,c['remote_root'],c['model'],c['model'],c['driver_root'],7,c['tag'])
    before=c['remote_root']+'/package/tools/reports/run_context_dual_probe_candidate.py'
    args[args.index(before)]=c['remote_root']+'/package/tools/reports/run_context_kv_probe.py'
    return args


def validate(c):
    parent=c['parent_operations'];pr,pc=parent_namespace(parent)['validate'](parent)
    base=Path(c['prepared']); package=base/'package';p=json.loads((package/'config.json').read_text())
    receipt=json.loads((base/'PREPARED.json').read_text())
    remote=Path(c['remote_root']);old=Path(parent['remote_root'])
    if (not re.fullmatch('/[a-zA-Z0-9_./-]+',str(remote)) or '..' in remote.parts
            or remote.is_relative_to(old) or old.is_relative_to(remote)
            or not re.fullmatch('[a-z0-9-]+',c['tag'])):
        raise ValueError('disjoint bounded safe KV paths required')
    if (receipt['image']!=IMAGE or receipt['argv']!=argv(c) or receipt['status']!='sealed_kv_probe'
            or receipt['files_sha256']!=verify(package) or receipt['files_sha256']!=c['package_manifest_sha256']
            or receipt['parent_manifest_sha256']!=pr['files_sha256'] or receipt['remote_root']!=c['remote_root']
            or c['remote_root']==parent['remote_root'] or c['host']!=parent['host'] or c['lease']!=parent['lease']):
        raise ValueError('new exact inherited KV probe required')
    source=Path(parent['prepared'])/'package'
    for path in source.rglob('*'):
        if path.is_file() and path.name not in ('FILES.json','config.json') and sha(path)!=sha(package/path.relative_to(source)):
            raise ValueError('inherited runtime/asset changed')
    expected=relocate(pc,parent['remote_root'],c['remote_root']);expected.pop('pins')
    expected['kv_probe']=dict(baseline_bytes=pc['policy']['kv_cache_memory_bytes'],
                              candidate_bytes=c['candidate_kv_cache_memory_bytes'])
    from tools.reports.run_context_kv_probe import arm_config
    arm_config(expected,'candidate')
    actual=dict(p);actual.pop('pins')
    if actual!=expected:raise ValueError('policy/input/capacity configuration changed')
    files=json.loads((package/'FILES.json').read_text())
    if p['pins']!={c['remote_root']+'/package/'+n:h for n,h in files.items() if n!='config.json'}:
        raise ValueError('complete probe pins required')
    for name in ADDITIONS:
        if sha(package/'tools/reports'/name)!=sha(Path(c['source'])/'tools/reports'/name):
            raise ValueError('declared KV implementation changed')
    return receipt,p


def audit(c,collection,handle):
    import torch
    from tools.reports.context_dual_probe_operations import check_runtime
    from tools.reports.run_context_kv_probe import compare
    receipt,p=validate(c);collection=Path(collection)
    launch=json.loads(Path(handle).read_text())
    inspect_path=Path(c['operations_root'])/(c['tag']+'-inspect.json')
    check_runtime(receipt,launch,json.loads(inspect_path.read_text()))
    transport_path=collection.with_suffix('.transport.json');tr=json.loads(transport_path.read_text())
    if (tr['status']!='transport_complete_independent_audit_required' or tr['state']['ExitCode']!=0
            or tr['state']['OOMKilled'] or tr['state']['Status']!='exited'
            or any(tr[k]!=launch[k] for k in ('container_id','image','remote_root'))
            or sha(collection.with_suffix('.tar'))!=tr['archive_sha256']):
        raise ValueError('complete exact successful transport required')
    if any(x.is_symlink() for x in collection.rglob('*')):raise ValueError('no symlinks')
    if {str(x.relative_to(collection)):sha(x) for x in collection.rglob('*') if x.is_file()}!=tr['files']:
        raise ValueError('collected bytes changed')
    if sha(collection/'config.json')!=sha(Path(c['prepared'])/'package/config.json'):
        raise ValueError('executed config changed')
    root=collection/'evaluation'
    def read(n):return json.loads((root/n).read_text())
    if read('source.json')['files_sha256']!=receipt['files_sha256']:raise ValueError('executed source differs')
    for name in ('terminal.json','baseline-exit.json','candidate-exit.json','baseline/terminal.json','candidate/terminal.json'):
        if read(name)['exit_code']!=0:raise ValueError('incomplete probe')
    reports={m:read(m+'/kv-measurement.json') for m in ('baseline','candidate')}
    for report in reports.values():
        if report['policy_id']!=p['policy']['policy_id'] or report['input_sha256']!=p['input_sha256']:
            raise ValueError('policy/input binding differs')
    result=compare(reports,torch.load(Path(c['prepared'])/'package/inputs.pt',map_location='cpu',weights_only=True),p)
    if result!=read('comparison.json'):raise ValueError('independent comparison differs')
    return dict(result,status='kv_probe_independently_compared',container_id=launch['container_id'],
        files_sha256=receipt['files_sha256'],transport_sha256=sha(transport_path))


def namespace(c):
    ns=parent_namespace(c);ns['validate']=validate;ns['audit']=audit
    return ns


def prepare(c,provenance):
    parent=json.loads(Path(c['parent_operations']).read_text());receipt,pc=parent_namespace(parent)['validate'](parent)
    dest=Path(c['output']);dest.mkdir(parents=True,exist_ok=False)
    package=dest/'package';shutil.copytree(Path(parent['prepared'])/'package',package)
    (package/'FILES.json').unlink()
    for name in ADDITIONS:shutil.copy2(Path(c['source'])/'tools/reports'/name,package/'tools/reports'/name)
    config=relocate(pc,parent['remote_root'],c['remote_root'])
    config['kv_probe']=dict(baseline_bytes=pc['policy']['kv_cache_memory_bytes'],
                            candidate_bytes=c['candidate_kv_cache_memory_bytes'])
    config['pins']={c['remote_root']+'/package/'+str(p.relative_to(package)):sha(p)
                   for p in package.rglob('*') if p.is_file() and p.name!='config.json'}
    (package/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    write_new(package/'FILES.json',{str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()})
    ops=dict(parent,prepared=str(dest),remote_root=c['remote_root'],tag=c['tag'],operations_root=c['operations_root'],
        source=c['source'],parent_operations=parent,package_manifest_sha256=verify(package),
        candidate_kv_cache_memory_bytes=c['candidate_kv_cache_memory_bytes'],
        finite_inference_dispatch_allowed=False)
    record=dict(receipt,status='sealed_kv_probe',remote_root=c['remote_root'],tag=c['tag'],
        files_sha256=verify(package),parent_manifest_sha256=receipt['files_sha256'],argv=argv(ops),config_provenance=provenance)
    write_new(dest/'PREPARED.json',record);validate(ops);write_new(dest/'operations.json',ops)
    return ops


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch'],required=True);a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else validate(c)[0] if a.action=='validate' else namespace(c)['launch'](c,provenance)
    print(json.dumps(result,indent=2))
