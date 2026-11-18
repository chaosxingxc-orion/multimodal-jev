"""Bind a fresh process to audited step-one LoRA/head/Adam, never the base."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from tools.reports.audit_context_dual_training import audit,expected_contract
from tools.reports.prepare_context_dual_throughput_pair import sha,write
from tools.reports.context_dual_training_operations import path
from tools.jev.context_rl_evaluation_plan import bound


def relocate(value,old,new):
    if isinstance(value,str):
        return new+value[len(old):] if value==old or value.startswith(old+'/') else value
    if isinstance(value,list):return [relocate(v,old,new) for v in value]
    if isinstance(value,dict):return {relocate(k,old,new):relocate(v,old,new) for k,v in value.items()}
    return value


def seal(c,provenance,admission,source_audit):
    """Internal preparation after full audit; validate its bound assets again."""
    operations,_=load_config(source_audit['operations_config'],source_audit['operations_local'])
    package=Path(operations['package']);root=Path(source_audit['collection'])
    manifest=json.loads((package/'PREPARED.json').read_text());old=path(manifest['remote_root'])
    remote=path(c['remote_root']);output=Path(c['output'])
    if (output.exists() or output.resolve().is_relative_to(package.resolve()) or
            package.resolve().is_relative_to(output.resolve()) or Path(remote).is_relative_to(old) or
            Path(old).is_relative_to(remote)):
        raise ValueError('separate new replay package and runtime root required')
    if (admission['passed'] is not True or admission['schema']!='context-dual-prefix-audit-v1' or
            admission['manifest_sha256']!=sha(package/'PREPARED.json') or
            admission['files']['PREPARED.json']!=sha(package/'PREPARED.json')):
        raise ValueError('audited original prefix identity required')
    actual={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p!=package/'PREPARED.json'}
    if actual!=manifest['files'] or any(p.is_symlink() for p in package.rglob('*')):
        raise ValueError('original package changed')
    def read(name):return json.loads(bound(root,admission,name).read_text())
    config=read('config.json');service=read('service-config.json')
    if config['start_update']!=0 or config['stop_update']!=2 or config.get('resume_root'):
        raise ValueError('replay must descend directly from initial two-step prefix')
    commit=read('run/checkpoint-0001-committed.json');policy=read('run/policy-0001/MANIFEST.json')
    if (commit['updates']!=1 or commit['contract']!=expected_contract(config) or
            commit['joint_policy_directory']!='policy-0001' or
            commit['joint_policy_manifest_sha256']!=admission['files']['run/policy-0001/MANIFEST.json'] or
            commit['joint_policy']['policy_id']!=policy['policy_id'] or policy['contract']!=config['source_contract']):
        raise ValueError('committed first-step policy and optimizer contract required')
    expected={f'rank-{rank}/checkpoint-0001.pt' for rank in range(4)}
    if set(commit['rank_shards'])!=expected:raise ValueError('all four step-one shards required')
    required={'run/checkpoint-0001-committed.json'}
    for name,digest in commit['rank_shards'].items():
        if sha(bound(root,admission,'run/'+name))!=digest:raise ValueError('committed shard changed')
        required.add('run/'+name)
    for directory in ('run/policy-0001','run/adapter-0001'):
        names={name for name in admission['files'] if name.startswith(directory+'/')}
        if not names:raise ValueError('complete activated LoRA/head assets required')
        required.update(names)
    for name in required:bound(root,admission,name)
    resumed=relocate(config,old,remote);server=relocate(service,old,remote)
    resumed.update(start_update=1,resume_root=old+'/run',initial_adapter=old+'/run/adapter-0001')
    resumed['pins'].update({old+'/'+name:admission['files'][name] for name in required})
    server['pins']=copy.deepcopy(resumed['pins'])
    server['policy'].update(snapshot=old+'/run/policy-0001',adapter=resumed['initial_adapter'],policy_id=policy['policy_id'])
    if expected_contract(resumed)!=commit['contract']:raise ValueError('replay changed objective/runtime contract')
    runner=(package/'run.py').read_text();anchor="if c['start_update']!=0 or c['stop_update']!=2:"
    if runner.count(anchor)!=1:raise ValueError('original supervisor cursor guard changed')
    runner=runner.replace(anchor,"if c['start_update']!=1 or c['stop_update']!=2:")
    # Runtime source files/manifest remain byte-identical; only the supervisor
    # cursor and external saved-state bindings change.
    output.mkdir(parents=True);shutil.copytree(package/'source',output/'source')
    shutil.copy2(package/'RUNTIME-SOURCES.json',output/'RUNTIME-SOURCES.json')
    (output/'run.py').write_text(runner)
    # Future packages may pin the supervisor as well as Python runtime files.
    if remote+'/run.py' in resumed['pins']:
        resumed['pins'][remote+'/run.py']=sha(output/'run.py');server['pins']=copy.deepcopy(resumed['pins'])
    write(output/'config.json',resumed);write(output/'service-config.json',server)
    write(output/'PARENT-AUDIT.json',admission)
    record=dict(manifest,remote_root=remote,status='integrated_replay_prepared_only',
        config_provenance=provenance,replay=dict(parent_container=admission['container_id'],
            parent_manifest_sha256=admission['manifest_sha256'],parent_remote_root=old,
            parent_audit_sha256=sha(output/'PARENT-AUDIT.json'),
            initial_behavior_policy_id=policy['policy_id'],
            required_files={name:admission['files'][name] for name in sorted(required)}),
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        pending=['fresh-process NPU step2','independent replay comparison'],training_dispatch_allowed=False)
    write(output/'PREPARED.json',record);return record


def prepare(c,provenance):
    source,source_provenance=load_config(c['parent_audit_config'],c['parent_audit_local'])
    # Recompute the actual terminal audit; a stored passed flag is insufficient.
    admission=audit(source);admission['config_provenance']=source_provenance
    return seal(c,provenance,admission,source)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);result=prepare(c,provenance)
    print(json.dumps(dict(status=result['status'],files=len(result['files']),training_dispatch_allowed=False)))
