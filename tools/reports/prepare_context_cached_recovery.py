"""Bind a fresh DP2 step2 replay to independently audited cached GRPO updates.

Reference remains the original Stage1 endpoint. Only behavior/optimizer resume
from step1. Copy trainable artifacts, never base weights or runtime Python.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_cached_training_operations import audit as audit_parent
from tools.reports.audit_context_cached_training import expected_contract
from tools.reports.audit_context_replica_training import mounted_names
from tools.reports.prepare_context_cached_training import runner, without_provenance
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.context_replica_training_operations import path


def read(p):return json.loads(Path(p).read_text())


def replay_runner(manifest):
    source=runner(manifest)
    before="if c['start_update']!=0 or c['stop_update']!=2:"
    after="if c['start_update']!=1 or c['stop_update']!=2 or not c.get('resume_root'):"
    if source.count(before)!=1:raise ValueError('shared replay gate anchor changed')
    return source.replace(before,after)


def inputs(c,provenance):
    spec=c['parent_audit'];report=audit_parent(spec,provenance)
    root=Path(spec['training_package']['output']);m=read(root/'PREPARED.json')
    original=read(root/'config.json');run=Path(spec['collection'])/'run'
    if (report['status']!='finite_cached_reference_training_outputs_audited'
            or report['world_size']!=2 or original['start_update']!=0 or original['stop_update']!=2
            or original['experiment_kind']!='engineering_prefix' or original.get('resume_root')
            or report['manifest_sha256']!=spec['training_package']['manifest_sha256']):
        raise ValueError('independently audited initial two-update DP2 cached prefix required')
    commit=read(run/'checkpoint-0001-committed.json')
    if (commit['updates']!=1 or commit['contract']!=expected_contract(original,mounted_names(m))
            or set(commit['rank_shards'])!={f'rank-{r}/checkpoint-0001.pt' for r in range(2)}
            or commit['joint_policy_directory']!='policy-0001'
            or commit['joint_policy_manifest_sha256']!=file_hash(run/'policy-0001/MANIFEST.json')):
        raise ValueError('exact first-step DP2 optimizer and policy contract required')
    transport=read(spec['transport'])['files']
    names={'checkpoint-0001-committed.json',*commit['rank_shards']}
    for directory in ('policy-0001','adapter-0001'):
        selected={n[4:] for n in transport if n.startswith('run/'+directory+'/')}
        if not selected:raise ValueError('complete step1 joint policy and native adapter required')
        names.update(selected)
    for name in names:
        p=Path(name)
        if (p.is_absolute() or '..' in p.parts or p.suffix=='.py' or (run/p).is_symlink()
                or file_hash(run/p)!=transport['run/'+name]):
            raise ValueError('unchanged trainable recovery assets required')
    for name,h in commit['rank_shards'].items():
        if file_hash(run/name)!=h:raise ValueError('committed rank shard differs')
    return root,m,original,run,report,commit,sorted(names)


def configured(original,manifest,remote,commit):
    c=rebase(original,manifest['remote_root'],remote)
    c.update(start_update=1,stop_update=2,resume_root=remote+'/resume',
             initial_adapter=remote+'/resume/adapter-0001')
    # The fixed reference must never be rebound to the recovered behavior.
    if c['reference_dataset']!=original['reference_dataset']:
        raise ValueError('recovery changed fixed reference dataset')
    if expected_contract(c,mounted_names(manifest))!=commit['contract']:
        raise ValueError('recovery changed objective, runtime or optimizer contract')
    return c


def paths(c,parent,collection):
    output=Path(c['output']);remote=path(c['remote_root'])
    old=read(parent/'PREPARED.json')['remote_root']
    if (any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
            for p in (parent,collection))
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('disjoint new recovery package and runtime required')
    return output,remote


def payload(original,m,remote,commit,files):
    config=configured(original,m,remote,commit)
    names={'config.json',*m['layout']['services'].values()}
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(m['remote_root'])}
    config['pins']=dict(external,**{remote+'/'+n:h for n,h in files.items() if n not in names})
    return config


def prepare(c,provenance):
    parent,m,original,run,proof,commit,names=inputs(c,provenance)
    output,remote=paths(c,parent,run.parent)
    if output.exists():raise ValueError('new recovery package required')
    configured(original,m,remote,commit)
    check_paths({'recovery_package':output});shutil.copytree(parent,output)
    (output/'PREPARED.json').unlink()
    for name in names:
        dest=output/'resume'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(run/name,dest)
    commit_json(output/'RECOVERY-PARENT-AUDIT.json',proof)
    (output/'run.py').write_text(replay_runner(m))
    files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    config=payload(original,m,remote,commit,files)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in m['layout']['services'].values():
        service=rebase(read(parent/name),m['remote_root'],remote);service['pins']=config['pins']
        service['policy'].update(snapshot=remote+'/resume/policy-0001',adapter=config['initial_adapter'],
                                 policy_id=commit['joint_policy']['policy_id'])
        (output/name).write_text(json.dumps(service,indent=2)+'\n')
    receipt=dict(m,status='cached_recovery_prepared_only',remote_root=remote,
        parent_container_id=proof['container_id'],
        parent_manifest_sha256=c['parent_audit']['training_package']['manifest_sha256'],
        initial_behavior_policy_id=commit['joint_policy']['policy_id'],
        recovery_files=names,config_provenance=provenance,
        training_dispatch_allowed=False,finite_npu_dispatch_allowed=False,numerical_admission=False,
        pending=['fresh-process NPU step2','independent output and exact continuous/replay comparison'])
    receipt['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',receipt)
    return dict(status=receipt['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
                training_dispatch_allowed=False)


def audit(c,provenance):
    parent,pm,original,run,proof,commit,names=inputs(c,provenance)
    root,remote=paths(c,parent,run.parent);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('recovery manifest differs')
    m=read(manifest)
    files={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if any(p.is_symlink() for p in root.rglob('*')) or files!=m['files']:
        raise ValueError('complete immutable recovery package required')
    expected=dict(pm,status='cached_recovery_prepared_only',remote_root=remote,
        parent_container_id=proof['container_id'],
        parent_manifest_sha256=c['parent_audit']['training_package']['manifest_sha256'],
        initial_behavior_policy_id=commit['joint_policy']['policy_id'],recovery_files=names,
        config_provenance=m['config_provenance'],training_dispatch_allowed=False,
        finite_npu_dispatch_allowed=False,numerical_admission=False,files=files,
        pending=['fresh-process NPU step2','independent output and exact continuous/replay comparison'])
    if m!=expected:raise ValueError('recovery source/topology/behavior binding differs')
    originals={str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*')
               if p.is_file() and p!=parent/'PREPARED.json'}
    if set(files)!=set(originals)|{'RECOVERY-PARENT-AUDIT.json'}|{'resume/'+n for n in names}:
        raise ValueError('unexpected recovery assets')
    changed={'run.py','config.json',*pm['layout']['services'].values()}
    if any(files[n]!=h for n,h in originals.items() if n not in changed):
        raise ValueError('recovery changed shared runtime or Stage1 assets')
    if any(files['resume/'+n]!=file_hash(run/n) for n in names):
        raise ValueError('recovery state differs from audited original')
    if without_provenance(read(root/'RECOVERY-PARENT-AUDIT.json'))!=without_provenance(proof):
        raise ValueError('parent audit differs')
    config=payload(original,pm,remote,commit,files)
    if read(root/'config.json')!=config or (root/'run.py').read_text()!=replay_runner(pm):
        raise ValueError('recovery configuration differs')
    for name in pm['layout']['services'].values():
        service=rebase(read(parent/name),pm['remote_root'],remote);service['pins']=config['pins']
        service['policy'].update(snapshot=remote+'/resume/policy-0001',adapter=config['initial_adapter'],
                                 policy_id=commit['joint_policy']['policy_id'])
        if read(root/name)!=service:raise ValueError('service must start from recovered behavior')
    return dict(status='cached_recovery_independently_bound_not_npu_admitted',
        manifest_sha256=c['manifest_sha256'],parent_container_id=proof['container_id'],
        source_policy_id=original['source_policy_id'],initial_behavior_policy_id=m['initial_behavior_policy_id'],
        actor_updates=1,training_dispatch_allowed=False,provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
