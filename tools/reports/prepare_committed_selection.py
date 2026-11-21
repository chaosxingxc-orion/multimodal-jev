"""Retarget the pinned evaluation runtime to a sealed committed GRPO policy.

Only inference admission. Full training audit proceeds independently; no
selection result may be published as final until it passes that separate gate.
"""
import copy
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.prepare_objective_training_candidate import read_bound,encode,sha
from tools.reports.context_completed_stage_receipts import check_tree


def read(p):return json.loads(Path(p).read_text())


def source(c):
    report=json.loads(read_bound(c['assets_report']));root=Path(report['root']);step=c['step']
    if type(step)is not int or step not in [*range(64,634,64),634]:raise ValueError('registered GRPO checkpoint required')
    launch=json.loads(read_bound(c['training_launch']))
    if report.get('status')!='sealed_subset_collected_training_audit_pending' or report.get('container_id')!=launch['container_id']:
        raise ValueError('sealed source container identity required')
    names=[f'run/checkpoint-{step:04d}-committed.json',f'run/policy-{step:04d}-committed.json']
    prefixes=[f'run/policy-{step:04d}/',f'run/adapter-{step:04d}/']
    if set(names)-report['files'].keys():raise ValueError('commit missing from sealed manifest')
    for prefix in prefixes:
        directory=root/prefix
        actual={str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file()}
        declared={n for n in report['files'] if n.startswith(prefix)}
        if not actual or actual!=declared:raise ValueError('complete sealed policy directory required')
    for n,h in report['files'].items():
        if n in names or any(n.startswith(p) for p in prefixes):
            if any(p.is_symlink() for p in [root/n,*(root/n).parents]) or file_hash(root/n)!=h:raise ValueError('committed asset changed')
    commit=read(root/names[0]);publication=read(root/names[1])
    training=json.loads(read_bound(c['training_config']))
    if (commit['updates']!=step or {k:v for k,v in commit.items() if k!='rank_shards'}!=publication
            or len(commit['rank_shards'])!=2 or commit['joint_policy_directory']!=f'policy-{step:04d}'):
        raise ValueError('matching policy and optimizer commit required')
    snapshot=root/prefixes[0];adapter=root/prefixes[1];policy=commit['joint_policy']['policy_id']
    if file_hash(snapshot/'MANIFEST.json')!=commit['joint_policy_manifest_sha256']:raise ValueError('committed policy changed')
    bundle=read_execution_bundle(snapshot,expected_policy_id=policy,expected_contract=training['source_contract'],expected_base_id=training['base_id'])
    verify_native_adapter(adapter,bundle)
    return training,snapshot,adapter,policy


def expected(c,provenance):
    parent=Path(c['template']['root']);m=json.loads(read_bound(dict(path=str(parent/'PREPARED.json'),sha256=c['template']['manifest_sha256'])))
    check_tree(parent,dict(m['files'],**{'PREPARED.json':c['template']['manifest_sha256']}))
    training,snapshot,adapter,policy=source(c);old=m['remote_root'];remote=c['remote_root']
    if not Path(remote).is_absolute() or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote):raise ValueError('fresh remote root required')
    config=read(parent/'config.json')
    for k in ('base_id','source_contract','dual_runtime_identity','release_sha256'):
        if config[k]!=training[k]:raise ValueError('evaluation runtime differs: '+k)
    files={n:(parent/n).read_bytes() for n in m['files'] if not n.startswith(('initial/policy/','initial/adapter/'))}
    for name,root in [('policy',snapshot),('adapter',adapter)]:
        for p in root.rglob('*'):
            if p.is_file():files['initial/'+name+'/'+str(p.relative_to(root))]=p.read_bytes()
    files['run.py']=Path(__file__).with_name('context_selection_runner.py').read_bytes()
    files['runtime-startup.py']=Path(__file__).with_name('context_runtime_startup.py').read_bytes()
    files['device-visibility.py']=Path(__file__).with_name('context_device_visibility.py').read_bytes()
    # This evidence describes the new policy. Historical template source-plan
    # files remain pinned construction inputs, never its training audit.
    files['COMMITTED-SOURCE.json']=encode(dict(step=c['step'],policy_id=policy,assets_report=c['assets_report'],training_config=c['training_config'],training_audit_pending=True))
    values={n:rebase(read(parent/n),old,remote) for n in ['config.json','parity-hf.json','parity-macro.json',*m['layout']['services'].values()]}
    for n,v in values.items():
        if n in m['layout']['services'].values():
            v['policy'].update(policy_id=policy)
        else:
            v['source_policy_id']=policy
            if 'policy_id' in v:v['policy_id']=policy
            if 'source_step' in v:v['source_step']=c['step']
    names=set(values)|{'parity-hf-local.json'}
    pins={n:h for n,h in values['config.json']['pins'].items() if not Path(n).is_relative_to(remote)}
    pins.update({remote+'/'+n:sha(raw) for n,raw in files.items() if n not in names})
    for v in values.values():v['pins']=pins
    values['parity-hf-local.json']=dict(schema='jev-local-macros-v1',macros={'REPLICA_SELECTION_CLIENT_CONFIG':values['parity-hf.json']})
    for n,v in values.items():files[n]=encode(v)
    m=copy.deepcopy(m);m.update(status='committed_grpo_selection_prepared_training_audit_pending',remote_root=remote,
        source_step=c['step'],source_policy_id=policy,config_provenance=provenance,parent_sha256=c['template']['manifest_sha256'],
        training_audit_pending=True)
    m['layout']['configs']+=['runtime-startup.py','device-visibility.py','COMMITTED-SOURCE.json']
    m['files']={n:sha(raw) for n,raw in files.items()}
    return files,m


def prepare(c,p):
    root=Path(c['output']);check_paths({'package':root});files,m=expected(c,p);root.mkdir(parents=True,exist_ok=False)
    for n,raw in files.items():
        f=root/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    commit_json(root/'PREPARED.json',m);return file_hash(root/'PREPARED.json')


def audit(c,p):
    root=Path(c['output']);m=read(root/'PREPARED.json')
    if file_hash(root/'PREPARED.json')!=c['manifest_sha256']:raise ValueError('prepared identity changed')
    files,wanted=expected(c,m['config_provenance'])
    if m!=wanted:raise ValueError('prepared metadata differs')
    check_tree(root,dict({n:sha(raw) for n,raw in files.items()},**{'PREPARED.json':c['manifest_sha256']}))
    return dict(status='committed_selection_content_verified',step=c['step'],policy_id=m['source_policy_id'],training_audited=False)
