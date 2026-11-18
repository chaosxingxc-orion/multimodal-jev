"""Ordinary Stage2 packaging with an unchanged forward model and fixed reference."""
import copy
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from multimodal_jev.baseline import file_hash
from tools.reports.prepare_objective_training_candidate import sha, encode, read_bound
from tools.reports.context_objective_bootstrap import MODULES, OBJECTIVE
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports import prepare_context_cached_training as parent_api


def runtime_files():
    root=Path(__file__).parent
    return {name:(root/source).read_bytes() for name,source in (
        ('run.py','context_stage2_runtime_runner.py'),
        ('runtime-startup.py','context_runtime_startup.py'),
        ('device-visibility.py','context_device_visibility.py'))}


def expected(c,provenance):
    proof=parent_api.audit(c['parent'],provenance)
    if proof['status']!='cached_training_independently_audited_not_npu_admitted':
        raise ValueError('independently reconstructed cached parent required')
    parent=Path(c['parent']['output']);old=json.loads((parent/'PREPARED.json').read_text())
    config=json.loads((parent/'config.json').read_text());remote=c['remote_root']
    parent_api.path(remote)
    output=Path(c['output'])
    if (output.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(output.resolve())
            or Path(remote).is_relative_to(old['remote_root']) or Path(old['remote_root']).is_relative_to(remote)
            or Path(remote).is_relative_to(config['reference_dataset']['root'])
            or Path(config['reference_dataset']['root']).is_relative_to(remote)):
        raise ValueError('disjoint new training package and reference required')
    if (config['experiment_kind']!='engineering_prefix' or config['start_update']!=0
            or config['stop_update']!=2 or config.get('resume_root')):
        raise ValueError('initial two-update prefix parent required')
    report=json.loads(read_bound(c['extension_cpu_report']))
    snapshot=Path(c['extension_source'])
    manifest=json.loads((snapshot/'FILES.json').read_text())
    if (sha((snapshot/'FILES.json').read_bytes())!=report['source_manifest_sha256']
            or report['counts']!=dict(tests=47,failures=0,errors=0,skipped=0)
            or report['image']!=old['image'] or report['state']['ExitCode']!=0
            or report['state']['OOMKilled'] or report['terminal']['exit_code']!=0):
        raise ValueError('exact successful trace CPU evidence required')
    # The actual bound forward tree, not the current project, must be the one
    # tested with this extension. Missing inherited entrypoints are not ignored.
    for name,h in json.loads((parent/'RUNTIME-SOURCES.json').read_text()).items():
        if manifest.get(name)!=h:raise ValueError('CPU evidence used another forward dependency: '+name)
    additions={}
    for name in [p for _,p in MODULES]+[OBJECTIVE]:
        p=snapshot/name
        if p.is_symlink() or sha(p.read_bytes())!=manifest[name]:
            raise ValueError('tested extension changed: '+name)
        if (parent/'source'/name).exists():raise ValueError('extension overrides forward source')
        additions[name]=p.read_bytes()
    additions['bootstrap.py']=read_bound(c['bootstrap'])
    if not isinstance(c['options'],dict):raise ValueError('explicit objective options required')
    json.dumps(c['options'],allow_nan=False)
    files={n:(parent/n).read_bytes() for n in old['files']}
    for n,raw in additions.items():files['objective-extension/'+n]=raw
    files.update(runtime_files())
    config=rebase(config,old['remote_root'],remote)
    config.update(objective_trace=True,objective_plugin=dict(path=remote+'/objective-extension/'+OBJECTIVE,
        sha256=sha(additions[OBJECTIVE]),options=copy.deepcopy(c['options'])),
        objective_extension=dict(root=remote+'/objective-extension',
            files={n:sha(raw) for n,raw in additions.items()},
            forward_manifest_sha256=config['dual_runtime_identity']['source_manifest_sha256']))
    if config['reference_dataset']!=json.loads((parent/'config.json').read_text())['reference_dataset']:
        raise ValueError('fixed reference binding changed')
    files['OBJECTIVE-DEPENDENCIES.json']=encode(dict(parent_manifest_sha256=c['parent']['manifest_sha256'],
        extension_cpu_report_sha256=c['extension_cpu_report']['sha256'],bootstrap_sha256=c['bootstrap']['sha256'],
        forward_identity=config['dual_runtime_identity'],objective_extension=config['objective_extension'],
        parent_audit=parent_api.without_provenance(proof)))
    names={'config.json',*old['layout']['services'].values()}
    external={n:h for n,h in config['pins'].items() if not Path(n).is_relative_to(remote)}
    config['pins']=dict(external,**{remote+'/'+n:sha(raw) for n,raw in files.items() if n not in names})
    files['config.json']=encode(config)
    for name in old['layout']['services'].values():
        value=rebase(json.loads((parent/name).read_text()),old['remote_root'],remote)
        value['pins']=config['pins'];files[name]=encode(value)
    m=dict(old,status='objective_training_candidate_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],files={n:sha(raw) for n,raw in files.items()},
        config_provenance=provenance,training_dispatch_allowed=False,finite_npu_dispatch_allowed=False,
        numerical_admission=False,objective_npu_admitted=False,
        pending=['bootstrap fixed-image validation','NPU forward parity and two updates',
                 'integrated independent objective and optimizer audit','fresh-process recovery'])
    # Include the dependency declaration in collection; extension bytes already
    # belong to pins and must receive effective readonly mounts at launch.
    m['layout']=copy.deepcopy(old['layout'])
    m['layout']['configs'] += ['OBJECTIVE-DEPENDENCIES.json', 'runtime-startup.py', 'device-visibility.py']
    return files,m


from tools.reports.prepare_context_cached_recovery import read
from tools.reports.audit_context_replica_training import mounted_names

def recovery_inputs(c,provenance,*,api,parent_audit,contract):
    spec=c['parent_audit'];report=parent_audit(spec,provenance)
    root=Path(spec['training_package']['output']);m=read(root/'PREPARED.json')
    original=read(root/'config.json');run=Path(spec['collection'])/'run'
    if (report['status']!='finite_objective_training_outputs_audited'
            or report['world_size']!=2 or original['start_update']!=0 or original['stop_update']!=2
            or original['experiment_kind']!='engineering_prefix' or original.get('resume_root')
            or report['manifest_sha256']!=spec['training_package']['manifest_sha256']):
        raise ValueError('independently audited initial two-update DP2 cached prefix required')
    commit=read(run/'checkpoint-0001-committed.json')
    if (commit['updates']!=1 or commit['contract']!=contract(original,mounted_names(m))
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


from tools.reports import prepare_objective_training_candidate as prefix
from tools.reports import context_cached_formal as formal
from tools.reports import prepare_context_cached_training as cached


def tail_entry():
    # Historical tested tail trainer is an explicit dependency. This wrapper
    # changes no source: the model integration remains pinned and audited.
    from tools.reports.prepare_stage2_objective_run import TAIL_ENTRY
    return TAIL_ENTRY


def stage_expected(c,provenance):
    prefix.audit(c['prefix_package'],provenance)
    parent=Path(c['prefix_package']['output']);old=read(parent/'PREPARED.json');remote=c['remote_root'];cached.path(remote)
    output=Path(c['output'])
    config=read(parent/'config.json');kind=c['kind']
    if (kind not in ('tail','efficiency','formal') or config['objective_plugin']['options']!={}
            or file_hash(parent/'objective-extension'/OBJECTIVE)!=file_hash(Path(__file__).parents[2]/OBJECTIVE)):
        raise ValueError('default accepted objective and explicit run kind required')
    if (output.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(output.resolve())
            or any(Path(remote).is_relative_to(p) or Path(p).is_relative_to(remote)
                for p in [old['remote_root'],config['reference_dataset']['root']])):
        raise ValueError('disjoint fresh run and readonly reference required')
    files={n:(parent/n).read_bytes() for n in old['files']}
    config=rebase(config,old['remote_root'],remote)
    if kind=='formal':
        config.update(experiment_kind='formal_source',engineering_prefix=False,training_dispatch_allowed=True,
            stop_update=634,evaluation_steps=list(formal.STEPS),objective_trace=False)
        config.pop('checkpoint_interval',None);formal.contract(config)

    elif kind=='efficiency':
        config.update(experiment_kind='efficiency',stop_update=100,objective_trace=False)

    else:
        config.update(experiment_kind='cached_reference_tail_admission',objective_trace=False)
        from multimodal_jev.context_cached_tail_window import contract
        contract(config)
        # Keep accepted extension modules byte-identical. This outer wrapper
        # derives the same plugin generator with the original tail generator.
        files['objective-tail-entry.py']=tail_entry().encode()
    files.update(runtime_files())
    names={'config.json',*old['layout']['services'].values()}
    pins={n:h for n,h in config['pins'].items() if not Path(n).is_relative_to(remote)}
    pins.update({remote+'/'+n:prefix.sha(raw) for n,raw in files.items() if n not in names});config['pins']=pins
    files['config.json']=prefix.encode(config)
    for name in old['layout']['services'].values():
        value=rebase(read(parent/name),old['remote_root'],remote);value['pins']=pins;files[name]=prefix.encode(value)
    m=copy.deepcopy(old);m.update(status='stage2_objective_run_prepared_only',remote_root=remote,
        parent_sha256=c['prefix_package']['manifest_sha256'],config_provenance=provenance,
        experiment=dict(kind=kind,updates=config['stop_update']),files={n:prefix.sha(raw) for n,raw in files.items()},
        numerical_admission=False,training_dispatch_allowed=False)
    if kind=='tail':m['layout']['configs'].append('objective-tail-entry.py')
    return files,m

