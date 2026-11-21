"""Prepare an independently pinned Stage2 window; never acquire devices or launch."""
import json
import os
from pathlib import Path
import shutil
import tempfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import regular
from multimodal_jev.stage2_view_runtime import indices, layout, pinned
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.context_replica_training_operations import path as canonical
from tools.reports.context_stage2_runtime_runner import plan
from tools.reports.context_objective_bootstrap import OBJECTIVE


ROOT=Path(__file__).resolve().parents[2]
COMPONENTS={'view-entry.py':'tools/reports/stage2_view_entry.py',
            'view-trainer.py':'tools/reports/stage2_view_actor.py',
            'task-reward.py':'src/multimodal_jev/context_task_reward.py',
            'view-runtime.py':'src/multimodal_jev/stage2_view_runtime.py',
            'run.py':'tools/reports/context_stage2_runtime_runner.py'}
WINDOWS=('prefix','recovery','tail','formal')


def validate(c):
    if set(c)-{'topology','assets'}!= {'schema','parent','bound_config','dataset_receipt','window','remote_root','output','objective_design','recovery','admission'} or c['schema']!='jev-stage2-view-package-v1' or c['window'] not in WINDOWS:
        raise ValueError('explicit Stage2 view preparation contract required')
    if bool(c['recovery'])!=(c['window']=='recovery'):raise ValueError('only recovery needs an audited prefix')
    if c['admission'] is not None and c['window']!='formal':raise ValueError('unexpected formal admission')
    canonical(c['remote_root']);canonical(c['output'])


def run(c,p):
    validate(c)
    parent=Path(c['parent']['root']);out=Path(c['output']);remote=c['remote_root']
    m=pinned(parent/'PREPARED.json',c['parent']['manifest_sha256'])
    def evidence(spec):return pinned(spec['path'],spec['sha256'])
    original=pinned(parent/'config.json',m['files']['config.json'])
    bound=evidence(c['bound_config']);receipt=evidence(c['dataset_receipt'])
    view=Path(bound['training_dataset']['root']);view_sha=bound['training_dataset']['manifest_sha256']
    if (receipt['bound_config_sha256']!=c['bound_config']['sha256'] or receipt['manifest_sha256']!=view_sha
            or receipt['status']!='stage2_dataset_release_audited' or receipt['training_dispatch_allowed'] is not False
            or original['start_update']!=0 or original['experiment_kind']!='engineering_prefix'
            or bound.get('training_dispatch_allowed') is not False or 'objective_extension' not in original):
        raise ValueError('same audited bound dataset and fresh objective parent required')
    vm=pinned(view/'MANIFEST.json',view_sha);consumer=pinned(view/'consumer.json',vm['files']['consumer.json'])
    if (file_hash(regular(parent,'initial/endpoint-commit.json'))!=vm['contract']['endpoint']['stage1_commit_sha256']
            or consumer!=receipt['consumer'] or consumer['reference_manifest_sha256']!=original['reference_dataset']['manifest_sha256']
            or consumer['source_policy_id']!=original['source_policy_id']
            or consumer['release_sha256']!=original['release_sha256']
            or vm['contract']['endpoint']['prompt_layout']!=original.get('prompt_layout')):
        raise ValueError('parent belongs to another Stage1 endpoint, reference or prompt layout')
    # Reconstruct binding from the exact pinned base, including ALL unrelated model/runtime options.
    from multimodal_jev.stage2_dataset_release import bind_config
    from types import SimpleNamespace
    ref=original['reference_dataset']
    proxy=SimpleNamespace(manifest_sha256=ref['manifest_sha256'],identity=ref['identity'],
                          population=ref['population'],policy_id=ref['identity']['policy_id'])
    if bind_config(original,root=view,manifest_sha256=view_sha,consumer=consumer,reference=proxy)!=bound:
        raise ValueError('bound training config does not reproduce from exact parent')
    old=m['remote_root']
    def overlap(a,b):
        a,b=Path(a).resolve(),Path(b).resolve()
        return a.is_relative_to(b) or b.is_relative_to(a)
    if out.exists() or any(overlap(out,q) for q in (view,parent,ROOT)) or any(overlap(remote,q) for q in (old,ref['root'])):
        raise ValueError('fresh disjoint package/output required')
    if old==remote:raise ValueError('fresh remote root required')
    if set(vm['files'])!={'selection.json','batches.json','consumer.json'}:raise ValueError('unexpected view release payload')
    design=evidence(c['objective_design']);source=regular(ROOT,design['objective_source'])
    from multimodal_jev.objective_plugin import ObjectivePlugin
    spec=dict(path=str(source),sha256=file_hash(source),options=design['options'])
    if 'objective_sha256' in design and design['objective_sha256'] != spec['sha256']:
        raise ValueError('objective source differs from pinned design')
    ObjectivePlugin(spec)
    topology = c.get('topology')
    if topology is not None:
        if set(topology) != {'training_devices', 'inference_services'}:
            raise ValueError('explicit training devices and inference services required')
        services = topology['inference_services']
        if ([service['name'] for service in services] != list(m['layout']['services'])
                or any(set(service) != {'name', 'device'} for service in services)):
            raise ValueError('inference service bindings must follow the parent service order')
        m = dict(m, **topology)
    from tools.reports.context_device_visibility import visibility
    _, service_ordinals = visibility(m)
    mounted = sorted([*m['training_devices'], *(s['device'] for s in m['inference_services'])])
    cfg=rebase(bound,old,remote)
    cfg['physical_devices'] = m['training_devices']
    assets = c.get('assets')
    if assets is not None:
        if set(assets) != {'encoded_audio_cache', 'audio_mirror_root'}:
            raise ValueError('explicit cache and mirror destinations required')
        for key, target in assets.items():
            canonical(target)
            source_path = cfg[key]
            old_root = str(Path(source_path).parent) if key == 'encoded_audio_cache' else source_path
            new_root = str(Path(target).parent) if key == 'encoded_audio_cache' else target
            if old_root == new_root:
                continue
            for name, digest in tuple(cfg['pins'].items()):
                if Path(name).is_relative_to(old_root):
                    relocated = str(Path(new_root) / Path(name).relative_to(old_root))
                    if cfg['pins'].get(relocated, digest) != digest or file_hash(relocated) != digest:
                        raise ValueError('relocated dependency differs: ' + relocated)
                    cfg['pins'][relocated] = cfg['pins'].pop(name)
            cfg[key] = target
    execution = design['execution']
    cfg['view_execution'] = evidence(dict(path=regular(ROOT, execution['source']), sha256=execution['sha256']))
    from multimodal_jev.stage2_view_runtime import validate_execution
    validate_execution(cfg['view_execution'])
    if 'objective_config' in cfg:
        if cfg['objective_config']['sha256']!=c['objective_design']['sha256']:
            raise ValueError('editable objective differs from the bound experiment')
    elif 'stage2_objective' in cfg:
        from multimodal_jev.stage2_view_runtime import validate_inherited_objective
        validate_inherited_objective(cfg)
    kind=c['window'];stage=kind=='formal';schedule=remote+'/view-data'
    cfg.update(view_window=kind,view_binding=dict(component='stage2_difficulty_release_v2',manifest_sha256=view_sha,
        selected_ids_sha256=consumer['selected_ids_sha256'],source_policy_id=consumer['source_policy_id'],
        reference_manifest_sha256=consumer['reference_manifest_sha256']),
        mirror_release_sha256=original.get('schedule_release_sha256',original['release_sha256']),
        schedule=schedule,training_dataset=dict(bound['training_dataset'],root=schedule),experiment_kind='formal_source' if stage else 'engineering_prefix',
        start_update=int(kind=='recovery'),stop_update=consumer['stop_update'] if stage else 2,
        engineering_prefix=not stage,training_dispatch_allowed=False,objective_trace=kind in ('recovery','prefix'),
        resume_root=remote+'/resume' if kind=='recovery' else None)
    cfg['objective_plugin'].update(sha256=spec['sha256'],options=spec['options'])
    cfg['objective_extension']['files'][OBJECTIVE]=spec['sha256']
    if not stage:cfg['checkpoint_interval']=1
    lineage=dict(parent_sha256=c['parent']['manifest_sha256'],dataset_sha256=view_sha,
        objective_design_sha256=c['objective_design']['sha256'],objective_sha256=spec['sha256'],
        runtime={name:file_hash(ROOT/relative) for name,relative in COMPONENTS.items()},
        assets=assets,
        topology=dict(training_devices=m['training_devices'], inference_services=m['inference_services']),
        plugin_api={n:file_hash(ROOT/'src/multimodal_jev'/n) for n in ('objective_plugin.py','objective_trace.py')})
    lineage['execution_sha256'] = execution['sha256']
    cfg['view_lineage']=lineage
    if 'objective_config' in cfg:
        from multimodal_jev.stage2_view_runtime import editable_objective
        cfg['objective_plugin']['numerics']=dict(
            advantage_std_floor=cfg['stage2_objective']['advantage_std_floor'],
            policy_clip=cfg['view_execution']['policy_clip'])
        editable_objective(cfg)
        ObjectivePlugin(dict(cfg['objective_plugin'],path=str(source)))
    if c['admission'] is not None:
        cfg['view_admission']=admission(c['admission'],lineage,world_size=layout(cfg)['dp_replicas'])
        cfg['training_dispatch_allowed']=True
    indices(cfg)
    if cfg['reference_dataset']!=ref:raise ValueError('rebasing changed the external fixed reference')
    check_paths({'package':out});out.parent.mkdir(parents=True,exist_ok=True)
    names=set(m['files'])
    if ({str(f.relative_to(parent)) for f in parent.rglob('*') if f.is_file()}!=names|{'PREPARED.json'}
            or any(f.is_symlink() for f in parent.rglob('*'))):raise ValueError('parent package membership differs')
    # A run-specific staging directory and exclusive ownership prevent concurrent publication.
    lock=out.parent/(out.name+'.prepare.lock')
    import fcntl
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():raise FileExistsError(out)
        with tempfile.TemporaryDirectory(dir=out.parent,prefix='.stage2-view-') as temp:
            dest=Path(temp)/'package';dest.mkdir()
            def copy_file(src,name,h):
                target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,target)
                if file_hash(target)!=h:raise ValueError('package source changed: '+name)
            for name,h in m['files'].items():copy_file(regular(parent,name),name,h)
            for mod in ('objective_plugin','objective_trace'):
                rel='src/multimodal_jev/'+mod+'.py';h=file_hash(ROOT/rel)
                copy_file(ROOT/rel,'objective-extension/'+rel,h)
                cfg['objective_extension']['files'][rel]=h
            for name,relative in COMPONENTS.items():copy_file(ROOT/relative,name,file_hash(ROOT/relative))
            copy_file(source,'objective-extension/'+OBJECTIVE,spec['sha256'])
            for name,h in dict(vm['files'],**{'MANIFEST.json':view_sha}).items():copy_file(regular(view,name),'view-data/'+name,h)
            services=m['layout']['services'];files=m['layout']['configs']
            initial=None
            if kind=='recovery':
                initial=restore_inputs(c['recovery'],dest,cfg)
            fullnames=set(COMPONENTS)|{'view-data/'+n for n in (*vm['files'],'MANIFEST.json')}
            cfg['pins'].update({remote+'/'+name:file_hash(dest/name) for name in fullnames|{'objective-extension/'+OBJECTIVE}})
            plan(remote,cfg,m,package_root=dest)
            for name in set(m['files'])-{'config.json',*services.values()}:
                if remote+'/'+name in cfg['pins']:cfg['pins'][remote+'/'+name]=file_hash(dest/name)
            def write(name,obj):(dest/name).write_text(json.dumps(obj,indent=2)+'\n')
            write('config.json',cfg)
            # Use the exact plugin API tested in this candidate. The forward
            # manifest and inference tree are intentionally never modified.
            devices = {service['name']: service['device'] for service in m['inference_services']}
            ordinals = dict(service_ordinals)
            for name, svcname in services.items():
                svc=rebase(json.loads((dest/svcname).read_bytes()),old,remote);svc['pins']=cfg['pins']
                if topology is not None:
                    svc['policy'].update(physical_devices=[devices[name]], tensor_parallel_size=1,
                                         max_batch_queries=cfg['batch_config']['inference_batch_queries'],
                                         process_device_scope=dict(
                        kind='same_container_runtime_visibility_v1',
                        mounted_physical=mounted, visible_ordinal=ordinals[name]))
                if initial:svc['policy'].update(initial)
                write(svcname,svc)
            manifest=dict(m,remote_root=remote,layout=dict(m['layout'],configs=sorted(set(files)|set(COMPONENTS))),
                status='stage2_view_package_prepared_not_admitted',parent_sha256=c['parent']['manifest_sha256'],
                view_window=kind,training_dispatch_allowed=False,dataset_sha256=view_sha,config_provenance=p)
            manifest['files']={str(x.relative_to(dest)):file_hash(x) for x in dest.rglob('*') if x.is_file()}
            commit_json(dest/'PREPARED.json',manifest)
            os.rename(dest,out);sync_directory(out.parent)
    return dict(manifest_sha256=file_hash(out/'PREPARED.json'),view_window=kind,npu_executed=False,training_dispatch_allowed=False)


def restore_inputs(spec,dest,cfg):
    from multimodal_jev.stage2_view_runtime import checkpoint_contract
    proof=pinned(spec['audit']['path'],spec['audit']['sha256'])
    run=Path(spec['run']);commit=pinned(run/'checkpoint-0001-committed.json',proof['step1_commit_sha256'])
    if (proof['lineage']!=cfg['view_lineage'] or proof['status']!='stage2_view_prefix_outputs_audited' or proof['dataset_sha256']!=cfg['schedule_sha256']
            or commit['updates']!=1 or commit['contract']['training_dataset']!=checkpoint_contract(cfg)
            or set(commit['rank_shards'])!={f'rank-{rank}/checkpoint-0001.pt' for rank in range(layout(cfg)['dp_replicas'])}):
        raise ValueError('recovery requires same-view prefix checkpoint')
    transport=pinned(spec['transport']['path'],spec['transport']['sha256'])
    if proof['transport_sha256']!=spec['transport']['sha256'] or proof['container_id']!=transport['container_id']:
        raise ValueError('prefix transport differs from independent audit')
    names={k[4:]:v for k,v in transport['files'].items() if k.startswith(('run/policy-0001/','run/adapter-0001/'))}
    names.update(commit['rank_shards']);names['checkpoint-0001-committed.json']=proof['step1_commit_sha256']
    if names.get('policy-0001/MANIFEST.json')!=commit['joint_policy_manifest_sha256'] or 'adapter-0001/adapter_model.safetensors' not in names:
        raise ValueError('incomplete trainable recovery assets')
    for name,h in names.items():
        src=regular(run,name);target=dest/'resume'/name;target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(src,target)
        if file_hash(target)!=h:raise ValueError('changed recovery input')
        cfg['pins'][cfg['resume_root']+'/'+name]=h
    cfg['initial_adapter']=cfg['resume_root']+'/adapter-0001'
    return dict(snapshot=cfg['resume_root']+'/policy-0001',adapter=cfg['initial_adapter'],policy_id=commit['joint_policy']['policy_id'])


def admission(spec,lineage,*,world_size=2):
    if set(spec)!={'prefix','recovery','tail'}:raise ValueError('three independent finite receipts required')
    ids=set()
    for kind,pin in spec.items():
        proof=pinned(pin['path'],pin['sha256']);cid=proof['container_id']
        if (proof['status']!='stage2_view_'+kind+'_outputs_audited' or proof['lineage']!=lineage
                or not cid or cid in ids or len(proof['updates'])!=(1 if kind=='recovery' else 2)
                or proof['world_size']!=world_size or not proof['storage_audit']['actual_readonly_mounts_verified']
                or (kind=='recovery' and (proof['fresh_process_replay_verified'] is not True
                    or proof['prefix_audit_sha256']!=spec['prefix']['sha256']
                    or proof['replay']['exact_objective_traces'] is not True))):
            raise ValueError('incomplete or foreign '+kind+' evidence')
        ids.add(cid)
    return {k:v['sha256'] for k,v in spec.items()}
