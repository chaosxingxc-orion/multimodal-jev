"""Bind a sealed common runtime to the real endpoint and four-device schedule.

This prepares the services used for numerical admission/reference generation.
It does not dispatch, build reference outputs, or release training. Python
source bytes remain identical to the common runtime throughout rebinding.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_model_profile import from_hf, validate_against_config
from multimodal_jev.context_replica_schedule import load_schedule
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.slurp_portable_release import load
from tools.reports.prepare_context_cached_runtime import audit as audit_runtime
from tools.reports.context_infonce_stage2_source import audit as audit_source
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.context_replica_training_operations import path


def read(p):
    return json.loads(Path(p).read_text())


def model_from_endpoint(c, initial, binding, runtime_identity):
    """Read model assets from the exact, independently audited Stage1 source."""
    spec=c.get('model_binding')
    if spec is None:
        return None
    if set(spec)!={'io_workers'}:
        raise ValueError('model binding requires explicit I/O width')
    plan=read(initial/'source-plan.json')
    training_path=Path(plan['source_operations']['package'])/'config.json'
    if file_hash(training_path)!=plan['training_config_sha256']:
        raise ValueError('Stage1 model configuration changed')
    training=read(training_path)
    contract=binding['source_contract']
    if (any(training[k]!=binding[k] for k in ('base_id','source_contract','release_sha256'))
            or training['runtime_profile']!=contract['runtime_profile']):
        raise ValueError('Stage1 model identity differs from endpoint')
    model=Path(path(training['model']))
    if model.is_symlink() or any(p.is_symlink() for p in model.rglob('*')):
        raise ValueError('ordinary immutable model files required')
    base_id,observed=checkpoint_identity(model,io_workers=spec['io_workers'])
    if base_id!=binding['base_id']:
        raise ValueError('Stage1 model bytes differ from endpoint base')
    pins={str(model/name):digest for name,digest in observed.items()}
    if any(training['pins'].get(p)!=digest for p,digest in pins.items()):
        raise ValueError('Stage1 model files differ from frozen training pins')
    profile=from_hf(model)
    validate_against_config(profile,read(initial/'design.json'))
    if (set(contract['lora'])!=set(profile.lora_targets()) or
            any(contract['branch'].get(k)!=v for k,v in profile.decision_head().items())):
        raise ValueError('Stage1 endpoint LoRA or decision head differs from model geometry')
    audio=read(model/'config.json')['audio_token_id']
    decision=contract['token']['id']
    if any(type(t)is not int or not 0<=t<profile.vocab_size for t in (audio,decision)) or audio==decision:
        raise ValueError('distinct in-vocabulary audio and decision token IDs required')
    if processor_cache_identity(initial/'policy')!=runtime_identity['processor_sha256']:
        raise ValueError('cross-model binding requires identical verified runtime processor')
    return dict(model=str(model),pins=pins,audio_token_id=audio,decision_token_id=decision)


def configured(original, services, c, binding, runtime, remote, model=None):
    """Deterministic rebind; no inherited topology/policy/optimizer cursor."""
    train=c['training_devices']; inference=c['inference_devices']
    devices=[*train,*inference]
    batch=binding['batch_config'];plan=derive(batch)
    if (len(train)!=batch['training_devices'] or len(train)!=plan['dp_replicas']
            or len(inference)!=batch['inference_instances'] or not inference or len(services)!=len(inference)
            or len(devices)!=len(set(devices)) or any(type(d)is not int or d<0 for d in devices)):
        raise ValueError('distinct physical devices must match declared actor and TP1 service widths')
    serving=c.get('serving')
    if serving is None:
        serving=read(Path(__file__).resolve().parents[2]/'configs/experiments/context-reference-serving-v1.json')
    if (set(serving)!={'max_tokens','max_decisions','max_batch_tokens','max_batch_queries'}
            or any(type(v) is not int or v<1 for v in serving.values())
            or serving['max_batch_queries']!=batch['inference_batch_queries']
            or serving['max_batch_tokens']<serving['max_tokens']):
        raise ValueError('explicit compatible service capacity required')
    if (original['release_sha256']!=binding['release_sha256']
            or (model is None and (original['base_id']!=binding['base_id']
                or original['source_contract']!=binding['source_contract']))):
        raise ValueError('runtime base/release/model interface differs from source')
    old=original['source_root'].removesuffix('/source')
    result=rebase(copy.deepcopy(original),old,remote)
    if model is not None:
        runtime_profile=binding['source_contract']['runtime_profile']
        if (original['runtime_profile']!=runtime_profile
                or original['context_runtime_profile']!=runtime_profile
                or original['source_contract']['request_version']!=binding['source_contract']['request_version']
                or original['source_contract']['output']!=binding['source_contract']['output']
                or original['audio_token_id']!=model['audio_token_id']
                or original['decision_token_id']!=model['decision_token_id']):
            raise ValueError('cross-model binding requires the same runtime and input/output interface')
        result.update(model=model['model'],base_id=binding['base_id'])
    for key in ('reference_cache','reference_dataset','resume_root','evaluation_steps',
                'experiment_kind','checkpoint_interval','socket_path','exchange'):
        result.pop(key,None)
    result.update(source_policy=remote+'/initial/policy',
        source_policy_id=binding['source_policy_id'],source_contract=binding['source_contract'],
        initial_adapter=remote+'/initial/adapter',reference_adapter=remote+'/initial/adapter',
        stage_lora_transition=binding['stage_lora_transition'],stage2_objective=binding['stage2_objective'],
        batch_config=batch,global_batch=plan['global_batch'],microbatch=batch['microbatch'],physical_devices=list(train),
        start_update=0,stop_update=2,experiment_kind='engineering_prefix',engineering_prefix=True,
        checkpoint_interval=1,training_dispatch_allowed=False,runtime_preparation_only=True,
        stage='task',algorithm=binding['stage2_objective']['algorithm'],dual_arm=binding['stage2_objective']['arm'],
        schedule=remote+'/source/schedule',schedule_sha256=c['schedule_sha256'],
        parent_schedule=remote+'/source/parent-schedule',parent_schedule_sha256=c['parent_schedule_sha256'],
        schedule_release_sha256=c['parent_release_sha256'],dual_runtime_identity=runtime)
    result['macro_client'].update(attempt_id=c['attempt_id'],max_queries=plan['macro_queries'],rollout_queries=batch['inference_batch_queries'],
        runtime_identity=runtime,journal_root=remote+'/run/pool-journal',
        replicas=[dict(socket=remote+'/service-'+n+'/socket',exchange=remote+'/service-'+n+'/exchange')
                  for n in sorted(services)])
    result['reference_source']=dict(endpoint=binding['source_step'],population=batch['source_queries'],max_tokens=serving['max_tokens'])
    if 'objective_config' in binding:
        result['objective_config']=binding['objective_config']
    if 'shared_inputs' in c:
        assets=c['shared_inputs']
        from multimodal_jev.context_decision_contract import PREFIX_LAYOUT
        for key in ('encoded_audio_cache','audio_mirror_root'):
            result[key]=path(assets[key])
        if assets['prompt_layout']!=PREFIX_LAYOUT:
            raise ValueError('explicit stable-prefix input layout required')
        result['prompt_layout']=assets['prompt_layout']
    if not isinstance(c['attempt_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',c['attempt_id']):
        raise ValueError('safe unique macro attempt id required')
    mounted=sorted(devices);values={}
    for (name,service),device in zip(sorted(services.items()),inference,strict=True):
        value=rebase(copy.deepcopy(service),old,remote)
        value.update(source_policy=result['source_policy'],dual_runtime_identity=runtime)
        value['policy'].update(snapshot=result['source_policy'],adapter=result['initial_adapter'],
            policy_id=binding['source_policy_id'],contract=binding['source_contract'],base_id=binding['base_id'],
            physical_devices=[device],tensor_parallel_size=1,max_batch_queries=serving['max_batch_queries'],max_batch_tokens=serving['max_batch_tokens'],
            max_decisions=serving['max_decisions'],process_device_scope=dict(kind='same_container_runtime_visibility_v1',
                mounted_physical=mounted,visible_ordinal=str(mounted.index(device))))
        if model is not None:
            value['policy']['model']=model['model']
        if 'attention_execution' in c:
            expected = dict(kind='bounded_prefill_v1',max_requests=8,acoustic_batch_chunks=32)
            if c['attention_execution'] != expected:
                raise ValueError('bounded attention8/acoustic32 execution required')
            value['attention_execution'] = copy.deepcopy(expected)
        if 'lora_arithmetic' in c:
            if c['lora_arithmetic'] != 'actor_order_bf16_qv_v1' or 'attention_execution' not in c:
                raise ValueError('actor BF16 Q/V arithmetic requires bounded prefill')
            value['lora_arithmetic'] = c['lora_arithmetic']
        values[name]=value
    return result,values,plan


def external_pins(original,c,remote,model=None):
    old_initial=Path(original['source_policy']).parent
    pins={p:h for p,h in original['pins'].items()
          if not Path(p).is_relative_to(remote) and not Path(p).is_relative_to(old_initial)}
    if model is not None:
        pins={p:h for p,h in pins.items() if not Path(p).is_relative_to(original['model'])}
        pins.update(model['pins'])
    if 'shared_inputs' in c:
        assets=c['shared_inputs']
        for key,old in [('encoded_audio_cache',Path(original['encoded_audio_cache']).parent),
                        ('audio_mirror_root',Path(original['audio_mirror_root']))]:
            root=Path(assets[key])
            if key=='encoded_audio_cache':root=root.parent
            expected=assets['manifests'][key]
            if file_hash(root/'MANIFEST.json')!=expected:
                raise ValueError('shared input manifest changed: '+key)
            pins={p:h for p,h in pins.items() if not Path(p).is_relative_to(old)}
            pins[str(root/'MANIFEST.json')]=expected
        manifest=read(Path(assets['encoded_audio_cache']).parent/'MANIFEST.json')
        if (manifest['prompt_layout']!=assets['prompt_layout']
                or manifest['release_sha256']!=original['release_sha256']
                or manifest['processor_sha256']!=original['dual_runtime_identity']['processor_sha256']):
            raise ValueError('shared input manifest differs from source runtime')
    return pins


def prepare(c,provenance,*, common_auditor=None,source_auditor=None):
    runtime_proof=(common_auditor or audit_runtime)(c['runtime'],provenance)
    source_proof=(source_auditor or audit_source)(c['source'],provenance)
    parent=Path(c['runtime']['package']);initial=Path(c['source']['output'])
    receipt=read(parent/'PREPARED.json');binding=read(initial/'binding.json')
    original=read(parent/'config.json');identity=receipt['dual_runtime_identity']
    if processor_cache_identity(initial/'policy')!=identity['processor_sha256']:
        raise ValueError('source processor differs from sealed runtime')
    remote=path(c['remote_root']);output=Path(c['output'])
    if (output.exists() or any(output.resolve().is_relative_to(p.resolve())
            or p.resolve().is_relative_to(output.resolve()) for p in (parent,initial,Path(c['schedule'])))
            or Path(remote).is_relative_to(receipt['remote_root'])
            or Path(receipt['remote_root']).is_relative_to(remote)):
        raise ValueError('fresh disjoint bound package required')
    services={n:read(parent/f) for n,f in receipt['layout']['services'].items()}
    model=model_from_endpoint(c,initial,binding,identity)
    config,services,plan=configured(original,services,c,binding,identity,remote,model)
    rows,_,_=load(original['release'],binding['release_sha256'],'learning',purpose='optimize')
    batches,schedule=load_schedule(c['schedule'],c['schedule_sha256'],
        release_sha256=binding['release_sha256'],population={r['id']:r for r in rows},
        batch_config=binding['batch_config'],parent_root=c['parent_schedule'],
        parent_manifest_sha256=c['parent_schedule_sha256'],parent_release_sha256=c['parent_release_sha256'])
    if len(rows)!=plan['configuration']['source_queries'] or len(batches)!=plan['total_updates']:
        raise ValueError('complete full TRAIN two-epoch schedule required')
    # New non-Python evidence may be bound; new code would invalidate reference identity.
    for root in (Path(c['schedule']),Path(c['parent_schedule']),initial):
        if any(p.is_symlink() or (p.is_file() and p.suffix=='.py') for p in root.rglob('*')):
            raise ValueError('regular non-code schedule/source assets required')
    check_paths({'bound_runtime':output});shutil.copytree(parent,output)
    (output/'PREPARED.json').unlink()
    for name,origin in [('schedule',c['schedule']),('parent-schedule',c['parent_schedule'])]:
        shutil.rmtree(output/'source'/name)
        shutil.copytree(origin,output/'source'/name)
    shutil.copytree(initial,output/'initial')
    commit_json(output/'SOURCE-BINDING-AUDIT.json',source_proof)
    commit_json(output/'COMMON-RUNTIME-AUDIT.json',runtime_proof)
    external=external_pins(original,c,receipt['remote_root'],model)
    names=['config.json',*receipt['layout']['services'].values()]
    pins=dict(external,**{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in names})
    for name,value in [('config.json',config),*[(receipt['layout']['services'][n],s) for n,s in services.items()]]:
        value['pins']=pins
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    actual_sources={str(p.relative_to(output/'source')):file_hash(p)
                    for p in (output/'source').rglob('*.py')}
    if actual_sources!=read(parent/'RUNTIME-SOURCES.json'):
        raise ValueError('source binding changed common runtime code')
    result=dict(receipt,status='stage2_endpoint_runtime_bound_reference_pending',remote_root=remote,
        parent_sha256=c['runtime']['manifest_sha256'],source_package_sha256=c['source']['manifest_sha256'],
        source_policy_id=binding['source_policy_id'],batch_plan=plan,training_devices=c['training_devices'],
        inference_services=[dict(name=n,device=d) for n,d in zip(sorted(services),c['inference_devices'],strict=True)],
        schedule_sha256=c['schedule_sha256'],experiment=dict(kind='engineering_prefix',updates=2),
        policy_provenance_admitted=True,numerical_admission=False,training_dispatch_allowed=False,
        finite_npu_dispatch_allowed=False,config_provenance=provenance,
        pending=['batch32 numerical admission','full TRAIN reference build/audit',
                 'readonly reference binding','DP2 NPU updates, tail and resume'])
    result['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',result)
    return dict(status=result['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
                runtime_identity=identity,training_dispatch_allowed=False)


def audit(c,provenance,*, common_auditor=None,source_auditor=None):
    """Reconstruct actual binding and schedule, not just a self-hashed manifest."""
    output=Path(c['output']);manifest=output/'PREPARED.json'
    if output.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('bound runtime manifest differs')
    receipt=read(manifest)
    actual={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
            if p.is_file() and p!=manifest}
    if (any(p.is_symlink() for p in output.rglob('*')) or actual!=receipt['files']
            or receipt['status']!='stage2_endpoint_runtime_bound_reference_pending'
            or receipt['training_dispatch_allowed'] is not False
            or receipt['finite_npu_dispatch_allowed'] is not False
            or receipt['numerical_admission'] is not False):
        raise ValueError('complete immutable preparation-only runtime required')
    runtime=(common_auditor or audit_runtime)(c['runtime'],provenance)
    source=(source_auditor or audit_source)(c['source'],provenance)
    parent=Path(c['runtime']['package']);initial=Path(c['source']['output'])
    common=read(parent/'PREPARED.json');original=read(parent/'config.json')
    services={n:read(parent/f) for n,f in common['layout']['services'].items()}
    binding=read(initial/'binding.json')
    model=model_from_endpoint(c,initial,binding,runtime['runtime_identity'])
    config,services,plan=configured(original,services,c,binding,
        runtime['runtime_identity'],c['remote_root'],model)
    if (receipt['remote_root']!=c['remote_root'] or receipt['batch_plan']!=plan
            or receipt['source_policy_id']!=source['source_policy_id']
            or receipt['parent_sha256']!=c['runtime']['manifest_sha256']
            or receipt['source_package_sha256']!=c['source']['manifest_sha256']
            or receipt['training_devices']!=c['training_devices']
            or receipt['inference_services']!=[dict(name=n,device=d) for n,d in
                zip(sorted(services),c['inference_devices'],strict=True)]):
        raise ValueError('bound manifest differs from independent origin/topology')
    for name in ('run.py','RUNTIME-SOURCES.json'):
        if (output/name).read_bytes()!=(parent/name).read_bytes():
            raise ValueError('common runtime entry/source identity changed')
    if {str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')} != read(parent/'RUNTIME-SOURCES.json'):
        raise ValueError('bound runtime code differs')
    if {str(p.relative_to(output/'initial')):file_hash(p) for p in (output/'initial').rglob('*') if p.is_file()} != {
            str(p.relative_to(initial)):file_hash(p) for p in initial.rglob('*') if p.is_file()}:
        raise ValueError('copied source package differs')
    rows,_,_=load(config['release'],config['release_sha256'],'learning',purpose='optimize')
    batches,_=load_schedule(output/'source/schedule',c['schedule_sha256'],
        release_sha256=config['release_sha256'],population={r['id']:r for r in rows},
        batch_config=config['batch_config'],parent_root=output/'source/parent-schedule',
        parent_manifest_sha256=c['parent_schedule_sha256'],parent_release_sha256=c['parent_release_sha256'])
    if len(rows)!=plan['configuration']['source_queries'] or len(batches)!=plan['total_updates']:
        raise ValueError('full two-epoch TRAIN schedule required')
    names=['config.json',*common['layout']['services'].values()]
    external=external_pins(original,c,common['remote_root'],model)
    pins=dict(external,**{c['remote_root']+'/'+name:digest for name,digest in actual.items() if name not in names})
    for name,value in [('config.json',config),*[(common['layout']['services'][n],s) for n,s in services.items()]]:
        value['pins']=pins
        if read(output/name)!=value:
            raise ValueError('configuration differs from independent binding: '+name)
    return dict(status='stage2_endpoint_runtime_binding_audited',manifest_sha256=c['manifest_sha256'],
        runtime_identity=runtime['runtime_identity'],source_policy_id=source['source_policy_id'],
        learning_population=len(rows),schedule_updates=len(batches),devices=len(c['training_devices'])+len(c['inference_devices']),
        numerical_admission=False,training_dispatch_allowed=False,npu_executed=False,provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
