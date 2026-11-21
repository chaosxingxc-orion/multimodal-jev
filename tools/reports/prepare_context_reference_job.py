"""Prepare a full TRAIN reference job on the sealed shared Stage2 runtime.

The package includes a supervised service/client entry, never a training actor.
Numerical admission and remote lifecycle are separate required release gates.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from string import Template
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_builder import validate_prefetch
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load
from tools.reports.context_reference_input_evidence import load as input_evidence
from tools.reports.prepare_context_stage2_runtime import audit as audit_parent
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.context_replica_training_operations import path
from tools.reports.context_device_visibility import visibility
from tools.reports.context_replica_runtime import shutdown_timeout as validate_shutdown_timeout


_RUNNER_TEMPLATE = Template('''import json,os,sys
from pathlib import Path
from context_process_supervisor import supervise_services
from multimodal_jev.local_runtime_storage import require_training_storage
r=Path(__file__).resolve().parent
require_training_storage(r)
c=json.loads((r/'reference-local.json').read_text())['macros']['REFERENCE_DATASET_BUILD_CONFIG']
commit=json.loads((r/'initial/endpoint-commit.json').read_bytes())
if (commit['updates']!=c['stage1_endpoint'] or
    commit['contract']['batch_config']['source_queries']!=c['expected_population']):
 raise ValueError('complete TRAIN endpoint reference only')
def env(role,visible):
 e=dict(os.environ);e.pop('ASCEND_RT_VISIBLE_DEVICES',None)
 cache=Path('/jev-tmp')/role;cache.mkdir(exist_ok=True)
 e['VLLM_RPC_BASE_PATH']=str(cache)
 for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR',
             'TRITON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
  e[key]=str(cache)
 e['ASCEND_RT_VISIBLE_DEVICES']=visible
 return e
services=[]
for name,visible in __SERVICES__:
 service=json.loads((r/('service-'+name+'.json')).read_text())
 services.append(dict(name=name,
  command=[sys.executable,str(r/$service_entry),
   '--config',str(r/('service-'+name+'.json')),'--output',str(r/('service-'+name))$service_args],
  env=env('service-'+name,visible),ready_path=r/('service-'+name)/'ready.json',
  socket_path=Path(service['socket_path'])))
report=supervise_services(services=services,
 actor_command=[sys.executable,str(r/$actor_entry),
  '--config',str(r/'reference-template.json'),'--local',str(r/'reference-local.json')],
 actor_env=env('reference-client',''),output=r/'supervision')
raise SystemExit(0 if report['passed'] else 1)
''')

_SERVICE_ENTRY = 'source/tools/reports/context_replica_runtime.py'
RUNNER = _RUNNER_TEMPLATE.substitute(actor_entry=repr('source/tools/reports/build_context_reference_dataset.py'),
                                    service_entry=repr(_SERVICE_ENTRY), service_args='')

def reference_runner(devices, shutdown_timeout=None, *, resume=False, legacy_engine_timeout=None):
    if legacy_engine_timeout is not None:
        legacy_engine_timeout = validate_shutdown_timeout(legacy_engine_timeout)
        if (not resume or validate_shutdown_timeout(shutdown_timeout) <= legacy_engine_timeout):
            raise ValueError('legacy engine deadline requires supervised resume and shutdown margin')
    source = _RUNNER_TEMPLATE.substitute(actor_entry=repr('resume_reference.py'),
        service_entry=repr('resume-service.py' if legacy_engine_timeout is not None else _SERVICE_ENTRY),
        service_args=(", '--engine-timeout', " + repr(str(legacy_engine_timeout))
                      if legacy_engine_timeout is not None else '')) if resume else RUNNER
    return with_shutdown_timeout(source.replace('__SERVICES__',repr(devices)), shutdown_timeout)


def with_shutdown_timeout(source, shutdown_timeout=None):
    if shutdown_timeout is None:
        return source
    return source.replace("output=r/'supervision')",
        "output=r/'supervision',shutdown_timeout="+repr(shutdown_timeout)+")")


def build_config(training, remote, evidence, *, prefetch=None):
    keys=('source_root','source_policy','source_policy_id','source_contract','base_id',
          'release','release_sha256','audio_token_id','encoded_audio_cache',
          'dual_runtime_identity','dual_runtime_manifest','macro_client')
    c={k:copy.deepcopy(training[k]) for k in keys}
    pool=c['macro_client']
    replicas=len(pool['replicas'])
    if (not replicas or any(type(pool[k]) is not int or pool[k]<=0 for k in ('max_queries','rollout_queries'))
            or pool['max_queries']<pool['rollout_queries']
            or pool['runtime_identity']!=c['dual_runtime_identity']):
        raise ValueError('declared TRAIN scoring pool required')
    pool['attempt_id']='reference-'+pool['attempt_id']
    pool['journal_root']=remote+'/reference-journal'
    c.update(stage1_commit=remote+'/initial/endpoint-commit.json',stage1_endpoint=training['reference_source']['endpoint'],
        expected_population=training['reference_source']['population'],max_tokens=training['reference_source']['max_tokens'],output=remote+'/reference-build',
        audio_mirror_root=str(Path(training['audio_mirror_root'])/'audio'))
    if training.get('prompt_layout') is not None:
        c['prompt_layout']=training['prompt_layout']
    if prefetch is not None:
        c['input_prefetch']=validate_prefetch(prefetch,pool['max_queries'])
    for name in ('input_proof','input_receipts'):
        c[name]=remote+'/reference-inputs/'+('proof.json' if name=='input_proof' else 'receipts.jsonl')
        c[name+'_sha256']=evidence[name+'_sha256']
    return c


def scoring_binding(spec, parent, manifest, training, remote):
    """Bind inference-only devices without changing the inherited training config."""
    old=manifest['remote_root']
    layout=copy.deepcopy(manifest)
    config=rebase(training,old,remote)
    services={name:rebase(json.loads((parent/file).read_text()),old,remote)
              for name,file in manifest['layout']['services'].items()}
    if 'engine_shutdown_timeout_seconds' in spec or (
            'shutdown_timeout_seconds' in spec and 'legacy_engine_timeout_seconds' not in spec):
        if 'legacy_engine_timeout_seconds' in spec:
            raise ValueError('only one configured engine deadline is allowed')
        engine=validate_shutdown_timeout(spec.get('engine_shutdown_timeout_seconds'))
        supervisor=validate_shutdown_timeout(spec.get('shutdown_timeout_seconds'))
        if supervisor <= engine:
            raise ValueError('supervisor shutdown deadline must exceed the engine deadline')
        for service in services.values():
            service['engine_shutdown_timeout_seconds']=engine
    if 'scoring_devices' not in spec:
        return layout,config,services
    devices=spec['scoring_devices']
    if (not isinstance(devices,list) or not devices or len(set(devices))!=len(devices)
            or any(type(d) is not int or d<0 for d in devices)):
        raise ValueError('distinct TRAIN scoring devices required')
    template=services[sorted(services)[0]]
    name=sorted(services)[0]
    services={}
    layout['training_devices']=[]
    layout['inference_services']=[]
    layout['layout']['services']={}
    config['physical_devices']=[]
    mounted=sorted(devices)
    replicas=[]
    for index,device in enumerate(devices):
        replica=chr(ord('a')+index)
        endpoint=remote+'/service-'+replica
        value=rebase(copy.deepcopy(template),remote+'/service-'+name,endpoint)
        value['policy'].update(physical_devices=[device],process_device_scope=dict(
            kind='same_container_runtime_visibility_v1', mounted_physical=mounted,
            visible_ordinal=str(mounted.index(device))))
        services[replica]=value
        layout['inference_services'].append(dict(name=replica,device=device))
        layout['layout']['services'][replica]='service-'+replica+'.json'
        replicas.append(dict(socket=value['socket_path'],exchange=value['exchange']))
    config['macro_client'].update(replicas=replicas,
        max_queries=config['macro_client']['rollout_queries']*len(devices))
    return layout,config,services


def resume_assets(c, output, build, remote, *, prepare=False):
    resume=c.get('resume')
    if resume is None:
        return {}
    from tools.reports.context_reference_resume_release import bind_assets
    return bind_assets(resume, output, build, remote, prepare=prepare,
                       legacy_service='legacy_engine_timeout_seconds' in c)


def reference_source(parent, original):
    from multimodal_jev.context_topology_batch_plan import derive
    commit=json.loads((parent/'initial/endpoint-commit.json').read_bytes())
    batch=commit['contract']['batch_config']
    if commit['updates']!=derive(batch)['total_updates'] or commit['joint_policy']['policy_id']!=original['source_policy_id']:
        raise ValueError('complete Stage1 endpoint from the bound policy required')
    # The input cache determines the accepted sequence capacity.
    spec=dict(endpoint=commit['updates'],population=batch['source_queries'],max_tokens=original['reference_source']['max_tokens'])
    if spec!=original['reference_source']:
        raise ValueError('declared reference source differs from committed Stage1')
    return spec


def prepare(c,provenance):
    proof=audit_parent(c['parent'],provenance)
    parent=Path(c['parent']['output']);m=json.loads((parent/'PREPARED.json').read_text())
    original=json.loads((parent/'config.json').read_text())
    evidence=dict(c['input_evidence'],prompt_layout=original.get('prompt_layout'))
    rows,task,_=load(original['release'],original['release_sha256'],'learning',purpose='optimize')
    _,input_proof=input_evidence(evidence,rows=rows,task=task,release_sha256=original['release_sha256'],
        processor_sha256=proof['runtime_identity']['processor_sha256'])
    if len(rows)!=reference_source(parent,original)['population']:raise ValueError('full TRAIN reference population required')
    output=Path(c['output']);remote=path(c['remote_root']);old=m['remote_root']
    if (output.exists() or output.resolve().is_relative_to(parent.resolve())
            or parent.resolve().is_relative_to(output.resolve())
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('fresh disjoint reference job required')
    m,training,services=scoring_binding(c,parent,m,original,remote)
    training=dict(training,reference_source=reference_source(parent,original))
    build=build_config(training,remote,evidence,prefetch=c.get('input_prefetch'))
    # Keeping the same mounted-device scope as numerical admission does not
    # start training ranks: this job starts only inference services.
    _,service_devices=visibility(m)
    check_paths({'reference_job':output});shutil.copytree(parent,output)
    (output/'PREPARED.json').unlink()
    (output/'reference-inputs').mkdir()
    for name,filename in [('input_proof','proof.json'),('input_receipts','receipts.jsonl')]:
        shutil.copyfile(evidence[name],output/'reference-inputs'/filename)
        if file_hash(output/'reference-inputs'/filename)!=evidence[name+'_sha256']:
            raise ValueError('copied independent evidence differs')
    commit_json(output/'REFERENCE-INPUT-AUDIT.json',input_proof)
    commit_json(output/'REFERENCE-PARENT-AUDIT.json',proof)
    resume_files=resume_assets(c,output,build,remote,prepare=True)
    (output/'reference-template.json').write_text('"${REFERENCE_DATASET_BUILD_CONFIG}"\n')
    (output/'run.py').write_text(reference_runner(service_devices,c.get('shutdown_timeout_seconds'),
        resume=bool(resume_files), legacy_engine_timeout=c.get('legacy_engine_timeout_seconds')))
    names=['config.json','reference-local.json','reference-build-config.json',*m['layout']['services'].values()]
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in names})
    build['pins']=pins;training['pins']=pins
    (output/'config.json').write_text(json.dumps(training,indent=2)+'\n')
    commit_json(output/'reference-build-config.json',build)
    commit_json(output/'reference-local.json',dict(schema='jev-local-macros-v1',
        macros={'REFERENCE_DATASET_BUILD_CONFIG':build}))
    for key,service in services.items():
        service['pins']=pins
        name=m['layout']['services'][key]
        (output/name).write_text(json.dumps(service,indent=2)+'\n')
    sources={str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')}
    if sources!=json.loads((parent/'RUNTIME-SOURCES.json').read_text()):
        raise ValueError('reference job changed sealed runtime sources')
    receipt=dict(m,status='full_train_reference_job_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],config_provenance=provenance,
        numerical_admission=False,reference_dispatch_allowed=False,training_dispatch_allowed=False,
        expected_population=len(rows),actor_updates=0,
        pending=['bind actual endpoint batch32 numerical admission',
                 'exact-container build lifecycle','full independent output/journal audit'])
    receipt['layout']=copy.deepcopy(m['layout'])
    receipt['layout']['configs']+=['reference-local.json','reference-template.json','reference-build-config.json']
    if resume_files:
        receipt['layout']['configs'].append('reference-prefix.json')
    receipt['layout']['directories']=['reference-build','reference-journal','supervision',
        *['service-'+name for name,_ in service_devices]]
    receipt['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',receipt)
    return dict(status=receipt['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
        runtime_identity=proof['runtime_identity'],reference_dispatch_allowed=False,training_dispatch_allowed=False)


def audit(c,provenance,*, parent_auditor=None):
    """Independently reconstruct the job; rehashing changed configs is insufficient."""
    root=Path(c['output']);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('reference job identity differs')
    m=json.loads(manifest.read_text())
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if (any(p.is_symlink() for p in root.rglob('*')) or actual!=m['files']
            or m['status']!='full_train_reference_job_prepared_only'
            or m['reference_dispatch_allowed'] is not False or m['training_dispatch_allowed'] is not False
            or m['numerical_admission'] is not False or m['actor_updates']!=0
            or not isinstance(m['expected_population'],int) or m['expected_population']<=0):
        raise ValueError('complete immutable reference-only job required')
    proof=(parent_auditor or audit_parent)(c['parent'],provenance)
    parent=Path(c['parent']['output']);pm=json.loads((parent/'PREPARED.json').read_text())
    original=json.loads((parent/'config.json').read_text())
    remote=path(c['remote_root']);old=pm['remote_root']
    binding,training,services=scoring_binding(c,parent,pm,original,remote)
    if (m['remote_root']!=remote or m['parent_sha256']!=c['parent']['manifest_sha256']
            or m['training_devices']!=binding['training_devices']
            or m['inference_services']!=binding['inference_services']):
        raise ValueError('reference parent or service scope differs')
    # All parent payload outside the explicit configuration/runner replacements
    # stays byte-identical, including source, endpoint and schedule assets.
    changed={'PREPARED.json','run.py','config.json',*pm['layout']['services'].values()}
    for p in parent.rglob('*'):
        name=str(p.relative_to(parent))
        if p.is_file() and name not in changed and actual.get(name)!=file_hash(p):
            raise ValueError('inherited reference dependency changed: '+name)
    expected_names={str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}
    expected_names.discard('PREPARED.json')
    expected_names.update(binding['layout']['services'].values())
    expected_names.update({'reference-inputs/proof.json','reference-inputs/receipts.jsonl',
        'REFERENCE-INPUT-AUDIT.json','REFERENCE-PARENT-AUDIT.json',
        'reference-template.json','reference-local.json','reference-build-config.json'})
    evidence=dict(c['input_evidence'],prompt_layout=original.get('prompt_layout'))
    rows,task,_=load(original['release'],original['release_sha256'],'learning',purpose='optimize')
    _,input_proof=input_evidence(evidence,rows=rows,task=task,release_sha256=original['release_sha256'],
        processor_sha256=proof['runtime_identity']['processor_sha256'])
    if len(rows)!=reference_source(parent,original)['population']:raise ValueError('full TRAIN reference population required')
    for name,filename in [('input_proof','proof.json'),('input_receipts','receipts.jsonl')]:
        if actual['reference-inputs/'+filename]!=evidence[name+'_sha256']:
            raise ValueError('reference input evidence differs')
    if m['expected_population']!=len(rows) or json.loads((root/'REFERENCE-INPUT-AUDIT.json').read_text())!=input_proof:
        raise ValueError('reference input audit differs')
    prior=json.loads((root/'REFERENCE-PARENT-AUDIT.json').read_text())
    # Provenance identifies the prepare invocation, not this fresh audit call.
    if {k:v for k,v in prior.items() if k!='provenance'}!={k:v for k,v in proof.items() if k!='provenance'}:
        raise ValueError('reference parent audit differs')
    training=dict(training,reference_source=reference_source(parent,original))
    build=build_config(training,remote,evidence,prefetch=c.get('input_prefetch'))
    resume_files=resume_assets(c,root,build,remote)
    if set(actual)!=expected_names | set(resume_files):
        raise ValueError('unexpected reference package assets')
    _,devices=visibility(binding)
    if (root/'run.py').read_text()!=reference_runner(devices,c.get('shutdown_timeout_seconds'),
            resume=bool(resume_files), legacy_engine_timeout=c.get('legacy_engine_timeout_seconds')):
        raise ValueError('reference supervised runner differs')
    if (root/'reference-template.json').read_text()!='"${REFERENCE_DATASET_BUILD_CONFIG}"\n':
        raise ValueError('reference macro template differs')
    names={'config.json','reference-local.json','reference-build-config.json',*binding['layout']['services'].values()}
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+name:h for name,h in actual.items() if name not in names})
    build['pins']=pins;training['pins']=pins
    expected={'config.json':training,'reference-build-config.json':build,
        'reference-local.json':dict(schema='jev-local-macros-v1',macros={'REFERENCE_DATASET_BUILD_CONFIG':build})}
    for key,service in services.items():
        expected[binding['layout']['services'][key]]=dict(service,pins=pins)
    for name,value in expected.items():
        if json.loads((root/name).read_text())!=value:
            raise ValueError('reference configuration differs: '+name)
    layout=copy.deepcopy(binding['layout'])
    layout['configs']+=['reference-local.json','reference-template.json','reference-build-config.json']
    if resume_files:
        layout['configs'].append('reference-prefix.json')
    layout['directories']=['reference-build','reference-journal','supervision',*['service-'+n for n,_ in devices]]
    if m['layout']!=layout:raise ValueError('reference collection layout differs')
    return dict(status='reference_job_independently_audited_not_numerical_admission',
        manifest_sha256=c['manifest_sha256'],runtime_identity=proof['runtime_identity'],
        source_policy_id=build['source_policy_id'],recordings=len(rows),actor_updates=0,
        reference_dispatch_allowed=False,training_dispatch_allowed=False,npu_executed=False,
        provenance=provenance)


def collected_audit_config(c):
    """Bind independently collected artifacts to the original prepared config.

    Transport/terminal and numerical admission remain separate checks. This
    function never rewrites the producer's remote paths or output identity.
    """
    root=Path(c['output']);collection=Path(c['collection'])
    expected=root/'reference-build-config.json';actual=collection/'reference-build-config.json'
    if file_hash(actual)!=file_hash(expected):
        raise ValueError('collected producer config differs from prepared job')
    build=json.loads(actual.read_text());remote=path(c['remote_root'])
    for key,relative in [('stage1_commit','initial/endpoint-commit.json'),
                         ('source_policy','initial/policy'),('input_proof','reference-inputs/proof.json'),
                         ('input_receipts','reference-inputs/receipts.jsonl'),('output','reference-build')]:
        if build[key]!=remote+'/'+relative:
            raise ValueError('unexpected producer path: '+key)
    pool=build['macro_client']
    if pool['journal_root']!=remote+'/reference-journal':
        raise ValueError('reference journal path differs')
    attempt=pool['attempt_id']
    import re
    if not isinstance(attempt,str) or not re.fullmatch('[A-Za-z0-9_-]+',attempt):
        raise ValueError('contained reference attempt required')
    result=dict(build_config=str(actual),build_config_sha256=file_hash(actual),
        build_output=str(collection/'reference-build'),
        journal=str(collection/'reference-journal'/attempt),
        stage1_policy=str(root/'initial/policy'),stage1_commit=str(root/'initial/endpoint-commit.json'),
        stage1_commit_sha256=file_hash(root/'initial/endpoint-commit.json'),output=c['results_audit_output'])
    if build.get('prompt_layout') is not None:
        result['prompt_layout']=build['prompt_layout']
    for key,filename in [('input_proof','proof.json'),('input_receipts','receipts.jsonl')]:
        result[key]=str(root/'reference-inputs'/filename)
        result[key+'_sha256']=build[key+'_sha256']
    return result


def replay_reference_outputs(c,provenance):
    """Recompute evidence without publishing a report on every caller's audit.

    Keep the sealed dataset auditor unchanged. Its CLI remains an immutable
    publisher; nested prepare/audit calls only need the freshly rebuilt proof.
    """
    from tools.reports.audit_context_reference_dataset import run
    publications=[]
    def capture(path,value):
        if str(path)!=str(c['output']) or publications:
            raise ValueError('unexpected reference audit publication')
        publications.append(value)
    replay=FunctionType(run.__code__,dict(run.__globals__,commit_json=capture),'readonly_reference_replay')
    result=replay(c,provenance)
    if publications!=[result]:raise ValueError('complete reference audit result required')
    return result


def audit_results(c,provenance,*, package_auditor=None):
    """Exact terminal runtime + complete transport + independent output replay."""
    if 'shutdown_recovery' in c:
        from tools.reports.context_reference_shutdown_recovery import audit_recovery
        return audit_recovery(c,provenance)
    if package_auditor is not None:
        return audit_attempt_outputs(c,provenance,package_auditor=package_auditor)
    return audit_attempt_outputs(c,provenance)


def audit_attempt_outputs(c,provenance,*,shutdown_failure=False,package_auditor=None):
    job=(package_auditor or audit)(c,provenance)
    root=Path(c['output']);collection=Path(c['collection'])
    transport=json.loads(Path(c['transport']).read_text())
    handle=json.loads(Path(c['launch']).read_text())
    observed=json.loads(Path(c['inspect']).read_text())
    from tools.reports.context_cached_storage_operations import reference_command,audit_container
    from tools.reports.prepare_context_replica_training_candidate import IMAGE
    m=json.loads((root/'PREPARED.json').read_text())
    config=json.loads((root/'config.json').read_text())
    argv=reference_command(c['deployment'],m,config)
    if c['expected_argv']!=argv:raise ValueError('reconstructed readonly reference command differs')
    storage=audit_container(argv,config,handle,observed,producer=True,
                            expected_exit=1 if shutdown_failure else 0)
    if (handle['manifest_sha256']!=c['manifest_sha256'] or handle['layout']!=m['layout']
            or handle['remote_root']!=c['remote_root'] or handle['image']!=IMAGE
            or transport['container_id']!=handle['container_id'] or transport['image']!=IMAGE
            or transport['remote_root']!=c['remote_root'] or transport['state']!=observed['State']
            or transport['status']!='transport_complete_independent_audit_required'
            or transport['bad_pins'] or transport['bad_package'] or transport['launch_binding_mismatches']):
        raise ValueError('complete original reference job transport required')
    actual={str(p.relative_to(collection)):file_hash(p) for p in collection.rglob('*') if p.is_file()}
    if (collection.is_symlink() or any(p.is_symlink() for p in collection.rglob('*'))
            or actual!=transport['files']):
        raise ValueError('complete unchanged reference collection required')
    for name in m['layout']['configs']+list(m['layout']['services'].values()):
        if actual.get(name)!=file_hash(root/name):
            raise ValueError('executed reference configuration differs: '+name)
    supervision=json.loads((collection/'supervision/terminal.json').read_text())
    if (supervision.get('passed') is not (not shutdown_failure)
            or supervision.get('actor_started') is not True or supervision.get('actor_exit_code')!=0
            or supervision.get('forced_cleanup') is not shutdown_failure
            or supervision.get('error') is not None
            or [s['name'] for s in supervision['services']]!=[s['name'] for s in m['inference_services']]
            or any(s.get('exit_code')!=0 for s in supervision['services'])):
        raise ValueError('expected ownership and successful builder required')
    result=replay_reference_outputs(collected_audit_config(c),provenance)
    if 'resume' in c:
        from tools.reports.context_reference_resume_release import audit_resumed
        result['resume']=audit_resumed(root,collection,result['macro_batches'])
    return dict(status='reference_job_outputs_audited_numerical_admission_separate',
        output_audit=result,job_audit=job,storage_audit=storage,container_id=handle['container_id'],
        transport_sha256=file_hash(c['transport']),inspect_sha256=file_hash(c['inspect']),
        numerical_parity_verified=False,training_dispatch_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit','audit-results'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance) if a.action=='audit' else audit_results(c,provenance)
    if a.action=='audit-results':commit_json(c['results_audit_output'],result['output_audit'])
    if a.action!='prepare':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
