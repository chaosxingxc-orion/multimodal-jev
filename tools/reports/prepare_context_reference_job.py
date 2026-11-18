"""Prepare a full TRAIN reference job on the sealed shared Stage2 runtime.

The package includes a supervised service/client entry, never a training actor.
Numerical admission and remote lifecycle are separate required release gates.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load
from tools.reports.context_reference_input_evidence import load as input_evidence
from tools.reports.prepare_context_stage2_runtime import audit as audit_parent
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.context_replica_training_operations import path
from tools.reports.context_device_visibility import visibility


RUNNER = '''import json,os,sys
from pathlib import Path
from context_process_supervisor import supervise_services
from multimodal_jev.local_runtime_storage import require_training_storage
r=Path(__file__).resolve().parent
require_training_storage(r)
c=json.loads((r/'reference-local.json').read_text())['macros']['REFERENCE_DATASET_BUILD_CONFIG']
if c['expected_population']!=40513 or c['stage1_endpoint']!=318:
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
  command=[sys.executable,str(r/'source/tools/reports/context_replica_runtime.py'),
   '--config',str(r/('service-'+name+'.json')),'--output',str(r/('service-'+name))],
  env=env('service-'+name,visible),ready_path=r/('service-'+name)/'ready.json',
  socket_path=Path(service['socket_path'])))
report=supervise_services(services=services,
 actor_command=[sys.executable,str(r/'source/tools/reports/build_context_reference_dataset.py'),
  '--config',str(r/'reference-template.json'),'--local',str(r/'reference-local.json')],
 actor_env=env('reference-client',''),output=r/'supervision')
raise SystemExit(0 if report['passed'] else 1)
'''


def build_config(training, remote, evidence):
    keys=('source_root','source_policy','source_policy_id','source_contract','base_id',
          'release','release_sha256','audio_token_id','encoded_audio_cache',
          'dual_runtime_identity','dual_runtime_manifest','macro_client')
    c={k:copy.deepcopy(training[k]) for k in keys}
    pool=c['macro_client']
    if (pool['max_queries']!=128 or pool['rollout_queries']!=32 or len(pool['replicas'])!=2
            or pool['runtime_identity']!=c['dual_runtime_identity']):
        raise ValueError('common batch32 macro128 two-service producer required')
    pool['attempt_id']='reference-'+pool['attempt_id']
    pool['journal_root']=remote+'/reference-journal'
    c.update(stage1_commit=remote+'/initial/endpoint-commit.json',stage1_endpoint=318,
        expected_population=40513,max_tokens=4096,output=remote+'/reference-build',
        audio_mirror_root=str(Path(training['audio_mirror_root'])/'audio'))
    for name in ('input_proof','input_receipts'):
        c[name]=remote+'/reference-inputs/'+('proof.json' if name=='input_proof' else 'receipts.jsonl')
        c[name+'_sha256']=evidence[name+'_sha256']
    return c


def prepare(c,provenance):
    proof=audit_parent(c['parent'],provenance)
    parent=Path(c['parent']['output']);m=json.loads((parent/'PREPARED.json').read_text())
    original=json.loads((parent/'config.json').read_text())
    evidence=c['input_evidence'];rows,task,_=load(original['release'],original['release_sha256'],'learning',purpose='optimize')
    _,input_proof=input_evidence(evidence,rows=rows,task=task,release_sha256=original['release_sha256'],
        processor_sha256=proof['runtime_identity']['processor_sha256'])
    if len(rows)!=40513:raise ValueError('full TRAIN reference population required')
    output=Path(c['output']);remote=path(c['remote_root']);old=m['remote_root']
    if (output.exists() or output.resolve().is_relative_to(parent.resolve())
            or parent.resolve().is_relative_to(output.resolve())
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('fresh disjoint reference job required')
    training=rebase(original,old,remote)
    build=build_config(training,remote,evidence)
    # Keeping the same mounted-device scope as numerical admission does not
    # start training ranks: this job starts only the two inference services.
    _,service_devices=visibility(m)
    if len(service_devices)!=2:raise ValueError('exactly two reference services required')
    check_paths({'reference_job':output});shutil.copytree(parent,output)
    (output/'PREPARED.json').unlink()
    (output/'reference-inputs').mkdir()
    for name,filename in [('input_proof','proof.json'),('input_receipts','receipts.jsonl')]:
        shutil.copyfile(evidence[name],output/'reference-inputs'/filename)
        if file_hash(output/'reference-inputs'/filename)!=evidence[name+'_sha256']:
            raise ValueError('copied independent evidence differs')
    commit_json(output/'REFERENCE-INPUT-AUDIT.json',input_proof)
    commit_json(output/'REFERENCE-PARENT-AUDIT.json',proof)
    (output/'reference-template.json').write_text('"${REFERENCE_DATASET_BUILD_CONFIG}"\n')
    (output/'run.py').write_text(RUNNER.replace('__SERVICES__',repr(service_devices)))
    names=['config.json','reference-local.json','reference-build-config.json',*m['layout']['services'].values()]
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in names})
    build['pins']=pins;training['pins']=pins
    (output/'config.json').write_text(json.dumps(training,indent=2)+'\n')
    commit_json(output/'reference-build-config.json',build)
    commit_json(output/'reference-local.json',dict(schema='jev-local-macros-v1',
        macros={'REFERENCE_DATASET_BUILD_CONFIG':build}))
    for name in m['layout']['services'].values():
        service=rebase(json.loads((parent/name).read_text()),old,remote)
        service['pins']=pins
        (output/name).write_text(json.dumps(service,indent=2)+'\n')
    sources={str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')}
    if sources!=json.loads((parent/'RUNTIME-SOURCES.json').read_text()):
        raise ValueError('reference job changed sealed runtime sources')
    receipt=dict(m,status='full_train_reference_job_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],config_provenance=provenance,
        numerical_admission=False,reference_dispatch_allowed=False,training_dispatch_allowed=False,
        expected_population=40513,actor_updates=0,
        pending=['bind actual endpoint batch32 numerical admission',
                 'exact-container build lifecycle','full independent output/journal audit'])
    receipt['layout']=copy.deepcopy(m['layout'])
    receipt['layout']['configs']+=['reference-local.json','reference-template.json','reference-build-config.json']
    receipt['layout']['directories']=['reference-build','reference-journal','supervision',
        *['service-'+name for name,_ in service_devices]]
    receipt['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',receipt)
    return dict(status=receipt['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
        runtime_identity=proof['runtime_identity'],reference_dispatch_allowed=False,training_dispatch_allowed=False)


def audit(c,provenance):
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
            or m['expected_population']!=40513):
        raise ValueError('complete immutable reference-only job required')
    proof=audit_parent(c['parent'],provenance)
    parent=Path(c['parent']['output']);pm=json.loads((parent/'PREPARED.json').read_text())
    original=json.loads((parent/'config.json').read_text())
    remote=path(c['remote_root']);old=pm['remote_root']
    if (m['remote_root']!=remote or m['parent_sha256']!=c['parent']['manifest_sha256']
            or m['inference_services']!=pm['inference_services']):
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
    expected_names.update({'reference-inputs/proof.json','reference-inputs/receipts.jsonl',
        'REFERENCE-INPUT-AUDIT.json','REFERENCE-PARENT-AUDIT.json',
        'reference-template.json','reference-local.json','reference-build-config.json'})
    if set(actual)!=expected_names:
        raise ValueError('unexpected reference package assets')
    evidence=c['input_evidence']
    rows,task,_=load(original['release'],original['release_sha256'],'learning',purpose='optimize')
    _,input_proof=input_evidence(evidence,rows=rows,task=task,release_sha256=original['release_sha256'],
        processor_sha256=proof['runtime_identity']['processor_sha256'])
    if len(rows)!=40513:raise ValueError('full TRAIN reference population required')
    for name,filename in [('input_proof','proof.json'),('input_receipts','receipts.jsonl')]:
        if actual['reference-inputs/'+filename]!=evidence[name+'_sha256']:
            raise ValueError('reference input evidence differs')
    if json.loads((root/'REFERENCE-INPUT-AUDIT.json').read_text())!=input_proof:
        raise ValueError('reference input audit differs')
    prior=json.loads((root/'REFERENCE-PARENT-AUDIT.json').read_text())
    # Provenance identifies the prepare invocation, not this fresh audit call.
    if {k:v for k,v in prior.items() if k!='provenance'}!={k:v for k,v in proof.items() if k!='provenance'}:
        raise ValueError('reference parent audit differs')
    training=rebase(original,old,remote);build=build_config(training,remote,evidence)
    _,devices=visibility(pm)
    if (root/'run.py').read_text()!=RUNNER.replace('__SERVICES__',repr(devices)):
        raise ValueError('reference supervised runner differs')
    if (root/'reference-template.json').read_text()!='"${REFERENCE_DATASET_BUILD_CONFIG}"\n':
        raise ValueError('reference macro template differs')
    names={'config.json','reference-local.json','reference-build-config.json',*pm['layout']['services'].values()}
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+name:h for name,h in actual.items() if name not in names})
    build['pins']=pins;training['pins']=pins
    expected={'config.json':training,'reference-build-config.json':build,
        'reference-local.json':dict(schema='jev-local-macros-v1',macros={'REFERENCE_DATASET_BUILD_CONFIG':build})}
    for name in pm['layout']['services'].values():
        expected[name]=rebase(json.loads((parent/name).read_text()),old,remote)
        expected[name]['pins']=pins
    for name,value in expected.items():
        if json.loads((root/name).read_text())!=value:
            raise ValueError('reference configuration differs: '+name)
    layout=copy.deepcopy(pm['layout'])
    layout['configs']+=['reference-local.json','reference-template.json','reference-build-config.json']
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


def audit_results(c,provenance):
    """Exact terminal runtime + complete transport + independent output replay."""
    job=audit(c,provenance)
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
    storage=audit_container(argv,config,handle,observed,producer=True)
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
    if (supervision.get('passed') is not True or supervision.get('actor_exit_code')!=0
            or supervision.get('forced_cleanup') is not False
            or [s['name'] for s in supervision['services']]!=[s['name'] for s in m['inference_services']]
            or any(s.get('exit_code')!=0 for s in supervision['services'])):
        raise ValueError('successful owned services and builder required')
    result=replay_reference_outputs(collected_audit_config(c),provenance)
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
