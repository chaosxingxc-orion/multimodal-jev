"""Bind endpoint TRAIN parity to the frozen shared runtime.

HF runs first in an owned process, then the declared inference services and a macro
client. Preparing the package does not execute or admit numerical equivalence.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_context_stage2_runtime import audit as audit_parent
from tools.reports.context_train_parity_inputs import audit as audit_inputs
from tools.reports.context_replica_training_operations import path
from tools.reports.prepare_context_reference_job import RUNNER as SHARED_RUNNER, scoring_binding, with_shutdown_timeout, reference_source
from tools.reports.context_device_visibility import visibility


def read(p):return json.loads(Path(p).read_text())


def configuration(training,m,inputs,remote):
    keys=('model','source_root','source_policy','source_policy_id','source_contract','base_id',
        'initial_adapter','release_sha256','audio_token_id','dual_runtime_identity','dual_runtime_manifest','macro_client')
    value={k:copy.deepcopy(training[k]) for k in keys}
    data=read(Path(inputs['output'])/'inputs.json')
    if (data['source']['release_sha256']!=value['release_sha256']
            or data['source']['processor_sha256']!=value['dual_runtime_identity']['processor_sha256']
            or inputs['cache_root']!=training['encoded_audio_cache']
            or data['source'].get('prompt_layout')!=training.get('prompt_layout')):
        raise ValueError('same TRAIN release, processor and cache required')
    devices=[*m['training_devices'],*[s['device'] for s in m['inference_services']]]
    visibility(m)
    client=value['macro_client']
    ref=training['reference_source']
    size=len(data['panel']['ids'])
    width=client['rollout_queries']
    macro=client['max_queries']
    if (type(width) is not int or type(macro) is not int or not 1<=width<=macro<=size
            or data['panel']['population']!=ref['population']
            or len(client['replicas'])!=len(m['inference_services'])
            or list(m['training_devices'])!=training['physical_devices']
            or client['runtime_identity']!=value['dual_runtime_identity']):
        raise ValueError('parity panel must exercise the declared runtime, topology and at least one complete macro')
    client.update(attempt_id='parity-'+client['attempt_id'],journal_root=remote+'/parity-journal')
    target=devices[0];mounted=sorted(devices)
    value.update(stage1_commit=remote+'/initial/endpoint-commit.json',stage1_endpoint=ref['endpoint'],expected_population=ref['population'],
        train_inputs=dict(output=remote+'/parity-inputs',manifest_sha256=inputs['manifest_sha256'],cache_root=inputs['cache_root']),
        max_model_len=ref['max_tokens'],physical_devices=[target],
        process_device_scope=dict(kind='same_container_runtime_visibility_v1',mounted_physical=mounted,
            visible_ordinal=str(mounted.index(target))))
    if training.get('prompt_layout') is not None:
        value['prompt_layout']=training['prompt_layout']
    return value


def runner(m, shutdown_timeout=None):
    # Keep the established service environment/supervisor; insert one owned HF
    # subprocess which must exit successfully before any service is started.
    source=SHARED_RUNNER.replace('import json,os,sys','import json,os,sys,subprocess')
    edits=[("from context_process_supervisor import supervise_services",
        "from context_process_supervisor import supervise_services,stop_owned\nfrom multimodal_jev.local_checkpoint import commit_json"),
        ("c=json.loads((r/'reference-local.json').read_text())['macros']['REFERENCE_DATASET_BUILD_CONFIG']",
         "c=json.loads((r/'parity-hf.json').read_text())"),
        ('services=[]',
         "control=r/'hf-process';control.mkdir(exist_ok=False)\n"
         "with (control/'process.log').open('x') as log:\n"
         " child=subprocess.Popen([sys.executable,str(r/'source/tools/reports/context_train_macro_parity.py'),\n"
         "  '--config',str(r/'parity-template.json'),'--local',str(r/'parity-hf-local.json'),'--phase','hf'],\n"
         "  env=env('parity-hf',c['process_device_scope']['visible_ordinal']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
         " try:\n  code=child.wait()\n finally:\n  stop_owned(child,30.)\n"
         "commit_json(control/'terminal.json',dict(exit_code=code,pid=child.pid))\n"
         "if code!=0:raise SystemExit(code)\n"
         "if json.loads((r/'hf/terminal.json').read_text())!={'exit_code':0}:raise ValueError('HF terminal missing')\n"
         "services=[]"),
        (" actor_command=[sys.executable,str(r/'source/tools/reports/build_context_reference_dataset.py'),\n  '--config',str(r/'reference-template.json'),'--local',str(r/'reference-local.json')],",
         " actor_command=[sys.executable,'-c',MACRO_CLIENT,str(r/'parity-macro.json'),str(r/'hf/reference.json')],"),
        ("actor_env=env('reference-client','')","actor_env=env('parity-client','')") ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('shared parity lifecycle anchor changed')
        source=source.replace(before,after)
    client=("import json,sys;from pathlib import Path;from multimodal_jev.baseline import file_hash;"
        "from tools.reports.context_train_macro_parity import run;"
        "c=json.loads(Path(sys.argv[1]).read_text());c['hf_reference']={'path':sys.argv[2],'sha256':file_hash(sys.argv[2])};"
        "run(c,{'source':'sealed_train_parity_client'},'macro')")
    source=source.replace('report=supervise_services(', 'MACRO_CLIENT='+repr(client)+'\nreport=supervise_services(')
    _,services=visibility(m)
    source=with_shutdown_timeout(source.replace('__SERVICES__',repr(services)),shutdown_timeout)
    compile(source,'<parity-runner>','exec')
    return source


def prepare(c,provenance,*, parent_auditor=None):
    parent_proof=(parent_auditor or audit_parent)(c['parent'],provenance);input_proof=audit_inputs(c['inputs'],provenance)
    parent=Path(c['parent']['output']);m=read(parent/'PREPARED.json');old=m['remote_root']
    output=Path(c['output']);remote=path(c['remote_root']);inputs=Path(c['inputs']['output'])
    if (output.exists() or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
            for p in (parent,inputs)) or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('fresh disjoint parity package required')
    original=read(parent/'config.json')
    m,training,services=scoring_binding(c,parent,m,original,remote)
    config=configuration(training,m,c['inputs'],remote)
    reference_source(parent,original)
    sources=read(parent/'RUNTIME-SOURCES.json')
    for name in ('context_train_macro_parity.py','context_train_parity_inputs.py'):
        if sources.get('tools/reports/'+name)!=file_hash(Path(__file__).with_name(name)):
            raise ValueError('shared runtime must already contain current parity client before binding')
    check_paths({'parity_package':output});shutil.copytree(parent,output);(output/'PREPARED.json').unlink()
    shutil.copytree(inputs,output/'parity-inputs')
    commit_json(output/'PARITY-PARENT-AUDIT.json',parent_proof);commit_json(output/'PARITY-INPUT-AUDIT.json',input_proof)
    (output/'run.py').write_text(runner(m,c.get('shutdown_timeout_seconds')))
    (output/'parity-template.json').write_text('"${TRAIN_MACRO_PARITY_CONFIG}"\n')
    config_names=['config.json','parity-hf.json','parity-hf-local.json','parity-macro.json',*m['layout']['services'].values()]
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in config_names})
    training['pins']=pins;commit=(output/'config.json');commit.write_text(json.dumps(training,indent=2)+'\n')
    config['pins']=pins;hf=dict(config,output=remote+'/hf');macro=dict(config,output=remote+'/evaluation')
    commit_json(output/'parity-hf.json',hf);commit_json(output/'parity-macro.json',macro)
    commit_json(output/'parity-hf-local.json',dict(schema='jev-local-macros-v1',macros={'TRAIN_MACRO_PARITY_CONFIG':hf}))
    for key,value in services.items():
        name=m['layout']['services'][key];value['pins']=pins
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    if {str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')}!=sources:
        raise ValueError('parity binding changed sealed runtime code')
    result=dict(m,status='endpoint_train_parity_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],input_manifest_sha256=c['inputs']['manifest_sha256'],
        training_dispatch_allowed=False,numerical_admission=False,finite_npu_dispatch_allowed=False,
        actor_updates=0,recordings=input_proof['recordings'],config_provenance=provenance)
    result['layout']=copy.deepcopy(m['layout'])
    result['layout']['configs']+=['parity-hf.json','parity-hf-local.json','parity-macro.json','parity-template.json']
    result['layout']['directories']=['hf','hf-process','evaluation','parity-journal','supervision',
        *['service-'+s['name'] for s in m['inference_services']]]
    result['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',result)
    return dict(status=result['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
        training_dispatch_allowed=False,npu_executed=False)


def audit(c,provenance,*, parent_auditor=None):
    """Reconstruct the bound job, independently of its self-reported hashes."""
    root=Path(c['output']);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('parity package identity differs')
    m=read(manifest)
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if any(p.is_symlink() for p in root.rglob('*')) or actual!=m['files']:
        raise ValueError('complete immutable parity package required')
    parent_proof=(parent_auditor or audit_parent)(c['parent'],provenance);input_proof=audit_inputs(c['inputs'],provenance)
    parent=Path(c['parent']['output']);pm=read(parent/'PREPARED.json')
    inputs=Path(c['inputs']['output']);remote=path(c['remote_root']);old=pm['remote_root']
    if (any(root.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(root.resolve())
            for p in (parent,inputs)) or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('disjoint parity package required')
    original=read(parent/'config.json')
    bound,training,services=scoring_binding(c,parent,pm,original,remote)
    config=configuration(training,bound,c['inputs'],remote)
    reference_source(parent,original)
    sources=read(parent/'RUNTIME-SOURCES.json')
    for name in ('context_train_macro_parity.py','context_train_parity_inputs.py'):
        if sources.get('tools/reports/'+name)!=file_hash(Path(__file__).with_name(name)):
            raise ValueError('shared runtime parity client differs')
    changed={'PREPARED.json','run.py','config.json',*pm['layout']['services'].values()}
    expected_names=set()
    for p in parent.rglob('*'):
        if not p.is_file():continue
        name=str(p.relative_to(parent))
        if name!='PREPARED.json':expected_names.add(name)
        if name not in changed and actual.get(name)!=file_hash(p):
            raise ValueError('inherited parity dependency changed: '+name)
    for p in inputs.rglob('*'):
        if not p.is_file():continue
        name='parity-inputs/'+str(p.relative_to(inputs));expected_names.add(name)
        if actual.get(name)!=file_hash(p):raise ValueError('parity input copy differs: '+name)
    expected_names.update({'PARITY-PARENT-AUDIT.json','PARITY-INPUT-AUDIT.json','parity-template.json',
        'parity-hf.json','parity-hf-local.json','parity-macro.json'})
    expected_names.update(bound['layout']['services'].values())
    if set(actual)!=expected_names:raise ValueError('unexpected parity package assets')
    for name,proof in [('PARITY-PARENT-AUDIT.json',parent_proof),('PARITY-INPUT-AUDIT.json',input_proof)]:
        prior=read(root/name)
        if {k:v for k,v in prior.items() if k!='provenance'}!={k:v for k,v in proof.items() if k!='provenance'}:
            raise ValueError('parity prerequisite audit differs: '+name)
    if (root/'run.py').read_text()!=runner(bound,c.get('shutdown_timeout_seconds')):
        raise ValueError('parity process lifecycle differs')
    if (root/'parity-template.json').read_text()!='"${TRAIN_MACRO_PARITY_CONFIG}"\n':
        raise ValueError('parity macro template differs')
    config_names={'config.json','parity-hf.json','parity-hf-local.json','parity-macro.json',
        *bound['layout']['services'].values()}
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+name:h for name,h in actual.items() if name not in config_names})
    training['pins']=pins;config['pins']=pins
    hf=dict(config,output=remote+'/hf');macro=dict(config,output=remote+'/evaluation')
    expected={'config.json':training,'parity-hf.json':hf,'parity-macro.json':macro,
        'parity-hf-local.json':dict(schema='jev-local-macros-v1',macros={'TRAIN_MACRO_PARITY_CONFIG':hf})}
    for key,service in services.items():
        expected[bound['layout']['services'][key]]=dict(service,pins=pins)
    for name,value in expected.items():
        if read(root/name)!=value:raise ValueError('parity configuration differs: '+name)
    expected_manifest=dict(bound,status='endpoint_train_parity_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],input_manifest_sha256=c['inputs']['manifest_sha256'],
        training_dispatch_allowed=False,numerical_admission=False,finite_npu_dispatch_allowed=False,
        actor_updates=0,recordings=input_proof['recordings'],config_provenance=m['config_provenance'],files=actual)
    expected_manifest['layout']=copy.deepcopy(bound['layout'])
    expected_manifest['layout']['configs']+=['parity-hf.json','parity-hf-local.json','parity-macro.json','parity-template.json']
    expected_manifest['layout']['directories']=['hf','hf-process','evaluation','parity-journal','supervision',
        *['service-'+s['name'] for s in bound['inference_services']]]
    if m!=expected_manifest:raise ValueError('parity manifest scope or collection layout differs')
    return dict(status='train_parity_job_independently_audited_not_numerical_admission',
        manifest_sha256=c['manifest_sha256'],source_policy_id=config['source_policy_id'],
        runtime_identity=config['dual_runtime_identity'],recordings=input_proof['recordings'],actor_updates=0,
        numerical_admission=False,training_dispatch_allowed=False,npu_executed=False,provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
