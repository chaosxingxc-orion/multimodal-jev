"""Bind independently audited full reference results to a two-update trainer.

The reference stays at its producer path and is mounted read-only. Preparation
does not admit numerical parity or dispatch NPU training.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import ReferenceDataset
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load
from tools.reports.prepare_context_stage2_runtime import audit as audit_parent
from tools.reports.prepare_context_reference_job import audit_results
from tools.reports.prepare_context_replica_training_candidate import RUN_PY,ENGINEERING_GATE,rebase
from tools.reports.context_replica_training_operations import path
from tools.reports.context_cached_reference_trainer import validate_objective
from multimodal_jev.context_cached_tail_window import contract as tail_contract


def read(p):
    return json.loads(Path(p).read_text())


def reference_binding(training,result,proof,*,population,candidate_ids,remote_dataset):
    identity=result['identity']
    if (proof['status']!='reference_dataset_journal_audited'
            or proof['manifest_sha256']!=result['manifest_sha256']
            or proof['policy_id']!=training['source_policy_id']
            or proof['recordings']!=40513 or result['recordings']!=40513
            or set(result['population'])!=set(population) or len(population)!=40513
            or identity['policy_id']!=training['source_policy_id']
            or identity['base_id']!=training['base_id']
            or identity['contract']!=training['source_contract']
            or identity['runtime_identity']!=training['dual_runtime_identity']
            or identity['release_sha256']!=training['release_sha256']
            or identity['candidate_ids']!=list(candidate_ids)):
        raise ValueError('same endpoint/runtime and exact full TRAIN reference required')
    return dict(root=path(remote_dataset),manifest_sha256=result['manifest_sha256'],
                identity=copy.deepcopy(identity),population=dict(result['population']))


def runner(m,kind='engineering_prefix'):
    from tools.reports.context_device_visibility import visibility
    train=m['training_devices'];services=m['inference_services']
    if len(train)!=2 or len(services)!=2 or len(set(train+[s['device'] for s in services]))!=4:
        raise ValueError('two ranks and two single-device inference services required')
    if kind not in ('engineering_prefix','cached_reference_tail_admission'):
        raise ValueError('known finite cached acceptance kind required')
    entry=('context_cached_tail_trainer.py' if kind=='cached_reference_tail_admission'
           else 'context_cached_reference_trainer.py')
    train_visible,service_visible=visibility(m)
    return (RUN_PY.replace('{SERVICES}',repr(service_visible))
        .replace('{world}','2').replace('{train_visible}',repr(train_visible))
        .replace('{entry}',entry)
        .replace('{GATE}',ENGINEERING_GATE))


def inputs(c,provenance):
    """Recheck independent producers before binding their immutable outputs."""
    parent_proof=audit_parent(c['parent'],provenance)
    ref_proof=audit_results(c['reference_job'],provenance)
    parent=Path(c['parent']['output']);m=read(parent/'PREPARED.json');original=read(parent/'config.json')
    job=c['reference_job'];collection=Path(job['collection'])
    build=read(Path(job['output'])/'reference-build-config.json')
    result=read(collection/'reference-build/result.json')
    rows,task,_=load(original['release'],original['release_sha256'],'learning',purpose='optimize')
    reference=reference_binding(original,result,ref_proof['output_audit'],
        population={r['id'] for r in rows},candidate_ids=[v['id'] for v in task['schema']['candidates']],
        remote_dataset=build['output']+'/dataset')
    ReferenceDataset(collection/'reference-build/dataset',manifest_sha256=reference['manifest_sha256'],
        expected_identity=reference['identity'],expected_population=reference['population'])
    return parent,m,original,reference,parent_proof,ref_proof


def configured(original,old,remote,reference,kind='engineering_prefix'):
    if kind not in ('engineering_prefix','cached_reference_tail_admission'):
        raise ValueError('known finite cached acceptance kind required')
    config=rebase(original,old,remote)
    config.update(reference_dataset=reference,start_update=0,stop_update=2,
        experiment_kind=kind,engineering_prefix=True,checkpoint_interval=1,
        training_dispatch_allowed=False,runtime_preparation_only=False)
    config.pop('reference_cache',None);config.pop('resume_root',None)
    validate_objective(config)
    if kind=='cached_reference_tail_admission':tail_contract(config)
    return config


def prepare(c,provenance):
    parent,m,original,reference,parent_proof,ref_proof=inputs(c,provenance)
    job=c['reference_job'];collection=Path(job['collection'])
    output=Path(c['output']);remote=path(c['remote_root']);old=m['remote_root']
    forbidden=[old,reference['root'],job['remote_root']]
    if (output.exists() or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
            for p in (parent,collection,Path(job['output'])))
            or any(Path(remote).is_relative_to(p) or Path(p).is_relative_to(remote) for p in forbidden)):
        raise ValueError('fresh independent training package and reference required')
    kind=c.get('acceptance_kind','engineering_prefix')
    config=configured(original,old,remote,reference,kind)
    if kind=='cached_reference_tail_admission':
        for name in ('tools/reports/context_cached_tail_trainer.py',
                     'src/multimodal_jev/context_cached_tail_window.py'):
            if name not in read(parent/'RUNTIME-SOURCES.json'):
                raise ValueError('tail entry must already belong to the admitted shared runtime')
    check_paths({'cached_training_package':output});shutil.copytree(parent,output)
    (output/'PREPARED.json').unlink()
    (output/'run.py').write_text(runner(m,kind))
    commit_json(output/'REFERENCE-RESULT-AUDIT.json',ref_proof)
    commit_json(output/'TRAINING-PARENT-AUDIT.json',parent_proof)
    names=['config.json',*m['layout']['services'].values()]
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    dataset=collection/'reference-build/dataset'
    for name in ('MANIFEST.json','records.json','population.json'):
        external[reference['root']+'/'+name]=file_hash(dataset/name)
    pins=dict(external,**{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in names})
    config['pins']=pins
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in m['layout']['services'].values():
        service=rebase(read(parent/name),old,remote);service['pins']=pins
        (output/name).write_text(json.dumps(service,indent=2)+'\n')
    if {str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')}!=read(parent/'RUNTIME-SOURCES.json'):
        raise ValueError('reference consumer changed sealed runtime code')
    receipt=dict(m,status='cached_reference_training_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],reference_job_sha256=job['manifest_sha256'],
        reference_manifest_sha256=reference['manifest_sha256'],config_provenance=provenance,
        numerical_admission=False,finite_npu_dispatch_allowed=False,training_dispatch_allowed=False,
        pending=['actual source batch32 numerical admission','two NPU updates','tail and fresh resume audit'])
    receipt['layout']=copy.deepcopy(m['layout'])
    receipt['layout']['directories']=['run','supervision',*['service-'+s['name'] for s in m['inference_services']]]
    receipt['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',receipt)
    return dict(status=receipt['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
        reference_manifest_sha256=reference['manifest_sha256'],training_dispatch_allowed=False)


def without_provenance(value):
    if isinstance(value,dict):
        return {k:without_provenance(v) for k,v in value.items() if k!='provenance'}
    if isinstance(value,list):return [without_provenance(v) for v in value]
    return value


def audit(c,provenance):
    """Reconstruct payload, configuration and reference binding independently."""
    root=Path(c['output']);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('cached training manifest differs')
    m=read(manifest)
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if (any(p.is_symlink() for p in root.rglob('*')) or actual!=m['files']
            or m['status']!='cached_reference_training_prepared_only'
            or any(m[k] is not False for k in ('numerical_admission',
                'finite_npu_dispatch_allowed','training_dispatch_allowed'))):
        raise ValueError('complete immutable unadmitted training package required')
    parent,pm,original,reference,parent_proof,ref_proof=inputs(c,provenance)
    remote=path(c['remote_root']);old=pm['remote_root'];job=c['reference_job']
    if (any(root.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(root.resolve())
            for p in (parent,Path(job['collection']),Path(job['output'])))
            or any(Path(remote).is_relative_to(p) or Path(p).is_relative_to(remote)
                   for p in (old,reference['root'],job['remote_root']))):
        raise ValueError('disjoint cached training package required')
    expected_meta=dict(remote_root=remote,parent_sha256=c['parent']['manifest_sha256'],
        reference_job_sha256=job['manifest_sha256'],reference_manifest_sha256=reference['manifest_sha256'])
    for key in ('training_devices','inference_services','dual_runtime_identity','source_policy_id','batch_plan'):
        if key in pm:expected_meta[key]=pm[key]
    if any(m.get(k)!=v for k,v in expected_meta.items()):
        raise ValueError('cached training parent/reference/topology differs')
    changed={'PREPARED.json','run.py','config.json',*pm['layout']['services'].values()}
    parent_files={str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()}
    expected_names=(set(parent_files)-{'PREPARED.json'})|{'REFERENCE-RESULT-AUDIT.json','TRAINING-PARENT-AUDIT.json'}
    if set(actual)!=expected_names:raise ValueError('unexpected cached training assets')
    for name,digest in parent_files.items():
        if name not in changed and actual.get(name)!=digest:
            raise ValueError('inherited training dependency changed: '+name)
    for name,proof in [('REFERENCE-RESULT-AUDIT.json',ref_proof),('TRAINING-PARENT-AUDIT.json',parent_proof)]:
        if without_provenance(read(root/name))!=without_provenance(proof):
            raise ValueError('cached training producer audit differs: '+name)
    kind=c.get('acceptance_kind','engineering_prefix')
    if kind=='cached_reference_tail_admission':
        for name in ('tools/reports/context_cached_tail_trainer.py',
                     'src/multimodal_jev/context_cached_tail_window.py'):
            if name not in read(parent/'RUNTIME-SOURCES.json'):
                raise ValueError('tail entry must already belong to the admitted shared runtime')
    if (root/'run.py').read_text()!=runner(pm,kind):raise ValueError('cached training runner differs')
    external={k:v for k,v in original['pins'].items() if not Path(k).is_relative_to(old)}
    dataset=Path(job['collection'])/'reference-build/dataset'
    for name in ('MANIFEST.json','records.json','population.json'):
        external[reference['root']+'/'+name]=file_hash(dataset/name)
    names={'config.json',*pm['layout']['services'].values()}
    pins=dict(external,**{remote+'/'+name:h for name,h in actual.items() if name not in names})
    config=configured(original,old,remote,reference,kind);config['pins']=pins
    expected={'config.json':config}
    for name in pm['layout']['services'].values():
        expected[name]=rebase(read(parent/name),old,remote);expected[name]['pins']=pins
    for name,value in expected.items():
        if read(root/name)!=value:raise ValueError('cached training configuration differs: '+name)
    layout=copy.deepcopy(pm['layout'])
    layout['directories']=['run','supervision',*['service-'+s['name'] for s in pm['inference_services']]]
    if m['layout']!=layout:raise ValueError('cached training collection layout differs')
    return dict(status='cached_training_independently_audited_not_npu_admitted',
        manifest_sha256=c['manifest_sha256'],reference_manifest_sha256=reference['manifest_sha256'],
        source_policy_id=original['source_policy_id'],recordings=40513,actor_updates=2,
        finite_npu_dispatch_allowed=False,training_dispatch_allowed=False,npu_executed=False,
        provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
