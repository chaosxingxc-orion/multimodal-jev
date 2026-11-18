"""Bind any registered InfoNCE point to the unchanged four-card runtime.

Only the pinned evaluation client is added outside the model source tree.
HF panel comparison is performed for each point before publishing scores.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle,export_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_infonce_selection import reconstruct_plan
from tools.reports.prepare_context_stage2_runtime import audit as audit_parent
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.prepare_context_train_parity_job import runner as parity_runner
from tools.reports.context_device_visibility import visibility
from tools.reports.context_replica_training_operations import path
from tools.reports import context_replica_selection_client as client


def read(p):return json.loads(Path(p).read_text())


def inputs(c,provenance):
    plan=reconstruct_plan(c);proof=audit_parent(c['parent'],provenance)
    parent=Path(c['parent']['output']);m=read(parent/'PREPARED.json');training=read(parent/'config.json')
    points=[p for p in plan['checkpoints'] if p['step']==c['source_step']]
    if type(c['source_step']) is not int or len(points)!=1:raise ValueError('one registered source point required')
    point=points[0];snapshot=Path(point['snapshot'])
    bundle=read_execution_bundle(snapshot,expected_policy_id=point['policy_id'],
        expected_contract=plan['source_contract'],expected_base_id=plan['base_id'])
    if (file_hash(snapshot/'MANIFEST.json')!=point['manifest_sha256']
            or processor_cache_identity(snapshot)!=training['dual_runtime_identity']['processor_sha256']
            or training['source_policy_id']!=plan['reference_endpoint']['policy_id']
            or training['source_contract']!=plan['source_contract'] or training['base_id']!=plan['base_id']
            or c['selection_inputs']['release']!=training['release']
            or c['selection_inputs']['release_sha256']!=training['release_sha256']):
        raise ValueError('actual registered point and same source/runtime required')
    e=c['selection_inputs']
    population,_,candidates=client.inputs(e)
    receipts,evidence=client.load_evidence(e,population,
        release_manifest=read(Path(e['release'])/'MANIFEST.json'),
        processor_sha256=training['dual_runtime_identity']['processor_sha256'])
    selected=client.panel(population,candidates,receipts)
    return plan,proof,parent,m,training,point,bundle,selected,evidence


def runner(m):
    source=parity_runner(m)
    changes=[
        ("if c['expected_population']!=40513 or c['stage1_endpoint']!=318:",
         "if c['expected_population']!=9226 or c['source_step'] not in [0,*range(32,318,32),318]:"),
        ("r/'source/tools/reports/context_train_macro_parity.py'","r/'selection-client.py'"),
        ('TRAIN_MACRO_PARITY_CONFIG','REPLICA_SELECTION_CLIENT_CONFIG'),
    ]
    for before,after in changes:
        if source.count(before)!=1 and before!='TRAIN_MACRO_PARITY_CONFIG':
            raise ValueError('shared selection runner anchor changed: '+before)
        source=source.replace(before,after)
    # The generated macro command imports the pinned wrapper by file path; its
    # model dependencies are still exclusively the unchanged source manifest.
    old="import json,sys;from pathlib import Path;from multimodal_jev.baseline import file_hash;from tools.reports.context_train_macro_parity import run;"
    new="import json,sys,runpy;from pathlib import Path;from multimodal_jev.baseline import file_hash;run=runpy.run_path(str(Path(sys.argv[1]).parent/'selection-client.py'))['run'];"
    if source.count(old)!=1:raise ValueError('shared macro command anchor changed')
    source=source.replace(old,new)
    compile(source,'<replica-selection-runner>','exec');return source


def configured(c,training,m,point,remote):
    value=rebase(training,m['remote_root'],remote)
    value.update(source_policy_id=point['policy_id'],encoded_audio_cache=c['selection_inputs']['cache_root'])
    value.pop('reference_cache',None);value.pop('reference_dataset',None)
    # The inherited config exists only for the shared container/storage layer.
    value.update(training_dispatch_allowed=False,runtime_preparation_only=True)
    evaluate=copy.deepcopy(value)
    evaluate.update(snapshot=value['source_policy'],policy_id=point['policy_id'],contract=value['source_contract'],
        macro_queries=128,expected_population=9226,source_step=point['step'],max_model_len=4096)
    evaluate['macro_client'].update(attempt_id='selection-v1',journal_root=remote+'/selection-journal')
    visibility(m);target=m['training_devices'][0]
    mounted=sorted(m['training_devices']+[v['device'] for v in m['inference_services']])
    evaluate.update(physical_devices=[target],process_device_scope=dict(kind='same_container_runtime_visibility_v1',
        mounted_physical=mounted,visible_ordinal=str(mounted.index(target))))
    for key in ('cache_audit','cache_receipts'):
        name='original-cache-audit.json' if key=='cache_audit' else 'original-cache-stream.jsonl'
        evaluate[key]=dict(path=remote+'/selection-inputs/'+name,sha256=c['selection_inputs'][key]['sha256'])
    return value,evaluate


def prepare(c,provenance):
    plan,proof,parent,m,training,point,bundle,selected,evidence=inputs(c,provenance)
    output=Path(c['output']);remote=path(c['remote_root']);old=m['remote_root']
    protected=[parent,Path(point['snapshot']),Path(c['source_plan'])]
    if (output.exists() or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected)
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('fresh disjoint selection package required')
    check_paths({'selection_package':output});shutil.copytree(parent,output)
    (output/'PREPARED.json').unlink()
    for name in ('policy','adapter'):shutil.rmtree(output/'initial'/name)
    shutil.copytree(point['snapshot'],output/'initial/policy');export_native_adapter(output/'initial/adapter',bundle)
    (output/'initial/endpoint-commit.json').unlink()
    commit_json(output/'SOURCE-PLAN.json',plan);commit_json(output/'SELECTION-PARENT-AUDIT.json',proof)
    (output/'selection-inputs').mkdir()
    commit_json(output/'selection-inputs/panel.json',selected)
    commit_json(output/'selection-inputs/evidence.json',evidence)
    for key,name in [('cache_audit','original-cache-audit.json'),('cache_receipts','original-cache-stream.jsonl')]:
        shutil.copy2(c['selection_inputs'][key]['path'],output/'selection-inputs'/name)
    shutil.copy2(client.__file__,output/'selection-client.py')
    (output/'run.py').write_text(runner(m))
    (output/'parity-template.json').write_text('"${REPLICA_SELECTION_CLIENT_CONFIG}"\n')
    config,evaluate=configured(c,training,m,point,remote)
    names={'config.json','parity-hf.json','parity-macro.json','parity-hf-local.json',*m['layout']['services'].values()}
    external={k:v for k,v in training['pins'].items() if not Path(k).is_relative_to(old)}
    pins=dict(external,**{remote+'/'+str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and str(p.relative_to(output)) not in names})
    config['pins']=pins;evaluate['pins']=pins
    hf=dict(evaluate,output=remote+'/hf');macro=dict(evaluate,output=remote+'/evaluation')
    configs={'config.json':config,'parity-hf.json':hf,'parity-macro.json':macro,
        'parity-hf-local.json':dict(schema='jev-local-macros-v1',macros=dict(REPLICA_SELECTION_CLIENT_CONFIG=hf))}
    for name in m['layout']['services'].values():
        service=rebase(read(parent/name),old,remote);service['pins']=pins
        service['policy']['policy_id']=point['policy_id'];configs[name]=service
    for name,value in configs.items():(output/name).write_text(json.dumps(value,indent=2)+'\n')
    if {str(p.relative_to(output/'source')):file_hash(p) for p in (output/'source').rglob('*.py')}!=read(parent/'RUNTIME-SOURCES.json'):
        raise ValueError('selection changed sealed model runtime')
    receipt=dict(m,status='replica_infonce_selection_prepared_only',remote_root=remote,
        source_policy_id=point['policy_id'],source_step=point['step'],source_plan_sha256=c['source_plan_sha256'],
        parent_sha256=c['parent']['manifest_sha256'],recordings=9226,actor_updates=0,
        numerical_admission=False,training_dispatch_allowed=False,config_provenance=provenance)
    receipt['layout']=copy.deepcopy(m['layout'])
    receipt['layout']['configs']+=['parity-hf.json','parity-macro.json','parity-hf-local.json','parity-template.json','selection-client.py','SOURCE-PLAN.json']
    receipt['layout']['directories']=['hf','hf-process','evaluation','selection-journal','supervision',
        *['service-'+s['name'] for s in m['inference_services']]]
    receipt['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    commit_json(output/'PREPARED.json',receipt)
    return dict(status=receipt['status'],manifest_sha256=file_hash(output/'PREPARED.json'),
        npu_executed=False,training_dispatch_allowed=False)


def audit(c,provenance):
    root=Path(c['output']);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('selection manifest differs')
    m=read(manifest)
    if (any(p.is_symlink() for p in root.rglob('*'))
            or {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}!=m['files']):
        raise ValueError('complete unchanged selection package required')
    # Independently reconstruct all configs, wrapper, source and policy bytes.
    # Only this newly created temporary directory is deleted on completion.
    check_paths({'reconstruction_parent':root.parent})
    with tempfile.TemporaryDirectory(prefix='selection-reconstruct-',dir=root.parent) as tmp:
        spec=dict(c,output=str(Path(tmp)/'package'))
        prepared=prepare(spec,m['config_provenance'])
        expected=read(Path(spec['output'])/'PREPARED.json')
        if expected!=m or prepared['manifest_sha256']!=c['manifest_sha256']:
            raise ValueError('selection differs from independent reconstruction')
    return dict(status='replica_infonce_selection_package_audited_not_executed',
        source_step=m['source_step'],policy_id=m['source_policy_id'],recordings=9226,
        manifest_sha256=c['manifest_sha256'],runtime_identity=m['dual_runtime_identity'],
        numerical_admission=False,npu_executed=False,provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps((prepare if a.action=='prepare' else audit)(c,provenance),indent=2))
