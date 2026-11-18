"""Prepare complete Stage1 selection evaluation using frozen dual-service code.

No deployment or policy selection. Copies only the existing adapter/head policy
package and input evidence; the base model and audio cache remain shared assets.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.selection_macro_cache_operations import audit_receipts,IMAGE

ADDITIONS=(
    'tools/reports/context_macro_evaluation.py',
    'tools/reports/evaluate_context_selection_macro.py',
    'tools/reports/context_selection_audited_cache.py',
    'tools/jev/evaluate_context_event_records.py',
)


def prepare(c,provenance,*,source_point=None,source_plan=None):
    parent=Path(c['parent']);source=parent/'package';output=Path(c['output'])
    receipt=json.loads((parent/'PREPARED.json').read_text())
    if verify(source)!=c['parent_manifest_sha256'] or receipt['image']!=IMAGE:
        raise ValueError('fixed-image frozen parent package required')
    if file_hash(c['parent_audit'])!=c['parent_audit_sha256']:raise ValueError('parent audit changed')
    proof=json.loads(Path(c['parent_audit']).read_text())
    expected_status='formal_dp2_outputs_audited' if source_point is not None else 'finite_dp2_training_audited'
    if proof['status']!=expected_status or proof['files_sha256']!=c['parent_manifest_sha256']:
        raise ValueError('audited DP2 training parent required')
    if file_hash(c['cache_audit'])!=c['cache_audit_sha256']:raise ValueError('cache verification audit changed')
    cache_proof=json.loads(Path(c['cache_audit']).read_text())
    actual=audit_receipts(c['binding_root'],c['cache_collection'],c['binding_sha256'])
    if (any(cache_proof.get(k)!=v for k,v in actual.items()) or cache_proof['image']!=IMAGE
            or cache_proof['state']['Status']!='exited' or cache_proof['state']['ExitCode']!=0
            or cache_proof['state']['OOMKilled']):raise ValueError('actual complete cache verification required')
    remote=Path(c['remote_root'])
    if (not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or len(str(remote/'ipc').encode())+8>107 or str(remote)==receipt['remote_root']
            or output.exists() or output.resolve().is_relative_to(parent.resolve())):
        raise ValueError('new isolated evaluation package/runtime required')
    old=json.loads((source/'training.json').read_text())
    if old['release_sha256']!=c['release_sha256'] or old['release']!=c['release']:
        raise ValueError('same portable source population required')
    initial=json.loads((source/'policy/MANIFEST.json').read_text())
    if (initial['policy_id']!=old['source_policy_id'] or initial['contract']!=old['source_contract']
            or any(json.loads((source/f'service-{i}.json').read_text())['policy']['policy_id']!=initial['policy_id']
                   for i in range(2))):
        raise ValueError('common Stage1 policy required, not a replay or updated policy')
    if source_point is not None:
        from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
        from multimodal_jev.cached_context_preparation import processor_cache_identity
        from multimodal_jev.local_runtime_storage import check_paths
        old_remote=Path(receipt['remote_root'])
        if remote.is_relative_to(old_remote) or old_remote.is_relative_to(remote):
            raise ValueError('source evaluation runtime must be disjoint from training')
        if (source_plan is None or source_point not in source_plan['checkpoints']
                or source_plan['initial_policy_id']!=initial['policy_id']
                or source_plan['source_audit']!=proof
                or source_plan['source_operations']['prepared']!=str(parent)):
            raise ValueError('audited full source plan required for updated selection policy')
        snapshot=Path(source_point['snapshot']);adapter=Path(source_point['adapter'])
        bundle=read_execution_bundle(snapshot,expected_policy_id=source_point['policy_id'],
            expected_contract=old['source_contract'],expected_base_id=old['base_id'])
        verify_native_adapter(adapter,bundle)
        if (file_hash(snapshot/'MANIFEST.json')!=source_point['manifest_sha256']
                or processor_cache_identity(snapshot)!=processor_cache_identity(source/'policy')):
            raise ValueError('source policy or reusable processor differs')
        check_paths({'selection_prepared':output})
    output.mkdir(parents=True);dest=output/'package';shutil.copytree(source,dest)
    for name in ('FILES.json','RUNTIME-SOURCES.json','training.json','service-0.json','service-1.json'):
        (dest/name).unlink()
    if source_point is not None:
        # Only cloned policy assets are replaced; immutable parents stay untouched.
        for directory,origin in (('policy',snapshot),('adapter',adapter)):
            shutil.rmtree(dest/directory);shutil.copytree(origin,dest/directory)
        (dest/'SOURCE-PLAN.json').write_text(json.dumps(source_plan,indent=2)+'\n')
    for name in ADDITIONS:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(c['project'])/name,target)
    evidence=dest/'evaluation-inputs';evidence.mkdir()
    shutil.copytree(c['binding_root'],evidence/'binding')
    shutil.copy2(Path(c['cache_collection'])/'input-receipts.json',evidence/'input-receipts.json')
    shutil.copy2(c['cache_audit'],evidence/'cache-verification.json')
    binding=json.loads((evidence/'binding/binding.json').read_text())
    for key,name in [('cache_audit','original-cache-audit.json'),('cache_receipts','original-cache-stream.jsonl')]:
        entry=binding[key]
        if file_hash(entry['path'])!=entry['sha256']:raise ValueError('original input evidence changed')
        shutil.copy2(entry['path'],evidence/name)
    shutil.copy2(c['parent_audit'],dest/'TRAINING-AUDIT.json')
    sources={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')}
    (dest/'RUNTIME-SOURCES.json').write_text(json.dumps(sources,sort_keys=True,indent=2)+'\n')
    runtime=dict(receipt['runtime_identity'],source_manifest_sha256=file_hash(dest/'RUNTIME-SOURCES.json'))
    root=str(remote/'package');ipc=str(remote/'ipc');policies=[]
    pins={root+'/'+str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    endpoints=[dict(socket=ipc+f'/r{i}.sock',exchange=ipc+f'/exchange-{i}') for i in range(2)]
    for i in range(2):
        service=json.loads((source/f'service-{i}.json').read_text())
        service.update(source_root=root,source_policy=root+'/policy',dual_runtime_manifest=root+'/RUNTIME-SOURCES.json',
            dual_runtime_identity=runtime,pins=pins,socket_path=endpoints[i]['socket'],exchange=endpoints[i]['exchange'])
        service['policy'].update(snapshot=root+'/policy',adapter=root+'/adapter')
        if source_point is not None:service['policy']['policy_id']=source_point['policy_id']
        policies.append(service['policy'])
        (dest/f'service-{i}.json').write_text(json.dumps(service,indent=2)+'\n')
    policy=policies[0]
    client=dict(schema='jev-macro-client-v1',attempt_id='selection-v1',journal_root=str(remote/'journal'),
        runtime_identity=runtime,replicas=endpoints,timeout_seconds=900,rollout_queries=32,max_queries=128)
    config=dict(release=c['release'],release_sha256=c['release_sha256'],snapshot=root+'/policy',
        policy_id=policy['policy_id'],contract=policy['contract'],base_id=policy['base_id'],
        encoded_audio_cache=c['cache_root'],output=str(remote/'evaluation'),pins=pins,macro_client=client,
        audio_token_id=151676,macro_queries=128,
        cache_audit=dict(path=root+'/evaluation-inputs/original-cache-audit.json',sha256=binding['cache_audit']['sha256']),
        cache_receipts=dict(path=root+'/evaluation-inputs/original-cache-stream.jsonl',sha256=binding['cache_receipts']['sha256']))
    (dest/'evaluation.json').write_text(json.dumps(config,indent=2)+'\n')
    files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'FILES.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
    result=dict(status='macro_selection_prepared_only',image=IMAGE,files_sha256=verify(dest),
        remote_root=str(remote),ipc_root=ipc,runtime_identity=runtime,device_scope=[[6],[7]],
        training_dispatch_allowed=False,policy_id=policy['policy_id'],policy_role='common_Stage1',
        parent_manifest_sha256=c['parent_manifest_sha256'],parent_audit_sha256=c['parent_audit_sha256'],
        original_runtime_sources_sha256=receipt['runtime_identity']['source_manifest_sha256'],
        source_changes=list(ADDITIONS),selection_recordings=9226,cache_verification_sha256=c['cache_audit_sha256'],
        config_provenance=provenance,scope='Prepared full Stage1 selection; deployment, actual macro execution and independent scoring pending')
    if source_point is not None:
        result.update(status='macro_source_point_prepared_only',policy_role='source_curve_point',
            source_step=source_point['step'],source_plan_sha256=file_hash(dest/'SOURCE-PLAN.json'),
            scope='Prepared full selection for one audited formal source curve point; no score or target result')
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();c,p=load_config(args.config,args.local);print(json.dumps(prepare(c,p),indent=2))
