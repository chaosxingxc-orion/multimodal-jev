"""Prepare a two-update DP2/dual-TP1 GRPO package from audited macro assets.

Preserves the full TRAIN schedule and Stage1 initialization; no deployment,
new data sampling, base-weight copy or DP4 optimizer-state migration.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_macro_trainer_candidate import generate, ALGORITHM
from tools.reports.audit_context_macro_probe import audit_outputs

ADDITIONS=(
    'src/multimodal_jev/slurp_portable_release.py',
    'tools/reports/context_macro_trainer_candidate.py',
    'tools/reports/context_dual_trainer_candidate.py',
    'tools/reports/context_dual_actor_candidate.py',
    'tools/reports/context_dual_session_candidate.py',
    'tools/reports/context_probability_policy_v2.py',
    'src/multimodal_jev/context_dual_credit.py',
    'src/multimodal_jev/context_lora_checkpoint_scope.py',
    'src/multimodal_jev/context_policy_snapshot.py',
    'src/multimodal_jev/context_checkpoint_policy.py',
    'src/multimodal_jev/context_policy_artifacts.py',
)


def prepare(c,provenance):
    parent=Path(c['macro_parent']);package=parent/'package';template=Path(c['training_parent'])
    receipt=json.loads((parent/'PREPARED.json').read_text())
    if verify(package)!=c['macro_manifest_sha256'] or receipt['files_sha256']!=c['macro_manifest_sha256']:
        raise ValueError('immutable macro parent required')
    audit_path=Path(c['macro_audit'])
    if file_hash(audit_path)!=c['macro_audit_sha256']:raise ValueError('macro audit changed')
    audited=json.loads(audit_path.read_text());actual=audit_outputs(package,c['macro_collection'])
    if (audited['files_sha256']!=receipt['files_sha256']
            or any(audited.get(k)!=v for k,v in actual.items())):
        raise ValueError('actual macro outputs differ from accepted audit')
    if file_hash(template/'PREPARED.json')!=c['training_parent_sha256']:
        raise ValueError('training parent identity changed')
    old_receipt=json.loads((template/'PREPARED.json').read_text())
    files={str(p.relative_to(template)):file_hash(p) for p in template.rglob('*')
           if p.is_file() and p!=template/'PREPARED.json'}
    if files!=old_receipt['files'] or any(p.is_symlink() for p in template.rglob('*')):
        raise ValueError('complete frozen training parent required')
    old=json.loads((template/'config.json').read_text());policy=json.loads((package/'service-0.json').read_text())['policy']
    if old['source_policy_id']!=policy['policy_id'] or old['source_contract']!=policy['contract']:
        raise ValueError('common Stage1 initialization required')
    output=Path(c['output']);remote=Path(c['remote_root']);ipc=remote/'ipc'
    if (not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or str(remote) in (receipt['remote_root'],old_receipt['remote_root'])
            or len(str(ipc).encode())+8>107 or output.exists()
            or output.resolve().is_relative_to(parent.resolve())
            or output.resolve().is_relative_to(template.resolve())):
        raise ValueError('new isolated package and runtime required')
    project=Path(c['project']);generate(project)
    output.mkdir(parents=True);dest=output/'package';shutil.copytree(package,dest)
    for name in ('FILES.json','RUNTIME-SOURCES.json','measurement.json','service-0.json','service-1.json'):
        (dest/name).unlink()
    for name in ADDITIONS:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,target)
    shutil.copytree(template/'source/schedule',dest/'schedule')
    shutil.copy2(audit_path,dest/'MACRO-AUDIT.json')
    generate(dest)
    sources={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')}
    (dest/'RUNTIME-SOURCES.json').write_text(json.dumps(sources,sort_keys=True,indent=2)+'\n')
    runtime=dict(receipt['runtime_identity'],source_manifest_sha256=file_hash(dest/'RUNTIME-SOURCES.json'))
    root=str(remote/'package');policy=dict(policy,snapshot=root+'/policy',adapter=root+'/adapter')
    client=dict(schema='jev-macro-client-v1',attempt_id='training-prefix-v1',
        journal_root=str(remote/'journal'),runtime_identity=runtime,timeout_seconds=900,
        rollout_queries=32,max_queries=128,replicas=[dict(socket=str(ipc/f'r{i}.sock'),
            exchange=str(ipc/f'exchange-{i}')) for i in range(2)])
    config=copy.deepcopy(old)
    config.update(source_root=root,source_policy=policy['snapshot'],initial_adapter=policy['adapter'],
        reference_adapter=policy['adapter'],schedule=root+'/schedule',physical_devices=[4,5],
        global_batch=128,microbatch=8,algorithm=ALGORITHM,dual_arm='joint',start_update=0,stop_update=2,
        checkpoint_interval=1,engineering_prefix=True,training_dispatch_allowed=False,
        reference_cache=str(remote/'reference-cache'),macro_client=client,
        dual_runtime_identity=runtime,dual_runtime_manifest=root+'/RUNTIME-SOURCES.json')
    for key in ('socket_path','exchange','resume_root'):config.pop(key,None)
    initial=str(Path(old['source_policy']).parent)
    external={p:h for p,h in old['pins'].items()
        if not Path(p).is_relative_to(old_receipt['remote_root']) and not Path(p).is_relative_to(initial)}
    from multimodal_jev.slurp_portable_release import load as load_portable, SCHEMA
    release=Path(c['release']);digest=c['release_sha256']
    if not release.is_relative_to(Path(c['shared_root'])):raise ValueError('shared portable dataset required')
    rows,task,publication=load_portable(release,digest,'learning',purpose='optimize')
    if (publication['schema']!=SCHEMA or len(rows)!=40513
            or publication['migration']['source_manifest_sha256']!=old['release_sha256']):
        raise ValueError('byte-preserving portable migration of frozen population required')
    previous=json.loads((Path(old['release'])/'MANIFEST.json').read_text())
    expected={n:h for n,h in previous['files'].items() if n!='publication-config.json'}
    if publication['migration']['unchanged_payload_sha256']!=expected:
        raise ValueError('portable migration changed admitted payload')
    config.update(release=str(release),release_sha256=digest,schedule_release_sha256=old['release_sha256'])
    for path in release.rglob('*'):
        if path.is_file():external[str(path)]=file_hash(path)
    pins={root+'/'+str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    config['pins']=dict(external,**pins)
    for i in range(2):
        endpoint=client['replicas'][i]
        service=dict(source_root=root,source_policy=policy['snapshot'],dual_runtime_identity=runtime,
            dual_runtime_manifest=root+'/RUNTIME-SOURCES.json',pins=pins,
            transport_protocol='bulk_dual_logits_candidate_v1',socket_path=endpoint['socket'],exchange=endpoint['exchange'],
            policy=dict(policy,physical_devices=[6+i]))
        (dest/f'service-{i}.json').write_text(json.dumps(service,indent=2)+'\n')
    (dest/'training.json').write_text(json.dumps(config,indent=2)+'\n')
    files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'FILES.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
    result=dict(status='macro_training_prefix_prepared_only',image=receipt['image'],
        files_sha256=verify(dest),remote_root=str(remote),ipc_root=str(ipc),runtime_identity=runtime,
        parent_macro_manifest=receipt['files_sha256'],parent_macro_audit=c['macro_audit_sha256'],
        training_parent_sha256=c['training_parent_sha256'],config_provenance=provenance,
        training_dispatch_allowed=False,finite_npu_dispatch_allowed=False,
        pending=['fixed-image integrated imports','three-container DP2 training launcher and full mount verification',
                 'real BF16/audio two-update and dual activation audit','fresh-process recovery'])
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
