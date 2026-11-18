"""Seal one source-frozen role for full target inference and same-policy HF parity."""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_transfer_source_admission import admit,seal
from tools.reports.context_transfer_portable_inputs import load
from tools.reports.audit_context_transfer_input_coverage import audit as coverage
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_full_transfer_operations import validate,IMAGE

ADDITIONS=(
    'tools/reports/context_transfer_portable_inputs.py',
    'tools/reports/evaluate_context_portable_transfer.py',
    'tools/reports/context_transfer_parity_inputs.py',
    'tools/reports/context_transfer_macro_parity.py',
    'tools/reports/context_transfer_macro_inputs.py',
    'tools/reports/context_selection_audited_cache.py',
    'tools/reports/context_macro_evaluation.py',
    'tools/reports/audit_context_macro_evaluation.py',
    'tools/reports/evaluate_context_transfer_macro.py',
    'tools/reports/audit_context_transfer_macro.py',
)


def prepare(c,provenance):
    source_admission=admit(c['source_admission'])
    parent=source_admission['source_operations'];training=source_admission['training']
    origin=Path(parent['prepared'])/'package';receipt=json.loads((origin.parent/'PREPARED.json').read_text())
    if verify(origin)!=parent['manifest_sha256'] or receipt['image']!=IMAGE:
        raise ValueError('immutable formal training parent required')
    admitted=load(c['portable_full_inputs']);coverage(c['portable_full_inputs'],c['portable_parity_inputs'])
    out=Path(c['output']);root=c['remote_root'];point=source_admission['point']
    if (out.exists() or out.resolve().is_relative_to(origin.resolve())
            or origin.resolve().is_relative_to(out.resolve()) or not Path(root).is_absolute()
            or '..' in Path(root).parts or ':' in root or len((root+'/ipc').encode())+8>107
            or Path(root).is_relative_to(parent['remote_root']) or Path(parent['remote_root']).is_relative_to(root)
            or admitted['processor_sha256']!=receipt['runtime_identity']['processor_sha256']):
        raise ValueError('new isolated target package with matching processor required')
    check_paths({'full_target_prepared':out,'full_target_operations':c['operations_root']})
    dest=out/'package';shutil.copytree(origin,dest)
    for name in ('FILES.json','training.json','RUNTIME-SOURCES.json','service-0.json','service-1.json'):
        (dest/name).unlink()
    for name,source in [('policy',point['snapshot']),('adapter',point['adapter'])]:
        shutil.rmtree(dest/name);shutil.copytree(source,dest/name)
    for name in ADDITIONS:
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(c['project'])/name,p)
    shutil.copytree(c['portable_full_inputs']['root'],dest/'full-inputs')
    shutil.copytree(c['portable_parity_inputs']['root'],dest/'parity-inputs')
    source_entry=seal(source_admission,dest/'source',root+'/package/source')
    write(dest/'SOURCE-ADMISSION.json',source_admission)
    write(dest/'RUNTIME-SOURCES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')})
    runtime=dict(receipt['runtime_identity'],source_manifest_sha256=file_hash(dest/'RUNTIME-SOURCES.json'))
    pins={root+'/package/'+str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    endpoints=[dict(socket=root+f'/ipc/r{i}.sock',exchange=root+f'/ipc/e{i}') for i in range(2)]
    for i in range(2):
        s=relocate(json.loads((origin/f'service-{i}.json').read_text()),parent['remote_root'],root)
        s.update(pins=pins,dual_runtime_identity=runtime,socket_path=endpoints[i]['socket'],exchange=endpoints[i]['exchange'])
        s['policy'].update(policy_id=point['policy_id'],max_decisions=len(admitted['candidate_ids']),
            max_batch_queries=32,max_batch_tokens=131072)
        write(dest/f'service-{i}.json',s)
    common=dict(dataset=admitted['dataset'],model=parent['model'],base_id=training['base_id'],
        policy_id=point['policy_id'],contract=training['source_contract'],snapshot=root+'/package/policy',
        adapter=root+'/package/adapter',audio_token_id=151676,pins=pins)
    evaluation=dict(common,policy_role=c['source_admission']['policy_role'],source_evaluation=source_entry,
        portable_full_inputs=dict(c['portable_full_inputs'],root=root+'/package/full-inputs'),
        output=root+'/evaluation',macro_queries=128,
        macro_client=dict(schema='jev-macro-client-v1',attempt_id='full-transfer-v1',journal_root=root+'/journal',
            runtime_identity=runtime,replicas=endpoints,timeout_seconds=900,rollout_queries=32,max_queries=128))
    if admitted['source_release'] is not None:evaluation['source_release']=admitted['source_release']
    hf=dict(common,portable_parity_inputs=dict(c['portable_parity_inputs'],root=root+'/package/parity-inputs'),
        output=root+'/hf',physical_devices=[6])
    write(dest/'evaluation.json',evaluation);write(dest/'hf.json',hf)
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    result=dict(status='full_portable_transfer_prepared_only',image=IMAGE,files_sha256=verify(dest),
        remote_root=root,ipc_root=root+'/ipc',runtime_identity=runtime,training_dispatch_allowed=False,
        policy_id=point['policy_id'],policy_role=evaluation['policy_role'],dataset=evaluation['dataset'],
        recordings=len(admitted['population']),parent_manifest_sha256=parent['manifest_sha256'],
        source_changes=list(ADDITIONS),config_provenance=provenance)
    write(out/'PREPARED.json',result)
    ops=dict(prepared=str(out),manifest_sha256=result['files_sha256'],remote_root=root,host=parent['host'],
        tag=c['tag'],model=parent['model'],driver_root=parent['driver_root'],lease=parent['lease'],
        cache_root=c['portable_full_inputs']['cache_root'],operations_root=c['operations_root'],
        source_admission=c['source_admission'],finite_inference_dispatch_allowed=False)
    validate(ops);write(out/'operations.json',ops)
    return dict(prepared=str(out),manifest_sha256=result['files_sha256'],dataset=evaluation['dataset'],
        policy_id=point['policy_id'],policy_role=evaluation['policy_role'],launched=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(prepare(c,p),indent=2))
