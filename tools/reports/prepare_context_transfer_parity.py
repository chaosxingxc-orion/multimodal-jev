"""Seal HF-first target-width parity using the unchanged batch32 service runtime."""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_macro_training_operations import validate as validate_parent
from tools.reports.context_transfer_parity_inputs import load as load_inputs
from tools.reports.context_transfer_parity_operations import validate,IMAGE
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify

ADDITIONS=(
    'tools/reports/context_transfer_parity_inputs.py',
    'tools/reports/context_transfer_macro_parity.py',
    'tools/reports/context_transfer_macro_inputs.py',
    'tools/reports/context_selection_audited_cache.py',
    'tools/reports/context_macro_evaluation.py',
    'tools/reports/audit_context_macro_evaluation.py',
    'tools/reports/evaluate_context_transfer_macro.py',
)


def prepare(c,provenance):
    parent=json.loads(Path(c['parent_operations']).read_text());r=validate_parent(parent)
    source=Path(parent['prepared'])/'package';root=c['remote_root'];out=Path(c['output'])
    admitted=load_inputs(c['portable_parity_inputs'])
    check_paths({'prepared':out,'operations':c['operations_root']})
    if (out.exists() or out.resolve().is_relative_to(source.resolve())
            or not Path(root).is_absolute() or '..' in Path(root).parts or ':' in root
            or Path(root).is_relative_to(parent['remote_root']) or Path(parent['remote_root']).is_relative_to(root)
            or len((root+'/ipc').encode())+8>107):
        raise ValueError('new disjoint bounded parity package/runtime required')
    old=json.loads((source/'training.json').read_text())
    if (old['source_policy_id']!=json.loads((source/'policy/MANIFEST.json').read_text())['policy_id']
            or old.get('resume_root') or old['start_update']!=0
            or admitted['processor_sha256']!=r['runtime_identity']['processor_sha256']):
        raise ValueError('common initial policy and identical processor required')
    for i in range(2):
        p=json.loads((source/f'service-{i}.json').read_text())['policy']
        if p['max_batch_queries']!=32 or p['max_batch_tokens']!=131072:
            raise ValueError('sealed service32 parent required')
    dest=out/'package';shutil.copytree(source,dest)
    for name in ('FILES.json','training.json','service-0.json','service-1.json','RUNTIME-SOURCES.json'):(dest/name).unlink()
    for name in ADDITIONS:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(c['project'])/name,target)
    shutil.copytree(c['portable_parity_inputs']['root'],dest/'parity-inputs')
    write(dest/'RUNTIME-SOURCES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')})
    runtime=dict(r['runtime_identity'],source_manifest_sha256=file_hash(dest/'RUNTIME-SOURCES.json'))
    pins={root+'/package/'+str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}
    endpoints=[dict(socket=root+f'/ipc/r{i}.sock',exchange=root+f'/ipc/e{i}') for i in range(2)]
    for i in range(2):
        s=relocate(json.loads((source/f'service-{i}.json').read_text()),parent['remote_root'],root)
        s.update(pins=pins,dual_runtime_identity=runtime,socket_path=endpoints[i]['socket'],exchange=endpoints[i]['exchange'])
        s['policy']['max_decisions']=len(admitted['candidate_ids'])
        write(dest/f'service-{i}.json',s)
    e=dict(dataset=admitted['dataset'],portable_parity_inputs=dict(c['portable_parity_inputs'],root=root+'/package/parity-inputs'),
        model=parent['model'],base_id=old['base_id'],snapshot=root+'/package/policy',adapter=root+'/package/adapter',
        policy_id=old['source_policy_id'],contract=old['source_contract'],output=root+'/evaluation',pins=pins,
        audio_token_id=151676,macro_client=dict(schema='jev-macro-client-v1',attempt_id='target-parity-v1',
            journal_root=root+'/journal',runtime_identity=runtime,replicas=endpoints,
            timeout_seconds=900,rollout_queries=32,max_queries=128))
    write(dest/'evaluation.json',e)
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    receipt=dict(status='macro_target_parity_prepared_only',image=IMAGE,files_sha256=verify(dest),
        remote_root=root,ipc_root=root+'/ipc',runtime_identity=runtime,training_dispatch_allowed=False,
        policy_id=e['policy_id'],dataset=e['dataset'],recordings=len(admitted['population']),
        parent_manifest_sha256=parent['manifest_sha256'],source_changes=list(ADDITIONS),
        config_provenance=provenance,pending=['actual isolated HF and dual TP1 execution','independent transport/output audit'])
    write(out/'PREPARED.json',receipt)
    ops=dict(prepared=str(out),manifest_sha256=receipt['files_sha256'],remote_root=root,
        host=parent['host'],tag=c['tag'],model=parent['model'],driver_root=parent['driver_root'],lease=parent['lease'],
        cache_root=c['portable_parity_inputs']['cache_root'],operations_root=c['operations_root'],
        finite_inference_dispatch_allowed=False)
    validate(ops);write(out/'operations.json',ops)
    return dict(prepared=str(out),manifest_sha256=receipt['files_sha256'],dataset=e['dataset'],launched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
