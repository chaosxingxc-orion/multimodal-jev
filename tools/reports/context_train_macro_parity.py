"""Endpoint-bound TRAIN HF versus batch32 replica comparison client.

The launcher owns device, package and numerical admission. This client never
launches services, reads gold, trains, switches policies or relaxes exactness.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_train_parity_inputs import load as load_inputs,encoder
from tools.reports.context_transfer_macro_parity import hf_reference as shared_hf,compare
from tools.reports.build_context_reference_dataset import validate_endpoint
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_replica_macro_client import connect_pool
from tools.reports.context_macro_evaluation import evaluate
from tools.reports.audit_context_macro_evaluation import audit as audit_macros


def panel(c,bundle):
    values=load_inputs(c['train_inputs'])
    if values['source']['release_sha256']!=c['release_sha256']:
        raise ValueError('TRAIN panel release differs from endpoint')
    assets,encode=encoder(c['train_inputs'],processor_sha256=c['dual_runtime_identity']['processor_sha256'],
        decision_token_id=bundle['contract']['token']['id'],audio_token_id=c['audio_token_id'])
    receipts=[]
    for asset in assets:
        _,receipt=encode(asset)
        # Waveform provenance is not part of the original independent stream.
        receipts.append({k:v for k,v in receipt.items() if k!='audio_receipt'})
    expected={r['id']:r for r in receipts}
    def checked(asset):
        row,receipt=encode(asset)
        receipt={k:v for k,v in receipt.items() if k!='audio_receipt'}
        if receipt!=expected[asset['id']]:raise ValueError('TRAIN cache changed between phases')
        return row,receipt
    selected=dict(schema='jev-source-train-runtime-panel-v1',dataset='SLURP_clean_TRAIN60',
        candidate_ids=values['candidate_ids'],assets=assets,input_receipts=receipts,
        selection_sha256=digest(values['panel']),input_manifest_sha256=c['train_inputs']['manifest_sha256'],
        labels_used=False)
    return selected,checked


def hf_reference(c,selected,bundle,encode,output):
    config=dict(c,model=c['model'],snapshot=c['source_policy'],adapter=c['initial_adapter'],
        policy_id=bundle['policy_id'],contract=bundle['contract'])
    return shared_hf(config,{},selected,bundle,encode,output)


def run(c,provenance,phase):
    if phase not in ('hf','macro'):raise ValueError('explicit HF or macro phase required')
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('parity dependency changed: '+name)
    runtime=verify_runtime(c);client=c['macro_client']
    replicas=len(client['replicas']); macro_queries=client['max_queries']
    if client['runtime_identity']!=runtime:
        raise ValueError('parity requires the bound scoring runtime')
    from multimodal_jev.policy_identity import checkpoint_identity
    if checkpoint_identity(c['model'])[0]!=c['base_id']:raise ValueError('actual base differs')
    bundle=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    verify_native_adapter(c['initial_adapter'],bundle)
    commit=Path(c['stage1_commit'])
    if c['pins'].get(str(commit))!=file_hash(commit):raise ValueError('pinned endpoint commit required')
    checkpoint=json.loads(commit.read_text());validate_endpoint(checkpoint,c,bundle)
    if checkpoint['joint_policy_manifest_sha256']!=file_hash(Path(c['source_policy'])/'MANIFEST.json'):
        raise ValueError('actual full Stage1 endpoint required')
    selected,encode=panel(c,bundle)
    if len(selected['assets'])<macro_queries:
        raise ValueError('at least one complete declared parity macro required')
    root=Path(c['output']);check_paths({'train_parity':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'panel.json',selected);commit_json(root/'provenance.json',provenance)
    pool=None
    try:
        if phase=='hf':
            result=hf_reference(c,selected,bundle,encode,root)
            commit_json(root/'reference.json',result)
        else:
            entry=c['hf_reference']
            if file_hash(entry['path'])!=entry['sha256']:raise ValueError('HF reference changed')
            reference=json.loads(Path(entry['path']).read_text())
            if reference['policy_id']!=bundle['policy_id'] or reference['panel_sha256']!=digest(selected):
                raise ValueError('same endpoint and input panel required before connecting services')
            pool=connect_pool(client,bundle=bundle)
            evaluate(pool,selected['assets'],encode,candidate_ids=selected['candidate_ids'],
                policy_id=bundle['policy_id'],runtime_identity=runtime,audio_token_id=c['audio_token_id'],
                macro_queries=macro_queries,output=root/'inference')
            pool.close();pool=None
            proof=audit_macros(root/'inference',Path(client['journal_root'])/client['attempt_id'],
                population=selected['assets'],candidate_ids=selected['candidate_ids'],
                input_receipts=selected['input_receipts'],policy_id=bundle['policy_id'],runtime_identity=runtime,
                macro_queries=macro_queries,replicas=replicas)
            commit_json(root/'macro-output-audit.json',proof)
            records=json.loads((root/'inference/predictions.json').read_text())['records']
            result=compare(reference,records,selected,policy_id=bundle['policy_id'])
            result.update(hf_reference=entry,macro_output_audit_sha256=file_hash(root/'macro-output-audit.json'),
                training_dispatch_allowed=False)
            commit_json(root/'comparison.json',result)
            if not result['exact']:raise ValueError('TRAIN endpoint HF/replica logits or probability differ')
        commit_json(root/'terminal.json',dict(exit_code=0));return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if pool is not None:pool.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--phase',choices=['hf','macro'],required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance,a.phase),indent=2))
