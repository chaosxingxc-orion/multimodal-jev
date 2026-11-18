"""Full SLURP selection on the sealed replica transport with per-policy parity.

HF and macro phases run in separate owned processes. The macro phase scores
only after its input-only HF panel matches exactly. Container/source/policy
admission and independent collection remain launcher obligations.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_replica_macro_client import connect_pool
from tools.reports.context_macro_evaluation import evaluate
from tools.reports.context_selection_audited_cache import load_evidence,encoder as original_encoder
from tools.reports.context_train_parity_inputs import select
from tools.reports.context_train_macro_parity import hf_reference
from tools.reports.context_transfer_macro_parity import compare
from tools.reports.evaluate_context_selection_macro import inputs
from tools.jev.evaluate_context_event_records import evaluate as score


def read(path):return json.loads(Path(path).read_text())


def panel(population,candidates,receipts):
    if (len(population)!=9226 or len(receipts)!=9226 or len(candidates)!=60
            or len(set(candidates))!=60 or len({a['id'] for a in population})!=9226
            or [r['id'] for r in receipts]!=[a['id'] for a in population]):
        raise ValueError('complete aligned selection population and native60 candidates required')
    for asset,receipt in zip(population,receipts,strict=True):
        if set(asset)!={'id','audio_sha256','request_sha256'} or any(receipt[k]!=v for k,v in asset.items()):
            raise ValueError('label-free selection identity required')
    by_id={r['id']:r for r in receipts};assets={a['id']:a for a in population}
    selected=select(by_id)
    bound=[]
    for sample in selected['ids']:
        receipt=by_id[sample];shape=receipt['tensors']['positions']['shape']
        if shape!=[1,60]:raise ValueError('complete native marker panel required')
        # Actual positions are authenticated when tensors are loaded, rather
        # than inferred from a hash here.
        bound.append(assets[sample])
    return dict(schema='jev-selection-parity-panel-v1',dataset='SLURP_clean_SELECTION60',
        candidate_ids=candidates,assets=bound,selection=selected,labels_used=False)


def prepare_inputs(c,bundle):
    population,requests,candidates=inputs(c)
    processor=processor_cache_identity(c['snapshot'])
    if processor!=c['dual_runtime_identity']['processor_sha256']:
        raise ValueError('same runtime and evaluation processor required')
    receipts,proof=load_evidence(c,population,
        release_manifest=read(Path(c['release'])/'MANIFEST.json'),processor_sha256=processor)
    readonly=FunctionType(original_encoder.__code__,dict(original_encoder.__globals__,
        EncodedAudioCache=ReadOnlyEncodedAudioCache))
    encode=readonly(c['encoded_audio_cache'],requests,candidates,receipts,processor_sha256=processor,
        decision_token_id=bundle['contract']['token']['id'],audio_token_id=c['audio_token_id'])
    selected=panel(population,candidates,receipts)
    def checked(asset):
        row,receipt=encode(asset)
        return row,{k:v for k,v in receipt.items() if k!='audio_receipt'}
    selected['input_receipts']=[checked(asset)[1] for asset in selected['assets']]
    return population,candidates,selected,checked,proof


def prepare_panel(c,provenance):
    """Freeze input-only panel membership; no remote cache read or inference."""
    population,_,candidates=inputs(c)
    processor=processor_cache_identity(c['snapshot'])
    receipts,proof=load_evidence(c,population,
        release_manifest=read(Path(c['release'])/'MANIFEST.json'),processor_sha256=processor)
    selected=panel(population,candidates,receipts)
    root=Path(c['output']);check_paths({'selection_panel':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'panel.json',selected)
    result=dict(status='selection_panel_prepared_not_numerically_admitted',recordings=9226,
        panel_queries=len(selected['assets']),panel_sha256=file_hash(root/'panel.json'),
        source_release_sha256=c['release_sha256'],processor_sha256=processor,input_evidence=proof,
        provenance=provenance,labels_used=False,npu_executed=False,cache_bytes_verified=False)
    commit_json(root/'result.json',result);return result


def validate_config(c):
    client=c['macro_client']
    if (client['schema']!='jev-replica-macro-client-v1' or client['max_queries']!=128
            or client['rollout_queries']!=32 or len(client['replicas'])!=2
            or client['runtime_identity']!=c['dual_runtime_identity'] or c['macro_queries']!=128
            or c['source_policy']!=c['snapshot'] or c['source_policy_id']!=c['policy_id']
            or c['source_contract']!=c['contract']):
        raise ValueError('same policy/runtime and replica macro128 rollout32 required')
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('selection dependency changed: '+name)
    verify_runtime(c)
    bundle=read_execution_bundle(c['snapshot'],expected_policy_id=c['policy_id'],
        expected_contract=c['contract'],expected_base_id=c['base_id'])
    verify_native_adapter(c['initial_adapter'],bundle)
    return bundle


def parity(reference,records,selected,policy):
    by_id={r['id']:r for r in records}
    if len(by_id)!=len(records):raise ValueError('unique full evaluation predictions required')
    actual=[by_id[a['id']] for a in selected['assets']]
    proof=compare(reference,actual,selected,policy_id=policy)
    if proof['exact'] is not True:raise ValueError('exact per-policy selection HF/replica parity required')
    return proof


def run(c,provenance,phase):
    if phase not in ('hf','macro'):raise ValueError('explicit selection HF or macro phase required')
    bundle=validate_config(c)
    population,candidates,selected,encode,evidence=prepare_inputs(c,bundle)
    root=Path(c['output']);check_paths({'selection':root});root.mkdir(parents=True,exist_ok=False)
    pool=None
    try:
        commit_json(root/'panel.json',selected);commit_json(root/'provenance.json',provenance)
        commit_json(root/'input-evidence.json',evidence)
        if phase=='hf':
            result=hf_reference(c,selected,bundle,encode,root)
            commit_json(root/'reference.json',result)
        else:
            entry=c['hf_reference']
            if file_hash(entry['path'])!=entry['sha256']:raise ValueError('HF selection reference changed')
            reference=read(entry['path'])
            if (reference['policy_id']!=c['policy_id'] or reference['panel_sha256']!=digest(selected)):
                raise ValueError('same-policy input panel required before service connection')
            pool=connect_pool(c['macro_client'],bundle=bundle)
            inference=evaluate(pool,population,encode,candidate_ids=candidates,policy_id=c['policy_id'],
                runtime_identity=c['dual_runtime_identity'],audio_token_id=c['audio_token_id'],
                macro_queries=128,output=root/'inference')
            pool.close();pool=None
            records=read(root/'inference/predictions.json')['records']
            if [r['id'] for r in records]!=[a['id'] for a in population]:
                raise ValueError('complete ordered selection predictions required before scoring')
            comparison=parity(reference,records,selected,c['policy_id'])
            commit_json(root/'parity.json',comparison)
            result=score(c['release'],c['release_sha256'],root/'inference/predictions.json',
                inference['predictions_sha256'],c['policy_id'],'selection')
            result.update(input_receipts_sha256=inference['input_receipts_sha256'],
                inference_result_sha256=file_hash(root/'inference/result.json'),
                parity_sha256=file_hash(root/'parity.json'))
            commit_json(root/'result.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0));return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if pool is not None:pool.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--phase',choices=['prepare','hf','macro'],required=True);a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    result=prepare_panel(c,provenance) if a.phase=='prepare' else run(c,provenance,a.phase)
    print(json.dumps(result,indent=2))
