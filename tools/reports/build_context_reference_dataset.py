"""Configured full TRAIN reference build using already admitted replica services."""
import argparse
import json
from pathlib import Path
import time
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_reference_builder import build
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from multimodal_jev.slurp_portable_release import load
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_replica_macro_client import connect_pool


def validate_endpoint(checkpoint,c,bundle):
    from multimodal_jev.context_topology_batch_plan import derive
    contract=checkpoint['contract'];plan=derive(contract['batch_config'])
    if (checkpoint['updates']!=c['stage1_endpoint'] or checkpoint['updates']!=plan['total_updates']
            or checkpoint['joint_policy']['policy_id']!=bundle['policy_id']
            or contract['stage']!='instruction' or contract['world_size']!=plan['dp_replicas']
            or contract['batch_config']['inference_instances']!=0 or contract['source_contract']!=bundle['contract']
            or contract['objective']['kind'] not in ('candidate_infonce','candidate_infonce_probability_margin_v1')
            or contract['objective']['probability_coefficient']!=0.
            or contract['release_sha256']!=c['release_sha256']
            or contract['base_id']!=c['base_id']
            or contract.get('prompt_layout')!=c.get('prompt_layout')
            or contract['batch_config']['source_queries']!=c['expected_population']):
        raise ValueError('full InfoNCE Stage1 endpoint on this TRAIN release required')
    if contract['objective']['kind']=='candidate_infonce_probability_margin_v1':
        from multimodal_jev.context_infonce import validate_objective
        validate_objective(contract['objective'])


def run(c,provenance,*,resume_pool=None):
    prefetch=c.get('input_prefetch')
    if prefetch is not None:
        from multimodal_jev.context_reference_builder import validate_prefetch
        prefetch=validate_prefetch(prefetch,c['macro_client']['max_queries'])
        torch.set_num_threads(1)
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('reference dependency changed: '+name)
    runtime=verify_runtime(c)
    if c['macro_client']['runtime_identity']!=runtime:
        raise ValueError('producer runtime differs from macro client')
    bundle=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    # Policy comes from a predeclared Stage1 endpoint, never a prior experiment.
    commit=Path(c['stage1_commit'])
    if c['pins'].get(str(commit))!=file_hash(commit):raise ValueError('pinned Stage1 endpoint required')
    checkpoint=json.loads(commit.read_text())
    validate_endpoint(checkpoint,c,bundle)
    if checkpoint['joint_policy_manifest_sha256']!=file_hash(Path(c['source_policy'])/'MANIFEST.json'):
        raise ValueError('actual Stage1 endpoint snapshot differs from commit')
    rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
    if len(rows)!=c['expected_population']:raise ValueError('full TRAIN population required')
    from tools.reports.context_reference_input_evidence import load as load_input_evidence,AuditedPreparer
    processor=processor_cache_identity(c['source_policy'])
    evidence,input_proof=load_input_evidence(c,rows=rows,task=task,
        release_sha256=c['release_sha256'],processor_sha256=processor)
    assets={r['id']:{k:r[k] for k in ('audio','audio_sha256')} for r in rows}
    if len(assets)!=len(rows):raise ValueError('distinct sample IDs required')
    candidates=[dict(id=v['id'],definition=v['description']) for v in task['schema']['candidates']]
    def no_processor():raise ValueError('offline input cache miss; no online preprocessing')
    preparer=CachedContextPreparer(no_processor,ReadOnlyEncodedAudioCache(c['encoded_audio_cache']),
        processor_sha256=processor,
        decision_token_id=c['source_contract']['token']['id'],audio_token_id=c['audio_token_id'],
        max_tokens=c['max_tokens'],audio_mirror=audio_mirror(c),prompt_layout=c.get('prompt_layout'))
    preparer=AuditedPreparer(preparer,evidence)
    identity=dict(schema='jev-reference-dataset-v1',policy_id=bundle['policy_id'],base_id=c['base_id'],
        contract=c['source_contract'],runtime_identity=runtime,release_sha256=c['release_sha256'],
        candidate_ids=[v['id'] for v in candidates])
    root=Path(c['output']);check_paths({'reference_build':root});root.mkdir(parents=True,exist_ok=False)
    pool=None;started=time.monotonic()
    try:
        commit_json(root/'input-evidence.json',dict(input_proof,
            input_proof_sha256=c['input_proof_sha256'],input_receipts_sha256=c['input_receipts_sha256']))
        pool=connect_pool(c['macro_client'],bundle=bundle)
        engine=pool
        if 'resume' in c:
            if resume_pool is None:
                raise ValueError('pinned reference resume entrypoint required')
            engine=resume_pool(pool,c,file_hash(commit))
        result=build(root/'dataset',assets=assets,candidates=candidates,requirements=task['thinking']['requirements'],
            preparer=preparer,engine=engine,identity=identity,macro_queries=c['macro_client']['max_queries'],
            audio_token_id=c['audio_token_id'],producer=dict(provenance=provenance,
                stage1_commit_sha256=file_hash(commit),endpoint=c['stage1_endpoint']),
            **({'prefetch':prefetch} if prefetch is not None else {}))
        if engine is not pool:
            result['resume']=engine.finish()
        pool.close();pool=None
        commit_json(root/'result.json',dict(result,total_seconds=time.monotonic()-started))
        commit_json(root/'terminal.json',dict(exit_code=0))
        return dict(status='reference_built_independent_audit_required',recordings=result['recordings'],
                    distinct_inputs=result['distinct_inputs'],manifest_sha256=result['manifest_sha256'])
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise
    finally:
        if pool is not None:pool.close()


def audio_mirror(c):
    from multimodal_jev.local_audio_mirror import ReadOnlyAudioMirror
    if not c.get('audio_mirror_root'):
        raise ValueError('explicit published audio mirror required for reference build')
    return ReadOnlyAudioMirror(c['audio_mirror_root'])


def main(*,resume_pool=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance,resume_pool=resume_pool),indent=2))


if __name__=='__main__':
    main()
