"""SLURP selection through already admitted isolated macro inference services.

This is a new entry point; it does not alter frozen training/evaluation packs.
Service image, code, device and actual backend parity admission are obligations
of the launcher and independent collector, not proved by client connection.
"""
import argparse
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load
from multimodal_jev.slurp_training import content_hash
from tools.jev.evaluate_context_event_records import evaluate as score
from tools.reports.context_macro_client import connect_pool
from tools.reports.context_macro_evaluation import evaluate
from tools.reports.context_selection_audited_cache import load_evidence, encoder


def inputs(config):
    rows,task,release=load(config['release'],config['release_sha256'],'selection',purpose='evaluate')
    if len(rows)!=9226 or release['counts']['selection']['recordings']!=9226:
        raise ValueError('complete SLURP selection population required')
    candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('fixed SLURP 60 candidates required')
    requests={};population=[]
    for row in rows:
        request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        requests[row['id']]=request
        population.append(dict(id=row['id'],audio_sha256=row['audio_sha256'],
                               request_sha256=content_hash(request.model_input())))
    # Native targets remain outside the inference population and request map.
    return population,requests,[c['id'] for c in candidates]


def run(config,provenance):
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    out=Path(config['output']);check_paths({'evaluation':out,'cache':Path(config['encoded_audio_cache'])})
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned dependency changed: '+path)
    bundle=read_execution_bundle(config['snapshot'],expected_policy_id=config['policy_id'],
        expected_contract=config['contract'],expected_base_id=config['base_id'])
    if (bundle['contract']['request_version']!='jev-context-exclusive_event-v1'
            or bundle['contract']['runtime_profile']!=PROFILE
            or bundle['contract']['token']['registration_status']!='admitted'):
        raise ValueError('admitted exclusive-event policy required')
    population,requests,candidates=inputs(config)
    import json
    release_manifest=json.loads((Path(config['release'])/'MANIFEST.json').read_text())
    processor_id=processor_cache_identity(config['snapshot'])
    receipts,evidence=load_evidence(config,population,release_manifest=release_manifest,processor_sha256=processor_id)
    encode=encoder(config['encoded_audio_cache'],requests,candidates,receipts,
        processor_sha256=processor_id,decision_token_id=bundle['contract']['token']['id'],
        audio_token_id=config['audio_token_id'])
    out.mkdir(parents=True,exist_ok=False);pool=None
    try:
        commit_json(out/'provenance.json',provenance)
        commit_json(out/'input-evidence.json',evidence)
        pool=connect_pool(config['macro_client'],bundle=bundle)
        inference=evaluate(pool,population,encode,candidate_ids=candidates,policy_id=config['policy_id'],
            runtime_identity=config['macro_client']['runtime_identity'],audio_token_id=config['audio_token_id'],
            macro_queries=config['macro_queries'],output=out/'inference')
        pool.close();pool=None
        result=score(config['release'],config['release_sha256'],out/'inference/predictions.json',
            inference['predictions_sha256'],config['policy_id'],'selection')
        result.update(input_receipts_sha256=inference['input_receipts_sha256'],
                      inference_result_sha256=file_hash(out/'inference/result.json'))
        commit_json(out/'result.json',result);commit_json(out/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(out/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if pool is not None:pool.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    run(config,provenance)
