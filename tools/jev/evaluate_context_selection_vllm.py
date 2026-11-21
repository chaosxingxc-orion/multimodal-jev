#!/usr/bin/env python3
"""Run one fixed joint policy on the complete clean selection audio population."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def run(c,out):
    import torch
    import torch_npu
    from multimodal_jev.model_provider import resolve_provider, cache_namespace
    provider = resolve_provider(c)
    from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.slurp_portable_release import load
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.span_request_identity import tensor_identity
    require_training_storage(out)
    require_training_storage(c['encoded_audio_cache'])
    if (len(c['physical_devices'])!=1 or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=
            ['davinci'+str(c['physical_devices'][0])]):
        raise ValueError('one isolated NPU required')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('dependency changed: '+path)
    torch.set_num_threads(2);torch.npu.set_device(0)
    bundle=read_execution_bundle(c['snapshot'],expected_policy_id=c['policy_id'],
        expected_contract=c['contract'],expected_base_id=c['base_id'])
    if (bundle['contract']['request_version']!='jev-context-exclusive_event-v1' or
            bundle['contract']['runtime_profile']!=provider.runtime_profile):
        raise ValueError('actual exclusive-event request and runtime contract required')
    if not callable(getattr(provider,'load_vllm',None)):
        raise ValueError('selected model_provider does not implement vLLM serving')
    rows,task,release=load(c['release'],c['release_sha256'],'selection',purpose='evaluate')
    if len(rows)!=9226 or release['counts']['selection']['recordings']!=9226:
        raise ValueError('complete fixed selection population required')
    # No labels, transcript or source intent is given to the preparer or engine.
    assets=[{k:r[k] for k in ('id','audio','audio_sha256')} for r in rows]
    del rows
    candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('full candidate ontology required')
    prepare=CachedContextPreparer(
        lambda:provider.processor(bundle['processor']),
        EncodedAudioCache(c['encoded_audio_cache']),processor_sha256=processor_cache_identity(c['snapshot']),
        decision_token_id=bundle['contract']['token']['id'],audio_token_id=c['audio_token_id'],max_tokens=c.get('max_model_len',4096),
        encode=provider.encode,provider_namespace=cache_namespace(c),prompt_layout=c.get('prompt_layout'))
    def encode(asset):
        request=compile_candidates(asset['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        value=prepare((request,asset['audio_sha256']))
        return asset,request,value
    engine_config={k:c[k] for k in ('model','base_id','snapshot','policy_id','contract','physical_devices')}
    engine_config['adapter'] = c.get('adapter', str(out/'native-adapter'))
    if 'model_provider' in c:
        engine_config['model_provider']=c['model_provider']
    engine_config.update(tensor_parallel_size=1,max_model_len=c.get('max_model_len',4096),max_decisions=60)
    records=[];input_receipts=[];started=time.monotonic()
    from multimodal_jev.policy_identity import checkpoint_identity
    if checkpoint_identity(c['model'])[0] != c['base_id']:
        raise ValueError('serving model differs from evaluation base')
    engine=provider.load_vllm(engine_config,bundle)
    try:
        with OrderedPrefetch(encode,assets,workers=1,ahead=2) as prepared:
            for index,(asset,request,value) in enumerate(prepared):
                begin=time.monotonic()
                row=engine.score(value.inputs,value.positions,audio_token_id=c['audio_token_id'])
                if index == 0:
                    atomic_json(out/'runtime.json',engine.runtime)
                expected=dict(policy_id=c['policy_id'],positions=value.positions[0].tolist(),
                    inputs={k:tensor_identity(v) for k,v in sorted(value.inputs.items())})
                if row['request_identity']!=expected:raise ValueError('engine consumed different input tensors')
                row.update(id=asset['id'],seconds=time.monotonic()-begin)
                receipt=dict(id=asset['id'],audio_sha256=asset['audio_sha256'],
                    request_sha256=content_hash(request.model_input()),engine_request_id=row['request_id'],
                    audio_receipt=value.audio_receipt,inputs=expected['inputs'],positions=expected['positions'])
                # Durable records survive interruption; incomplete runs never
                # publish a complete population result or selection metric.
                commit_json(out/f'record-{index:05d}.json',dict(prediction=row,input_receipt=receipt))
                records.append(row);input_receipts.append(receipt)
                if (index+1)%64==0:
                    atomic_json(out/'progress.json',dict(completed=index+1,total=9226,
                        seconds=time.monotonic()-started,policy_id=c['policy_id']))
    finally:
        engine.close()
    if len(records)!=9226 or len({r['id'] for r in records})!=9226:
        raise ValueError('incomplete selection run')
    commit_json(out/'predictions.json',dict(records=records))
    commit_json(out/'input-receipts.json',input_receipts)
    from evaluate_context_event_records import evaluate
    result=evaluate(c['release'],c['release_sha256'],out/'predictions.json',
        file_hash(out/'predictions.json'),c['policy_id'],'selection')
    result.update(input_receipts_sha256=file_hash(out/'input-receipts.json'),seconds=time.monotonic()-started)
    commit_json(out/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    run(json.loads(a.config.read_text()),a.output)
