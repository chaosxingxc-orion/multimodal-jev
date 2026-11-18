"""Bind existing independent selection cache evidence to portable SLURP inputs.

Cache misses are errors: evaluation never rebuilds inputs or invokes a processor.
The evidence binding alone does not verify that remote cache files still exist;
each file and its actual tensor values are checked when consumed.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import CachedContextPreparer
from multimodal_jev.encoded_audio_cache import EncodedAudioCache,key_for
from multimodal_jev.span_request_identity import tensor_identity
from tools.jev.context_cache_receipts import parse_stream


def bind_evidence(proof,stream,population,*,release_sha256,release_manifest,processor_sha256):
    state=proof['state'];manifest=proof['manifest']
    if (state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or manifest['schema']!='context-selection-cache-v1' or manifest['role']!='selection'
            or manifest['processor_sha256']!=processor_sha256
            or manifest['request_version']!='jev-context-exclusive_event-v1'
            or manifest['recordings']!=9226 or len(population)!=9226):
        raise ValueError('complete successful selection cache with matching processor required')
    source=manifest['release_sha256']
    if source!=release_sha256:
        migration=release_manifest.get('migration',{})
        unchanged=migration.get('unchanged_payload_sha256',{})
        if (release_manifest['schema']!='slurp-official-clean-intent-v2'
                or migration.get('source_manifest_sha256')!=source
                or any(name not in unchanged or unchanged[name]!=release_manifest['files'].get(name)
                       for name in ('selection.jsonl','selection.audio.jsonl','tasks.json'))):
            raise ValueError('explicit byte-preserving release migration required')
    records,diagnostics=parse_stream(stream)
    if records[-1]['value']!=manifest or records[0]['value'].get('recordings')!=9226:
        raise ValueError('audit stream manifest or terminal differs')
    values=[r['value'] for r in records[1:-1]]
    if len({r['id'] for r in values})!=9226 or len({r['id'] for r in population})!=9226:
        raise ValueError('complete unique input identities required')
    for asset,receipt in zip(population,values,strict=True):
        if any(receipt[k]!=v for k,v in asset.items()):raise ValueError('cache request/audio/order differs')
        tensors=receipt['tensors']
        if (set(tensors)!={'input_ids','attention_mask','input_features','input_features_mask','positions'}
                or tensors['positions']['shape']!=[1,60] or tensors['positions']['dtype']!='torch.int64'
                or tensors['input_features']['dtype']!='torch.bfloat16'):
            raise ValueError('complete independent BF16 input tensor identities required')
    return values,dict(recordings=len(values),processor_sha256=processor_sha256,
        source_release_sha256=source,release_sha256=release_sha256,driver_diagnostics=diagnostics,
        scope='Existing independent cache evidence bound to full current requests; remote cache existence and actual bytes checked on consumption')


def load_evidence(config,population,*,release_manifest,processor_sha256):
    def bound(entry):
        if set(entry)!={'path','sha256'} or file_hash(entry['path'])!=entry['sha256']:
            raise ValueError('cache evidence artifact changed')
        return Path(entry['path']).read_text()
    proof=json.loads(bound(config['cache_audit']))
    stream=bound(config['cache_receipts'])
    if config['cache_receipts']['sha256']!=proof['independent_receipts_sha256']:
        raise ValueError('input stream is not bound by independent audit')
    return bind_evidence(proof,stream,population,release_sha256=config['release_sha256'],
        release_manifest=release_manifest,processor_sha256=processor_sha256)


def encoder(cache_root,requests,candidate_ids,receipts,*,processor_sha256,decision_token_id,audio_token_id):
    root=Path(cache_root)
    if not root.is_dir():raise ValueError('existing audited cache directory required')
    cache=EncodedAudioCache(root)
    def no_processor():raise ValueError('evaluation cache rebuild forbidden')
    prepare=CachedContextPreparer(no_processor,cache,processor_sha256=processor_sha256,
        decision_token_id=decision_token_id,audio_token_id=audio_token_id,max_tokens=4096)
    by_id={r['id']:r for r in receipts}
    if len(by_id)!=len(receipts) or set(by_id)!=set(requests):raise ValueError('complete cache/request identities required')
    def encode(asset):
        import torch
        request=requests[asset['id']];prior=by_id[asset['id']]
        if any(prior[k]!=v for k,v in asset.items()):raise ValueError('independent asset binding differs')
        identity=prepare.identity(request,asset['audio_sha256']);key=key_for(identity)
        if key!=prior['cache_key'] or file_hash(root/(key+'.pt'))!=prior['cache_sha256']:
            raise ValueError('cache identity or bytes changed')
        value=cache.read(identity)
        if value is None:raise ValueError('audited cache missing')
        inputs,positions,audio_receipt=value
        actual={k:tensor_identity(v) for k,v in dict(inputs,positions=positions).items()}
        if actual!=prior['tensors']:raise ValueError('cache tensor differs from independent audit')
        markers=(inputs['input_ids'][0]==decision_token_id).nonzero().flatten()
        if (positions.shape!=(1,len(candidate_ids)) or not torch.equal(markers,positions[0])
                or inputs['input_ids'].shape[1]>4096):
            raise ValueError('full ordered candidate marker positions required')
        return dict(inputs=inputs,positions=positions),dict(asset,candidate_ids=candidate_ids,
            inputs={k:v for k,v in actual.items() if k!='positions'},positions=positions[0].tolist(),
            audio_receipt=audio_receipt)
    return encode


def run(config,provenance):
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import check_paths
    from tools.reports.evaluate_context_selection_macro import inputs
    population,_,candidates=inputs(config)
    manifest=json.loads((Path(config['release'])/'MANIFEST.json').read_text())
    values,report=load_evidence(config,population,release_manifest=manifest,
        processor_sha256=processor_cache_identity(config['snapshot']))
    root=Path(config['output']);check_paths({'evidence':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'population.json',population)
    commit_json(root/'candidate-ids.json',candidates)
    commit_json(root/'cache-receipts.json',values)
    report.update(provenance=provenance,cache_audit=config['cache_audit'],cache_receipts=config['cache_receipts'],
        artifacts={p.name:file_hash(p) for p in root.iterdir() if p.is_file()})
    commit_json(root/'binding.json',report)
    return report


if __name__=='__main__':
    from multimodal_jev.local_config import load_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    print(json.dumps(run(config,provenance),indent=2))
