"""Full fresh-process cache audit plus processor recomputation at length extremes."""
import hashlib
import inspect
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_decision_contract import encode_context_request
from multimodal_jev.encoded_audio_cache import EncodedAudioCache, binding, key_for
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.preflight_minds_context_inputs import authenticate
from tools.reports.prepare_minds_context_inputs import reconstruct


def validate_features(inputs, samples):
    """Admit at most one hop's endpoint rounding; never 30s clipping.

    Raw mask slicing rounds differently with longest/max-length padding.
    Every whole 10ms hop must remain represented. Padding is not evidence.
    """
    mask=inputs['input_features_mask'];features=inputs['input_features']
    effective=max(samples,8000)
    frames=int(mask.sum())
    if (mask.ndim!=2 or mask.shape[0]!=1 or features.shape!=(1,128,mask.shape[-1])
            or features.dtype!=torch.bfloat16 or not torch.isfinite(features).all()
            or frames not in {effective//160,(effective+159)//160}
            or not torch.equal(mask.bool(),torch.arange(mask.shape[-1])[None,:]<frames)
            or not ((mask==0)|(mask==1)).all()):
        raise ValueError('Feature duration, prefix mask or BF16 dimensions differ')
    return frames


def verify(config, output):
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only verification required')
    torch.set_num_threads(1)
    output=Path(output)
    rows,definitions=authenticate(config)
    manifest=json.loads((output/'MANIFEST.json').read_text())
    if (manifest['schema']!='minds14-context-processor-v1' or manifest['recordings']!=8168
            or manifest['decisions']!=14 or manifest['context_limit']!=4096
            or manifest['input_manifest_sha256']!=config['input_manifest_sha256']
            or manifest['receipts_sha256']!=file_hash(output/'inputs.jsonl')
            or manifest['cache_root']!=str(output/'cache')):
        raise ValueError('Complete input preparation manifest required')
    identity=processor_cache_identity(config['source_policy'])
    if identity!=manifest['processor_sha256']:raise ValueError('Processor assets changed')
    cache=EncodedAudioCache(output/'cache')
    records=[json.loads(line) for line in (output/'inputs.jsonl').read_text().splitlines()]
    longest=max(range(len(records)),key=lambda i:records[i]['audio_samples'])
    shortest=min(range(len(records)),key=lambda i:records[i]['audio_samples'])
    odd=next((i for i,r in enumerate(records) if r['audio_samples']%160),None)
    sentinels={shortest,longest}|({odd} if odd is not None else set())
    from transformers import Qwen3ASRProcessor
    proc=Qwen3ASRProcessor.from_pretrained(Path(config['source_policy'])/'processor',local_files_only=True)
    for name,obj in (('processor',proc),('feature_extractor',proc.feature_extractor)):
        if (output/('actual-'+name+'-source.py')).read_text()!=inspect.getsource(type(obj)):
            raise ValueError('Runtime processor implementation differs')
    feats=proc.feature_extractor
    if (feats.hop_length!=160 or feats.min_length!=8000 or feats.feature_size!=128
            or feats.n_window!=50 or feats.sampling_rate!=16000):
        raise ValueError('Unreviewed feature extractor configuration')
    keys=set();recomputed=[];long_audio=0;maximum=0;max_samples=0
    for index,(source,row) in enumerate(zip(rows,records,strict=True)):
        for name in ('id','audio_sha256','request_sha256','rendered_sha256'):
            if source[name]!=row[name]:raise ValueError('Input order or binding differs')
        request=reconstruct(source,definitions)
        expected=binding(request=request,audio_sha256=source['audio_sha256'],
            processor_sha256=identity,preprocessing=None)
        expected['context_encoding']=dict(decision_token_id=151706,audio_token_id=151676,
            max_tokens=4096,features_dtype='bfloat16',version='context-cache-v1')
        key=key_for(expected)
        if key!=row['cache_key'] or file_hash(cache.root/(key+'.pt'))!=row['cache_sha256']:
            raise ValueError('Cache identity differs')
        loaded=cache.read(expected)
        if loaded is None:raise ValueError('Missing cache')
        inputs,positions,audio_receipt=loaded
        hashes={k:tensor_identity(v) for k,v in dict(inputs,positions=positions).items()}
        if hashes!=row['tensors'] or audio_receipt!=row['audio_receipt']:
            raise ValueError('Cache tensor or audio receipt differs')
        n=row['audio_samples']
        if (audio_receipt['input_samples']!=n or audio_receipt['output_samples']!=n
                or audio_receipt['retained_original_interval']!=[0,n]
                or audio_receipt['input_waveform_sha256']!=audio_receipt['output_waveform_sha256']):
            raise ValueError('Waveform preprocessing changed evidence')
        import soundfile as sf
        if file_hash(source['audio'])!=source['audio_sha256'] or sf.info(source['audio']).frames!=n:
            raise ValueError('Input audio hash or sample count differs')
        frames=validate_features(inputs,n)
        tokens=inputs['input_ids'].shape[-1]
        actual=(inputs['input_ids'][0]==151706).nonzero().flatten()
        audio_tokens=int((inputs['input_ids']==151676).sum())
        expected_tokens=13*(frames//100)+((frames%100+7)//8)
        if (frames!=row['feature_frames'] or audio_tokens!=expected_tokens
                or positions.shape!=(1,14) or not torch.equal(positions[0],actual)
                or positions.tolist()!=row['positions'] or tokens!=row['tokens'] or tokens>4096):
            raise ValueError('Frame/token/readout coverage differs')
        if index in sentinels:
            from multimodal_jev.audio_policy import load_audio
            audio,receipt=load_audio(source['audio'],return_receipt=True)
            fresh,p=encode_context_request(proc,request,audio,decision_token_id=151706,
                audio_token_id=151676,max_tokens=4096)
            fresh={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in fresh.items()}
            if (receipt!=audio_receipt or not torch.equal(p,positions) or fresh.keys()!=inputs.keys()
                    or any(not torch.equal(fresh[k],inputs[k]) for k in inputs)):
                raise ValueError('Fresh actual processor recomputation differs')
            recomputed.append(dict(id=row['id'],samples=n,frames=frames,tokens=tokens))
        keys.add(key+'.pt');long_audio+=n>30*16000;maximum=max(maximum,tokens);max_samples=max(max_samples,n)
    if {p.name for p in cache.root.iterdir()}!=keys:raise ValueError('Unexpected cache population')
    if maximum!=manifest['max_tokens'] or max_samples!=manifest['max_samples']:
        raise ValueError('Maximum input statistics differ')
    return dict(schema='minds14-processor-cache-verification-v1',recordings=len(rows),decisions=14,
        output_manifest_sha256=file_hash(output/'MANIFEST.json'),processor_sha256=identity,
        input_manifest_sha256=config['input_manifest_sha256'],max_tokens=maximum,
        max_audio_seconds=max_samples/16000,recordings_over_30_seconds=long_audio,
        all_cache_tensors_verified=True,all_whole_audio_hops_present=True,
        rounding_scope='Feature-mask endpoints allow floor/ceil at one 10ms hop; original waveform retained.',
        fresh_processor_recomputation=recomputed,verifier_sha256=file_hash(__file__),
        scope='Full cache/hash/duration audit and length-extreme processor recomputation; no model inference.')


if __name__=='__main__':
    import argparse
    from multimodal_jev.local_checkpoint import commit_json
    p=argparse.ArgumentParser()
    for name in ('config','output','receipt'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=verify(json.loads(Path(a.config).read_text()),a.output)
    if Path(a.receipt).exists():raise ValueError('New receipt required')
    commit_json(Path(a.receipt),r);print(json.dumps(r,indent=2))
