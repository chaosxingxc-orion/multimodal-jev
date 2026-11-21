"""Rebuild label-free prompt tokens with authenticated reusable acoustic features.

Every old prompt is reproduced through the real processor before conversion.
A deterministic length-stratified panel additionally checks fresh waveform
processing, including both context extremes. No model inference or labels.
"""
import argparse
import json
from pathlib import Path
import threading
import time
from types import MethodType

from multimodal_jev.baseline import atomic_json, file_hash


def run(c, out):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_decision_contract import encode_context_request, PREFIX_LAYOUT
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from multimodal_jev.local_audio_mirror import LocalAudioMirror
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.slurp_portable_release import load
    from multimodal_jev.span_request_identity import tensor_identity
    import numpy as np
    if list(Path('/dev').glob('davinci[0-9]*')): raise ValueError('CPU only')
    require_training_storage(out); torch.set_num_threads(1)
    for name,h in c['pins'].items():
        if file_hash(name)!=h: raise ValueError('dependency changed: '+name)
    rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
    if len(rows)!=40513: raise ValueError('full clean TRAIN required')
    old_manifest=json.loads((Path(c['old_cache']).parent/'MANIFEST.json').read_text())
    if old_manifest['release_sha256']!=c['release_sha256']: raise ValueError('release mismatch')
    receipts=[json.loads(x) for x in Path(c['old_inputs']).read_text().splitlines()]
    if file_hash(c['old_inputs'])!=old_manifest['inputs_sha256']: raise ValueError('input receipts changed')
    if [r['id'] for r in receipts]!=[r['id'] for r in rows]: raise ValueError('population mismatch')
    ordered=sorted(range(len(rows)),key=lambda i:receipts[i]['tensors']['input_ids']['shape'][-1])
    panel={ordered[round(i*(len(rows)-1)/31)] for i in range(32)}
    candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    factory=lambda:Qwen3ASRProcessor.from_pretrained(Path(c['source_policy'])/'processor',local_files_only=True)
    sha=processor_cache_identity(c['source_policy'])
    if sha!=old_manifest['processor_sha256']: raise ValueError('processor mismatch')
    old=ReadOnlyEncodedAudioCache(c['old_cache']);new=EncodedAudioCache(out/'cache')
    kwargs=dict(processor_sha256=sha,decision_token_id=151706,audio_token_id=c['audio_token_id'],max_tokens=4096)
    old_preparer=CachedContextPreparer(factory,old,**kwargs)
    new_preparer=CachedContextPreparer(factory,new,prompt_layout=PREFIX_LAYOUT,**kwargs)
    mirror=LocalAudioMirror(Path(c['audio_mirror_root'])/'audio',read_only=True)
    local=threading.local()
    templates={};template_lock=threading.Lock();template_processor=factory()
    def one(job):
        i,row=job; prior=receipts[i]
        request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        identity=old_preparer.identity(request,row['audio_sha256']);key=key_for(identity)
        if key!=prior['cache_key'] or file_hash(old.root/(key+'.pt'))!=prior['cache_sha256']: raise ValueError('old cache changed')
        inputs,pos,receipt=old.read(identity)
        features={k:v for k,v in inputs.items() if k in ('input_features','input_features_mask')}
        token_count=int((inputs['input_ids']==c['audio_token_id']).sum())
        with template_lock:
            if token_count not in templates:
                processor=template_processor
                original=processor._process_audio
                def cached_audio(self,audio,**kw):
                    data=dict(features)
                    data['num_audio_tokens']=self._get_audio_token_length(data['input_features_mask'].sum(-1),kw.get('n_window',50))
                    if int(data['num_audio_tokens'][0])!=token_count:raise ValueError('audio token length differs')
                    return data,[self.replace_audio_token(data,0)]
                processor._process_audio=MethodType(cached_audio,processor)
                dummy=np.zeros(16000,dtype=np.float32)
                try:
                    before,bp=encode_context_request(processor,request,dummy,decision_token_id=151706,audio_token_id=c['audio_token_id'],max_tokens=4096)
                    after,ap=encode_context_request(processor,request,dummy,decision_token_id=151706,audio_token_id=c['audio_token_id'],max_tokens=4096,prompt_layout=PREFIX_LAYOUT)
                finally:processor._process_audio=original
                templates[token_count]=({k:v for k,v in before.items() if k not in features},bp,{k:v for k,v in after.items() if k not in features},ap)
            before,bp,text_after,ap=templates[token_count]
        if any(not torch.equal(v,inputs[k]) for k,v in before.items()) or not torch.equal(bp,pos):raise ValueError('old processor reproduction failed')
        after=dict(text_after,**features)
        if i in panel:
            if not hasattr(local,'processor'):local.processor=factory()
            processor=local.processor
            wave,new_receipt=load_audio(mirror.resolve(row['audio_sha256']),return_receipt=True)
            fresh,fp=encode_context_request(processor,request,wave,decision_token_id=151706,audio_token_id=c['audio_token_id'],max_tokens=4096,prompt_layout=PREFIX_LAYOUT)
            fresh={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in fresh.items()}
            if new_receipt!=receipt or set(fresh)!=set(after) or any(not torch.equal(v,after[k]) for k,v in fresh.items()) or not torch.equal(fp,ap):raise ValueError('fresh audio parity failed')
        nid=new_preparer.identity(request,row['audio_sha256']);path=new.write(nid,after,ap,receipt)
        return dict(id=row['id'],cache_key=key_for(nid),cache_sha256=file_hash(path),request_sha256=prior['request_sha256'],audio_sha256=row['audio_sha256'],cache_hit=False,fresh_audio_checked=i in panel,tensors={k:tensor_identity(v) for k,v in dict(after,positions=ap).items()})
    start=time.monotonic();minimum=4096;maximum=0;checked=0
    with (out/'inputs.jsonl').open('x') as f, OrderedPrefetch(one,enumerate(rows),workers=c['workers'],ahead=c['workers']*2) as work:
        for n,r in enumerate(work,1):
            f.write(json.dumps(r,separators=(',',':'))+'\n');length=r['tensors']['input_ids']['shape'][-1]
            minimum=min(minimum,length);maximum=max(maximum,length);checked+=r['fresh_audio_checked']
            if n%256==0:
                f.flush();atomic_json(out/'progress.json',dict(completed=n,total=len(rows),seconds=time.monotonic()-start))
        f.flush();__import__('os').fsync(f.fileno())
    manifest=dict(schema='context-full-learning-cache-v1',recordings=len(rows),release_sha256=c['release_sha256'],processor_sha256=sha,prompt_layout=PREFIX_LAYOUT,request_version='jev-context-exclusive_event-v1',inputs_sha256=file_hash(out/'inputs.jsonl'),cache_root=str(new.root),seconds=time.monotonic()-start,existing_hits=0,old_manifest_sha256=c['pins'][str(Path(c['old_cache']).parent/'MANIFEST.json')],fresh_audio_checked=checked,min_tokens=minimum,max_tokens=maximum,scope='Full authenticated old processor reproduction; new prefix tokens with unchanged acoustic features; length-stratified fresh audio parity.')
    commit_json(out/'MANIFEST.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:
        result=run(json.loads(Path(a.config).read_text()),out)
        atomic_json(out/'terminal.json',dict(exit_code=0,result=result))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
