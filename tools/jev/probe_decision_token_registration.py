#!/usr/bin/env python3
"""Isolated tokenizer experiment: reserved IDs, roundtrip, post-audio readouts."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def run(config, output):
    import torch
    from transformers import AddedToken,Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU probe only')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned file changed: '+p)
    torch.set_num_threads(2)
    root=Path(config['model']);model_config=json.loads((root/'config.json').read_text())
    reserved={}
    def walk(value,prefix=''):
        if isinstance(value,dict):
            for k,v in value.items():
                if k.endswith(('_token_id','_token_ids')):
                    for i in v if isinstance(v,list) else [v]:
                        if type(i) is int and i>=0:reserved.setdefault(i,[]).append(prefix+k)
                if isinstance(v,dict):walk(v,prefix+k+'.')
    walk(model_config)
    token='<|jev_decision|>'
    naive=Qwen3ASRProcessor.from_pretrained(root,local_files_only=True)
    before_vocab=naive.tokenizer.get_vocab()
    if token in before_vocab:raise ValueError('requires unmodified source tokenizer')
    naive.tokenizer.add_special_tokens({'extra_special_tokens':[AddedToken(token,special=True,normalized=False)]},replace_extra_special_tokens=False)
    naive_id=naive.tokenizer.convert_tokens_to_ids(token)
    collision=reserved.get(naive_id,[])
    processor=Qwen3ASRProcessor.from_pretrained(root,local_files_only=True)
    before_special=set(processor.tokenizer.all_special_tokens)
    # Test-only placeholder occupies metadata-reserved gaps; no original asset
    # or embedding is changed. Production adoption needs timestamp semantics review.
    guard=[]
    next_id=max(before_vocab.values())+1
    while next_id in reserved:
        name=f'<|jev_probe_reserved_id_{next_id}|>'
        processor.tokenizer.add_special_tokens({'extra_special_tokens':[AddedToken(name,special=True,normalized=False)]},replace_extra_special_tokens=False)
        if processor.tokenizer.convert_tokens_to_ids(name)!=next_id:raise ValueError('unexpected reservation allocation')
        guard.append(dict(token=name,id=next_id,reserved_by=reserved[next_id]))
        next_id+=1
    processor.tokenizer.add_special_tokens({'extra_special_tokens':[AddedToken(token,special=True,normalized=False)]},replace_extra_special_tokens=False)
    decision_id=processor.tokenizer.convert_tokens_to_ids(token)
    if decision_id in reserved or decision_id!=next_id:raise ValueError('unsafe decision ID')
    after_vocab=processor.tokenizer.get_vocab()
    if any(after_vocab.get(k)!=v for k,v in before_vocab.items()):raise ValueError('existing token mapping changed')
    if not before_special.issubset(set(processor.tokenizer.all_special_tokens)):raise ValueError('existing special-token status lost')
    if processor.tokenizer.encode(token,add_special_tokens=False)!=[decision_id]:raise ValueError('not a single token')
    save=output/'processor';processor.save_pretrained(save)
    reloaded=Qwen3ASRProcessor.from_pretrained(save,local_files_only=True)
    if reloaded.tokenizer.get_vocab()!=processor.tokenizer.get_vocab():raise ValueError('tokenizer mapping changed on reload')
    if set(reloaded.tokenizer.all_special_tokens)!=set(processor.tokenizer.all_special_tokens):raise ValueError('special-token status changed on reload')
    asset=config['audio'][0]
    if file_hash(asset['audio'])!=asset['sha256']:raise ValueError('audio changed')
    audio,receipt=load_audio(asset['audio'],return_receipt=True)
    text='Question 1: Does the audio express a request? '+token+'\nQuestion 2: Is evidence sufficient under the supplied definition? '+token
    system='English\nReturn numeric decisions at the supplied special tokens. Do not generate text.'
    messages=[{'role':'system','content':[{'type':'text','text':system}]},
              {'role':'user','content':[{'type':'audio','audio':audio}]},
              {'role':'assistant','content':text}]
    compiled=[]
    for current in (processor,reloaded):
        rendered=current.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        if text not in rendered or system not in rendered:raise ValueError('template lost definitions')
        batch=current(text=[rendered],audio=[audio],sampling_rate=16000,return_tensors='pt')
        ids=batch['input_ids'][0].tolist()
        positions=[i for i,t in enumerate(ids) if t==decision_id]
        audio_positions=[i for i,t in enumerate(ids) if t==model_config['audio_token_id']]
        if len(positions)!=2 or not audio_positions or max(audio_positions)>=positions[0]:raise ValueError('special readout not after audio')
        compiled.append(dict(positions=positions,readout='marker_token_itself',tokens=len(ids),
            identities={k:tensor_identity(v) for k,v in batch.items()}))
    if compiled[0]!=compiled[1]:raise ValueError('processor roundtrip changed actual tensors')
    return dict(schema='decision-token-registration-cpu-v3',naive=dict(id=naive_id,collision=collision),
        candidate=dict(id=decision_id,token=token,reservation_placeholders=guard,
            within_existing_embedding=decision_id<model_config['text_config']['vocab_size'],
            existing_token_ids_preserved=True,existing_special_token_status_preserved=True,
            single_token=True,processor_roundtrip_exact=True),
        audio_sha256=asset['sha256'],audio_processing=receipt,compiled=compiled[0],
        saved_files={str(p.relative_to(save)):file_hash(p) for p in sorted(save.rglob('*')) if p.is_file()},
        training_dispatch_allowed=False,
        scope='Test-only tokenizer reservation and real audio processor roundtrip. No new embedding initialization, model forward, trained head or NPU/vLLM acceptance. Questions are structural fixtures without supervision; source tokenizer unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
