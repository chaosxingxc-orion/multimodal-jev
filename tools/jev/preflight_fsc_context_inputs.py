#!/usr/bin/env python3
"""Encode the full frozen FSC input roster on CPU, without target labels."""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(c,out):
 import torch
 from transformers import Qwen3ASRProcessor
 from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
 from multimodal_jev.context_decision_contract import ApplicabilityQuestion,ContextDecisionRequest
 from multimodal_jev.encoded_audio_cache import EncodedAudioCache,key_for
 from multimodal_jev.local_runtime_storage import require_training_storage
 from multimodal_jev.local_checkpoint import commit_json
 from multimodal_jev.ordered_prefetch import OrderedPrefetch
 from multimodal_jev.slurp_training import content_hash
 from multimodal_jev.span_request_identity import tensor_identity
 if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only input preflight')
 require_training_storage(out);torch.set_num_threads(1)
 for p,h in c['pins'].items():
  if file_hash(p)!=h:raise ValueError('dependency changed: '+p)
 source=Path(c['inputs']);manifest=json.loads((source/'MANIFEST.json').read_text())
 if file_hash(source/'requests.jsonl')!=manifest['files']['requests.jsonl']:raise ValueError('input roster changed')
 rows=list(map(json.loads,(source/'requests.jsonl').read_text().splitlines()))
 if len(rows)!=manifest['recordings'] or len(rows)!=3793 or len({r['id'] for r in rows})!=3793:
  raise ValueError('full official input population required')
 def factory():
  proc=Qwen3ASRProcessor.from_pretrained(Path(c['source_policy'])/'processor',local_files_only=True)
  vocab=proc.tokenizer.get_vocab();proc.tokenizer.get_vocab=lambda:dict(vocab)
  return proc
 cache=EncodedAudioCache(out/'cache');identity=processor_cache_identity(c['source_policy'])
 prepare=CachedContextPreparer(factory,cache,processor_sha256=identity,
  decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
 def one(row):
  raw=row['request']
  if set(raw)!={'audio','questions','requirements','history','version'}:raise ValueError('label-free request required')
  request=ContextDecisionRequest(raw['audio'],tuple(ApplicabilityQuestion(**q) for q in raw['questions']),
   tuple(raw['requirements']),tuple(raw['history']),version=raw['version'])
  if request.version!='jev-context-exclusive_event-v1' or len(request.questions)!=31:
   raise ValueError('31 native decision requests required')
  if content_hash(request.model_input())!=row['request_sha256']:raise ValueError('request changed')
  value=prepare((request,row['audio_sha256']))
  key=key_for(prepare.identity(request,row['audio_sha256']))
  return dict(id=row['id'],audio_sha256=row['audio_sha256'],request_sha256=row['request_sha256'],
   cache_key=key,cache_sha256=file_hash(cache.root/(key+'.pt')),
   tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()},
   positions=value.positions.tolist(),tokens=value.inputs['input_ids'].shape[-1])
 started=time.monotonic();maximum=0
 with (out/'inputs.jsonl').open('x') as stream:
  with OrderedPrefetch(one,rows,workers=2,ahead=4) as prepared:
   for index,row in enumerate(prepared):
    stream.write(json.dumps(row,separators=(',',':'))+'\n');maximum=max(maximum,row['tokens'])
    if (index+1)%256==0:
     stream.flush();atomic_json(out/'progress.json',dict(completed=index+1,total=len(rows),max_tokens=maximum))
  stream.flush()
  import os
  os.fsync(stream.fileno())
 report=dict(schema='fsc-context-processor-preflight-v1',recordings=len(rows),decisions=31,
  max_tokens=maximum,context_limit=4096,processor_sha256=identity,cache_root=str(cache.root),
  inputs_sha256=file_hash(out/'inputs.jsonl'),request_manifest_sha256=file_hash(source/'MANIFEST.json'),
  seconds=time.monotonic()-started,scope='Full audio hash/decode/processor and marker alignment; no model forward, predictions or transfer gain')
 commit_json(out/'MANIFEST.json',report);return report


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False);run(json.loads(a.config.read_text()),a.output)
