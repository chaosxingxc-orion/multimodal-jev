"""Compile a complete scheduled audio batch once for actor and rollout.

Caller authenticates the full learning release and immutable epoch schedule.
This joins their metadata and checks audio bytes; it grants no runtime admission.
Targets remain in actor items, never in renderer inputs or policy identities.
"""
import json
import math
import torch
from .baseline import file_hash
from .audio_policy import load_audio
from .slurp_official_training import example
from .span_contract import encode_span_request, span_labels
from .span_request_identity import bind_span_request


def compile_batch(population, task, slots, processor, *, policy_id, global_batch, context_limit=4096):
    if (type(global_batch) is not int or global_batch not in (128,256)
            or len(slots) != global_batch or context_limit != 4096
            or not isinstance(policy_id,str) or not policy_id):
        raise ValueError('complete fixed simple batch and policy required')
    real_ids=set(); seen_padding=False
    for slot in slots:
        row=population[slot['id']]
        if (row['id']!=slot['id'] or row['role']!='learning' or row['known_pending_source']
                or row['source_group']!=slot['source_group']
                or (slot['view'],slot['variant'])!=('direct','native')
                or type(slot['padding']) is not bool
                or type(slot['weight']) not in (int,float) or not math.isfinite(slot['weight'])
                or (slot['padding'] and slot['weight']!=0)
                or (not slot['padding'] and slot['weight']<=0)):
            raise ValueError('admitted native learning slot and explicit padding required')
        if slot['padding']:
            seen_padding=True
        else:
            if seen_padding or slot['id'] in real_ids:
                raise ValueError('distinct real slots must precede terminal padding')
            real_ids.add(slot['id'])
    if not real_ids:raise ValueError('batch must include real learning requests')
    # Bounded to this batch. A subsequent batch reauthenticates audio; no mutable
    # cross-version cache or unbounded retention of CPU feature tensors.
    compiled={}; entries=[]; renderer=[]; metadata=[]
    for slot in slots:
        uid=slot['id'];row=population[uid]
        if uid not in compiled:
            if file_hash(row['audio'])!=row['audio_sha256']:
                raise ValueError('scheduled waveform changed')
            request,target,_=example(row,task,'direct','native')
            audio,receipt=load_audio(row['audio'],return_receipt=True)
            inputs,positions=encode_span_request(processor,request,audio)
            mapping=span_labels(processor.tokenizer,request.questions)
            candidates=[m['token_ids'] for m in mapping]
            if (len(candidates)!=1 or len(candidates[0])!=60
                    or inputs['input_ids'].shape[-1]>context_limit
                    or any(v.device.type!='cpu' for v in inputs.values())):
                raise ValueError('single CPU native Final60 within untruncated context required')
            rid,identity=bind_span_request(request,receipt,inputs,positions,candidates,policy_id)
            compiled[uid]=(inputs,positions,candidates,target,rid,identity)
            # Row consumed by ResidentSpanRollout; all other model metadata is
            # identity-bound below. No gold, source labels or sample weights.
            renderer.append((dict(id=uid),inputs,positions,candidates,rid,identity,processor))
        inputs,positions,candidates,target,rid,identity=compiled[uid]
        entries.append(dict(identity=identity,item=dict(policy_id=policy_id,inputs=inputs,
            positions=positions,candidates=candidates,targets=torch.tensor([target['targets']],dtype=torch.long),
            supervised=torch.tensor([target['supervised']],dtype=torch.bool),
            span_weights=torch.tensor([target['span_weights']],dtype=torch.float32),
            request_weight=slot['weight'],real_request=not slot['padding'])))
        metadata.append(dict(recording_id=uid,request_id=rid,weight=slot['weight'],padding=slot['padding']))
    return entries,renderer,dict(global_rows=len(entries),global_real_requests=len(real_ids),
        unique_audio_inputs=len(renderer),records=metadata,policy_id=policy_id,
        scope='Full scheduled batch compiled once; caller authenticates population/schedule and actual backend execution.')


def renderer_for_policy(renderer, policy_id):
    """Reuse exact CPU inputs for a different immutable adapter; no reprocessing."""
    import hashlib
    if not isinstance(policy_id,str) or not policy_id:raise ValueError('explicit policy required')
    result=[]
    for row,inputs,positions,candidates,rid,identity,processor in renderer:
        original=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
        if hashlib.sha256(original.encode()).hexdigest()!=rid:raise ValueError('renderer identity changed')
        wire=json.loads(original);wire['policy_id']=policy_id
        digest=hashlib.sha256(json.dumps(wire,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        result.append((row,inputs,positions,candidates,digest,wire,processor))
    return result
