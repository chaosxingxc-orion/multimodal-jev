"""Recreate authenticated label-free native inputs for HF or resident vLLM.

Shared by the independent backend gate and in-training full-selection scoring.
No labels or optimizer state enter the renderer.
"""
import json
from .baseline import file_hash


def prepared(config,artifact, *, processor=None):
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import SpanRequest,SpanQuestion,encode_span_request,span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.span_request_identity import tensor_identity,bind_span_request
    from multimodal_jev.policy_identity import checkpoint_identity
    if processor is None:
        from transformers import Qwen3ASRProcessor
        processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    base_id=checkpoint_identity(config['model'])[0];adapter_id=checkpoint_identity(config['adapter'])[0]
    if base_id!=config['base_id'] or adapter_id!=config['adapter_id']:raise ValueError('execution policy changed')
    for row in artifact['records']:
        raw=row['request'];request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
            for q in raw['questions']),raw['final_id'],tuple(raw['requirements']))
        if file_hash(request.audio)!=row['audio_sha256']:raise ValueError('audio bytes changed')
        audio,receipt=load_audio(request.audio,return_receipt=True)
        batch,positions=encode_span_request(processor,request,audio)
        mapping=span_labels(processor.tokenizer,request.questions)
        if (receipt!=row['audio_processing'] or positions.tolist()!=row['positions']
                or json.loads(json.dumps(mapping))!=row['candidate_mapping']):raise ValueError('CPU input identity differs')
        for name in ('input_ids','input_features','input_features_mask'):
            if tensor_identity(batch[name])!=row[name]:raise ValueError('encoded input changed: '+name)
        if batch['input_ids'].shape[-1]>config['max_model_len']:raise ValueError('full input exceeds engine context')
        candidates=[m['token_ids'] for m in mapping]
        rid,identity=bind_span_request(request,receipt,batch,positions,candidates,base_id+':'+adapter_id)
        yield row,batch,positions,candidates,rid,identity,processor

