#!/usr/bin/env python3
"""Native pooling proof on the identical real audio Span prompt as HF smoke."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.vllm_spans import register_span_model

# Spawned engine workers re-import this entrypoint. Register the distinct
# architecture there too; overriding an existing name only in the parent fails.
register_span_model()


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import SpanRequest,SpanQuestion,encode_span_request,span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.vllm_audio_features import enable_feature_passthrough
    from multimodal_jev.vllm_spans import register_span_model
    from vllm import LLM,PoolingParams
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('physical NPU 4 isolation required')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('input changed: '+path)
    reference=json.loads(Path(config['reference']).read_text())
    adapter=config.get('adapter')
    lora_request=None
    lora_options={}
    if adapter:
        from vllm.lora.request import LoRARequest
        adapter_config=json.loads((Path(adapter)/'adapter_config.json').read_text())
        targets=adapter_config['target_modules']
        if not isinstance(targets,list) or not targets or any(
                not name.startswith('model.language_model.layers.') or
                name.rsplit('.',1)[-1] not in {'q_proj','v_proj'} for name in targets):
            raise ValueError('only explicitly named language q/v adapters admitted')
        packed_targets=sorted({name.replace('model.language_model.','language_model.model.',1)
                               .rsplit('.',1)[0]+'.qkv_proj' for name in targets})
        lora_request=LoRARequest('jev-span-pinned-v1',1,adapter)
        lora_dtype = config.get('lora_execution_dtype','bfloat16')
        if lora_dtype not in {'bfloat16','float32'}:
            raise ValueError('unsupported LoRA dtype')
        from safetensors import safe_open
        with safe_open(str(Path(adapter)/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
            if any(saved.get_tensor(k).dtype != getattr(torch,lora_dtype) for k in saved.keys()):
                raise ValueError('adapter storage must match declared execution dtype; no silent conversion')
        lora_options=dict(enable_lora=True,max_loras=1,max_lora_rank=8,lora_dtype=getattr(torch,lora_dtype),
                          lora_target_modules=packed_targets)
    raw=reference['input_request']
    request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema'])) for q in raw['questions']),raw['final_id'],tuple(raw['requirements']))
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    audio,audio_receipt=load_audio(request.audio,config.get('audio_preprocessing'),return_receipt=True)
    batch,positions=encode_span_request(processor,request,audio)
    mapping=span_labels(processor.tokenizer,request.questions)
    trace_layers = config.get('trace_layers',False)
    if trace_layers:
        from multimodal_jev.span_trace import run_hf_trace
        actual = run_hf_trace(config,batch,positions,mapping,output)
        torch.testing.assert_close(torch.tensor(actual),torch.tensor(reference['candidate_logprobs']),
                                   atol=1e-5,rtol=0)
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_request_identity import bind_span_request
    base_id,base_files=checkpoint_identity(config['model'])
    adapter_id=checkpoint_identity(adapter)[0] if adapter else None
    request_id,request_identity=bind_span_request(request,audio_receipt,batch,positions,
        [m['token_ids'] for m in mapping],base_id+(':'+adapter_id if adapter_id else ''))
    atomic_json(output/'request-identity.json',dict(request_id=request_id,payload=request_identity,
                                                 base_files=base_files,adapter_id=adapter_id))
    # Collapse only the contiguous audio placeholders; engine owns re-expansion.
    ids=batch['input_ids'][0].tolist()
    audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    audio_positions=[i for i,t in enumerate(ids) if t==audio_id]
    if audio_positions!=list(range(min(audio_positions),max(audio_positions)+1)):
        raise ValueError('one contiguous audio placeholder required')
    compact=ids[:audio_positions[0]]+[audio_id]+ids[audio_positions[-1]+1:]
    frames=int(batch['input_features_mask'].sum())
    data={'input_audio_features':batch['input_features'][0,:,:frames].contiguous(),
          'audio_feature_lengths':torch.tensor([frames]),
          'feature_attention_mask':batch['input_features_mask'][:,:frames].contiguous()}
    register_span_model();enable_feature_passthrough()
    atomic_json(output/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
    engine=LLM(model=config['model'],tokenizer=config['model'],runner='pooling',
        hf_overrides={'architectures':['Qwen3ASRForSpanDecisions'],'head_dtype':'model'},
        pooler_config={'task':'token_classify','tok_pooling_type':'ALL','use_activation':False},
        dtype='bfloat16',max_model_len=1024,gpu_memory_utilization=0.3,
        tensor_parallel_size=1,enforce_eager=True,max_num_seqs=1,seed=17,
        worker_extension_cls='multimodal_jev.lora_span_audit.SpanAuditWorkerExtension',
        enable_chunked_prefill=False,enable_prefix_caching=False,
        enable_mm_embeds=True,limit_mm_per_prompt={'audio':1},**lora_options)
    params=PoolingParams(use_activation=False,extra_kwargs={
        'span_positions':positions[0].tolist(),'candidate_ids':[m['token_ids'] for m in mapping],
        'expected_input_ids':ids})
    try:
        if config.get('align_language_rotary',False):
            atomic_json(output/'rotary-intervention.json',
                        engine.llm_engine.collective_rpc('align_jev_language_rotary'))
        if config.get('align_language_attention',False):
            atomic_json(output/'attention-intervention.json',
                        engine.llm_engine.collective_rpc('align_jev_language_attention'))
        if config.get('align_conv_layout',False):
            atomic_json(output/'conv-layout-intervention.json',
                        engine.llm_engine.collective_rpc('align_jev_conv_layout'))
        if config.get('align_language_norms',False):
            atomic_json(output/'norm-intervention.json',
                        engine.llm_engine.collective_rpc('align_jev_language_norms'))
        if adapter:
            atomic_json(output/'worker-lora-before.json',
                        engine.llm_engine.collective_rpc('inspect_jev_span_lora'))
        atomic_json(output/'progress.json',dict(phase='pooling',heartbeat_unix=time.time()))
        prompt={'prompt_token_ids':compact,'multi_modal_data':{'audio':data}}
        base_logits=None
        if adapter:
            base_output=engine.encode([prompt],pooling_params=params,pooling_task='token_classify',use_tqdm=False)
            base_logits=base_output[0].outputs.data.float().cpu()
        if trace_layers:
            engine.llm_engine.collective_rpc('begin_jev_span_trace')
            # Separate multimodal request identity forces a real audio tower
            # call, avoiding reuse of the preceding base request embedding.
            prompt={**prompt,'multi_modal_uuids':{'audio':['jev-trace-adapter']}}
        outputs=engine.encode([prompt],pooling_params=params,pooling_task='token_classify',use_tqdm=False,
                              lora_request=lora_request)
        if trace_layers:
            engine.llm_engine.collective_rpc('finish_jev_span_trace',args=(str(output/'vllm-trace.pt'),))
            from multimodal_jev.span_trace import compare_traces
            atomic_json(output/'trace-comparison.json',compare_traces(
                torch.load(output/'hf-trace.pt',weights_only=True),
                torch.load(output/'vllm-trace.pt',weights_only=True)))
        assert len(outputs)==1 and outputs[0].prompt_token_ids==ids
        logits=outputs[0].outputs.data.float().cpu()
        assert logits.shape==(3,2) and torch.isfinite(logits).all()
        logp=torch.log_softmax(logits,-1)
        hf=torch.tensor(reference['candidate_logprobs']).squeeze(1)
        error=float((logp-hf).abs().max())
        from multimodal_jev.span_parity import compare_span_policies
        joint_parity=compare_span_policies(hf.tolist(),logp.tolist())
        recovery_error=None
        if adapter:
            atomic_json(output/'worker-lora-loaded.json',
                        engine.llm_engine.collective_rpc('inspect_jev_span_lora'))
            recovered=engine.encode([prompt],pooling_params=params,pooling_task='token_classify',use_tqdm=False)
            recovery_error=float((recovered[0].outputs.data.float().cpu()-base_logits).abs().max())
            if recovery_error>1e-5 or torch.equal(logits,base_logits):
                raise ValueError('adapter effect or return-to-base isolation failed')
        return dict(status='diagnostic_complete',input_tokens=len(ids),span_positions=positions.tolist(),
            candidate_logits=logits.tolist(),candidate_logprobs=logp.tolist(),hf_candidate_logprobs=hf.tolist(),
            maximum_logprob_error=error,parity_threshold=0.02,parity_passed=error<=0.02,
            complete_action_parity=joint_parity,rl_admitted=False,
            adapter=adapter,adapter_dtype=lora_dtype if adapter else None,
            request_id=request_id,
            lora_target_modules=packed_targets if adapter else None,
            adapter_logit_effect=float((logits-base_logits).abs().max()) if adapter else None,
            return_to_base_error=recovery_error,
            runner='pooling',generated_tokens=0,optimizer_updates=0,
            scope='Single real-audio native pooling adapter; HF parity diagnostic, no task or RL gain.')
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
