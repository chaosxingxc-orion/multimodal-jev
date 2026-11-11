#!/usr/bin/env python3
"""Sequential real-audio BF16 parity suite in one native vLLM engine.

Consumes independently saved HF probabilities. Never substitutes activations
or transcripts. Labels are only used to describe coverage, not to tune parity.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.vllm_spans import register_span_model

register_span_model()


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from safetensors import safe_open
    from vllm import LLM, PoolingParams
    from vllm.lora.request import LoRARequest
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import SpanRequest, SpanQuestion, encode_span_request, span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.vllm_audio_features import enable_feature_passthrough
    from multimodal_jev.span_runtime_profile import install_rollout_profile
    from multimodal_jev.span_parity import compare_span_policies
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_request_identity import bind_span_request

    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('physical NPU 4 isolation required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('pinned input changed: '+path)
    identity = json.loads(Path(config['identity']).read_text())
    reference = json.loads(Path(config['hf_results']).read_text())
    rows = {row['id']:row for row in reference['records']}
    if len(rows) != len(reference['records']):
        raise ValueError('duplicate HF reference identity')
    adapter = Path(config['adapter'])
    targets = json.loads((adapter/'adapter_config.json').read_text())['target_modules']
    if not isinstance(targets, list) or not targets or any(
            not t.startswith('model.language_model.layers.') or t.rsplit('.',1)[-1] not in {'q_proj','v_proj'}
            for t in targets):
        raise ValueError('explicit language q/v adapters required')
    packed = sorted({t.replace('model.language_model.','language_model.model.',1).rsplit('.',1)[0]+'.qkv_proj'
                     for t in targets})
    with safe_open(str(adapter/'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        if any(saved.get_tensor(k).dtype != torch.bfloat16 for k in saved.keys()):
            raise ValueError('BF16 adapter storage required')
    base_id = checkpoint_identity(config['model'])[0]
    adapter_id = checkpoint_identity(adapter)[0]
    processor = Qwen3ASRProcessor.from_pretrained(config['model'], local_files_only=True)
    enable_feature_passthrough()
    engine = LLM(model=config['model'], tokenizer=config['model'], runner='pooling',
        hf_overrides={'architectures':['Qwen3ASRForSpanDecisions'], 'head_dtype':'model'},
        pooler_config={'task':'token_classify','tok_pooling_type':'ALL','use_activation':False},
        dtype='bfloat16', max_model_len=1024, gpu_memory_utilization=0.3,
        tensor_parallel_size=1, enforce_eager=True, max_num_seqs=1, seed=17,
        worker_extension_cls='multimodal_jev.lora_span_audit.SpanAuditWorkerExtension',
        enable_chunked_prefill=False, enable_prefix_caching=False, enable_mm_embeds=True,
        limit_mm_per_prompt={'audio':1}, enable_lora=True, max_loras=1, max_lora_rank=8,
        lora_dtype=torch.bfloat16, lora_target_modules=packed)
    records = []
    try:
        atomic_json(output/'runtime-profile.json', install_rollout_profile(engine, config['runtime_profile']))
        lora = LoRARequest('jev-suite-'+adapter_id, 1, str(adapter))
        for index, item in enumerate(identity['bundles']):
            raw = item['request']
            request = SpanRequest(raw['audio'], tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
                for q in raw['questions']), raw['final_id'], tuple(raw['requirements']))
            if file_hash(request.audio) != item['audio_sha256']:
                raise ValueError('audio identity changed')
            audio, receipt = load_audio(request.audio, return_receipt=True)
            if receipt != item['audio_processing']:
                raise ValueError('HF reference audio preprocessing differs')
            batch, positions = encode_span_request(processor, request, audio)
            mapping = span_labels(processor.tokenizer, request.questions)
            ids = batch['input_ids'][0].tolist()
            if batch['input_ids'].tolist()!=item['input_ids'] or positions.tolist()!=item['positions']:
                raise ValueError('HF reference expanded input/positions differ')
            request_id, payload = bind_span_request(request, receipt, batch, positions,
                [m['token_ids'] for m in mapping], base_id+':'+adapter_id)
            audio_id = processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
            at = [i for i,t in enumerate(ids) if t==audio_id]
            if not at or at!=list(range(at[0],at[-1]+1)):
                raise ValueError('one contiguous audio span required')
            compact = ids[:at[0]]+[audio_id]+ids[at[-1]+1:]
            frames = int(batch['input_features_mask'].sum())
            data = {'input_audio_features':batch['input_features'][0,:,:frames].contiguous(),
                    'audio_feature_lengths':torch.tensor([frames]),
                    'feature_attention_mask':batch['input_features_mask'][:,:frames].contiguous()}
            params = PoolingParams(use_activation=False, extra_kwargs={'span_positions':positions[0].tolist(),
                'candidate_ids':[m['token_ids'] for m in mapping], 'expected_input_ids':ids})
            prompt = {'prompt_token_ids':compact, 'multi_modal_data':{'audio':data},
                      'multi_modal_uuids':{'audio':[request_id]}}
            result = engine.encode([prompt], pooling_params=params, pooling_task='token_classify',
                                   use_tqdm=False, lora_request=lora)[0]
            if result.prompt_token_ids != ids:
                raise ValueError('worker prompt expansion differs')
            logits = result.outputs.data.float().cpu()
            logp = torch.log_softmax(logits,-1)
            hf = torch.tensor(rows[item['id']]['candidate_logprobs']).squeeze(1)
            audit = compare_span_policies(hf.tolist(),logp.tolist())
            records.append(dict(id=item['id'], request_id=request_id, request_identity=payload,
                input_tokens=len(ids), audio_seconds=len(audio)/16000, labels=item['targets'],
                supervised=item['supervised'], candidate_logits=logits.tolist(), candidate_logprobs=logp.tolist(),
                hf_candidate_logprobs=hf.tolist(), parity=audit))
            atomic_json(output/'progress.json', dict(completed=index+1, total=len(identity['bundles']),
                passed=sum(r['parity']['passed'] for r in records), heartbeat_unix=time.time()))
        atomic_json(output/'worker.json', engine.llm_engine.collective_rpc('inspect_jev_span_lora'))
        return dict(passed=bool(records) and all(r['parity']['passed'] for r in records), records=records,
            runtime_profile=config['runtime_profile'], base_id=base_id, adapter_id=adapter_id,
            optimizer_updates=0, generated_tokens=0,
            scope='Sequential single-request BF16 parity against independently saved HF; learning diagnostics, not held-out performance or batched/distributed acceptance.')
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30.)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        result=run(json.loads(args.config.read_text()),args.output)
        atomic_json(args.output/'result.json',result)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
