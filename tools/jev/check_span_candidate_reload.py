#!/usr/bin/env python3
"""Independent HF/vLLM TP1 full-candidate gate on native SLURP inputs.

Consumes the CPU-verified label-free input artifact. No labels, optimization,
text generation or promotion beyond the pinned engineering population.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.vllm_spans import register_span_model

# Spawned EngineCore imports this entrypoint before resolving the architecture.
# Register at module scope, matching check_span_runtime_suite.py.
register_span_model()


def validate(config):
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('input asset changed: '+path)
    from multimodal_jev.span_device_scope import validate_tp1
    validate_tp1(config)
    artifact=json.loads(Path(config['inputs']).read_text())
    population=config.get('population',{'recordings':4,'requests':44})
    if (set(population)!={'recordings','requests'} or any(type(n)!=int or n<1 for n in population.values())
            or not artifact['passed'] or artifact['recordings']!=population['recordings']
            or artifact['requests']!=population['requests'] or len(artifact['records'])!=population['requests']
            or len({r['id'] for r in artifact['records']})!=population['requests']):
        raise ValueError('complete pinned CPU-verified input population required')
    return artifact


def prepared(config,artifact):
    from multimodal_jev.simple_prepared_inputs import prepared as render
    return render(config,artifact)


def phase(config,output,backend):
    import torch
    import torch_npu
    from safetensors import safe_open
    from multimodal_jev.span_runtime_profile import configure_actor_profile,install_rollout_profile
    from multimodal_jev.span_candidate_transport import unpack_candidate_logits
    from multimodal_jev.span_parity import compare_span_policies
    artifact=validate(config);torch.set_num_threads(4);torch.npu.set_device(0)
    adapter=Path(config['adapter'])
    with safe_open(str(adapter/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
        if any(saved.get_tensor(k).dtype!=torch.bfloat16 for k in saved.keys()):raise ValueError('BF16 adapter required')
    engine=None;records=[]
    try:
        if backend=='hf':
            from multimodal_jev.asr import load_runtime
            from multimodal_jev.span_model import NativeAudioSpanModel
            from peft import PeftModel
            atomic_json(output/'runtime-profile.json',configure_actor_profile(config['runtime_profile']))
            _,base=load_runtime(config['model'],'npu:0','bfloat16')
            adapted=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
            model=NativeAudioSpanModel(adapted.get_base_model()).eval()
            from multimodal_jev.span_batch_candidate_runtime import install_candidate
            from multimodal_jev.span_audio_collate import forward_span_audio_batch
            import transformers.models.qwen3.modeling_qwen3 as qwen3
            component=config.get('candidate_component','full')
            if component not in ('full','projections','attention'):
                raise ValueError('explicit supported candidate component required')
            original_attention=qwen3.eager_attention_forward
            original_projections=[(m,m.forward) for m in adapted.modules() if type(m) is torch.nn.Linear]
            candidate_handle,candidate_profile=install_candidate(adapted,model)
            if component=='projections':
                qwen3.eager_attention_forward=original_attention
            elif component=='attention':
                for module,forward in original_projections:module.forward=forward
            candidate_profile['active_component']=component
            atomic_json(output/'candidate-profile.json',candidate_profile)
            if any(p.dtype!=torch.bfloat16 for n,p in adapted.named_parameters() if 'lora_' in n):raise ValueError('adapter execution dtype differs')
        else:
            from multimodal_jev.vllm_audio_features import enable_feature_passthrough
            enable_feature_passthrough()
            from vllm import LLM,PoolingParams
            from vllm.lora.request import LoRARequest
            reference=json.loads(Path(config['hf_results']).read_text())
            if reference['base_id']!=config['base_id'] or reference['adapter_id']!=config['adapter_id']:
                raise ValueError('HF policy identity differs')
            references={r['id']:r for r in reference['records']}
            if len(references)!=artifact['requests'] or set(references)!={r['id'] for r in artifact['records']}:
                raise ValueError('complete independent HF population required')
            targets=json.loads((adapter/'adapter_config.json').read_text())['target_modules']
            if not isinstance(targets,list) or not targets or any(not t.startswith('model.language_model.layers.') or t.rsplit('.',1)[-1] not in {'q_proj','v_proj'} for t in targets):
                raise ValueError('explicit q/v targets required')
            packed=sorted({t.replace('model.language_model.','language_model.model.',1).rsplit('.',1)[0]+'.qkv_proj' for t in targets})
            engine=LLM(model=config['model'],tokenizer=config['model'],runner='pooling',
                hf_overrides={'architectures':['Qwen3ASRForSpanDecisions'],'head_dtype':'model'},
                pooler_config={'task':'token_classify','tok_pooling_type':'ALL','use_activation':False},
                dtype='bfloat16',max_model_len=config['max_model_len'],gpu_memory_utilization=.3,
                tensor_parallel_size=1,enforce_eager=True,max_num_seqs=1,seed=17,
                worker_extension_cls=('multimodal_jev.span_batch_rollout_candidate.BatchCandidateWorkerExtension'
                    if config.get('rollout_candidate_component') in ('attention','full_native_lora') else
                    'multimodal_jev.lora_span_audit.SpanAuditWorkerExtension'),
                enable_chunked_prefill=False,enable_prefix_caching=False,enable_mm_embeds=True,
                limit_mm_per_prompt={'audio':1},enable_lora=True,max_loras=1,max_lora_rank=8,
                lora_dtype=torch.bfloat16,lora_target_modules=packed)
            atomic_json(output/'runtime-profile.json',install_rollout_profile(engine,config['runtime_profile']))
            if config.get('rollout_candidate_component') is not None:
                if config['rollout_candidate_component'] not in ('attention','full_native_lora'):
                    raise ValueError('explicit supported rollout candidate required')
                atomic_json(output/'candidate-profile.json',engine.llm_engine.collective_rpc(
                    'align_jev_candidate_packed_attention'))
                if config['rollout_candidate_component']=='full_native_lora':
                    atomic_json(output/'projection-profile.json',engine.llm_engine.collective_rpc(
                        'align_jev_candidate_fixed_base'))
            lora=LoRARequest('native-jev-'+config['adapter_id'],1,str(adapter))
        for row,batch,positions,candidates,rid,identity,processor in prepared(config,artifact):
            if backend=='hf':
                with torch.inference_mode():
                    inputs={k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()}
                    options=dict(pad_token_id=processor.tokenizer.pad_token_id,audio_token_id=base.config.audio_token_id,
                        audio_chunk_size=base.config.audio_config.n_window*2,audio_infer_window=base.config.audio_config.n_window_infer)
                    payload=dict(inputs=batch,positions=positions,candidates=candidates)
                    logp=[p[0].cpu().tolist() for p in forward_span_audio_batch(model,[payload],device='npu:0',dtype=torch.bfloat16,**options)[0]]
                parity=None
            else:
                hf=references[row['id']]
                if hf['request_id']!=rid or hf['request_identity']!=identity:raise ValueError('HF/vLLM exact request mismatch')
                ids=batch['input_ids'][0].tolist();audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
                at=[i for i,x in enumerate(ids) if x==audio_id]
                if not at or at!=list(range(at[0],at[-1]+1)):raise ValueError('contiguous audio tokens required')
                compact=ids[:at[0]]+[audio_id]+ids[at[-1]+1:];frames=int(batch['input_features_mask'].sum())
                data={'input_audio_features':batch['input_features'][0,:,:frames].contiguous(),
                      'audio_feature_lengths':torch.tensor([frames]),'feature_attention_mask':batch['input_features_mask'][:,:frames].contiguous()}
                prompt={'prompt_token_ids':compact,'multi_modal_data':{'audio':data},'multi_modal_uuids':{'audio':[rid]}}
                params=PoolingParams(use_activation=False,extra_kwargs={'span_positions':positions[0].tolist(),
                    'candidate_ids':candidates,'expected_input_ids':ids})
                result=engine.encode([prompt],pooling_params=params,pooling_task='token_classify',use_tqdm=False,lora_request=lora)[0]
                if result.prompt_token_ids!=ids:raise ValueError('worker expanded input differs')
                logp=[p.tolist() for p in unpack_candidate_logits(result.outputs.data,candidates)]
                parity=compare_span_policies(hf['candidate_logprobs'],logp)
            records.append(dict(id=row['id'],request_id=rid,request_identity=identity,candidate_logprobs=logp,parity=parity))
            atomic_json(output/'progress.json',dict(completed=len(records),total=artifact['requests'],updated_unix=time.time()))
        if engine is not None:atomic_json(output/'worker.json',engine.llm_engine.collective_rpc('inspect_jev_span_lora'))
        passed=len(records)==artifact['requests'] and all(r['parity'] is None or r['parity']['passed'] for r in records)
        atomic_json(output/'result.json',dict(passed=passed,records=records,backend=backend,
            base_id=config['base_id'],adapter_id=config['adapter_id'],max_model_len=config['max_model_len'],
            scope='Native full-candidate engineering inputs only; no task labels, training or benchmark result.'))
        if not passed:raise ValueError('native candidate parity failed; original tolerances retained')
    finally:
        if engine is not None:engine.llm_engine.engine_core.shutdown(timeout=30.)


def run(config,output):
    from multimodal_jev.local_ipc import require_local_ipc
    atomic_json(output/'ipc-storage.json',require_local_ipc())
    artifact=validate(config)
    for backend in ('hf','vllm'):
        c={**config,'pins':dict(config['pins'])}
        if backend=='vllm':
            reference=output/'hf/result.json';c['hf_results']=str(reference);c['pins'][str(reference)]=file_hash(reference)
            if json.loads((output/'hf/terminal.json').read_text())['exit_code']!=0:raise ValueError('HF phase failed')
        path=output/(backend+'-config.json');atomic_json(path,c)
        subprocess.run([sys.executable,__file__,'--config',str(path),'--output',str(output/backend),'--phase',backend],check=True)
    atomic_json(output/'READY.json',dict(passed=True,requests=artifact['requests'],base_id=config['base_id'],adapter_id=config['adapter_id'],
        result_sha256=file_hash(output/'vllm/result.json'),scope='Only this frozen engineering input population; no full-dataset training admission.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('hf','vllm','all'),default='all')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        config=json.loads(a.config.read_text());run(config,a.output) if a.phase=='all' else phase(config,a.output,a.phase)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
