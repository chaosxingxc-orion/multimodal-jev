#!/usr/bin/env python3
"""One resident TP1 vLLM engine, accepted A/B/A adapters, full audio population."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_parity import compare_span_policies
from queue_span_native_actor_npu import read
from multimodal_jev.vllm_spans import register_span_model

# Spawned EngineCore imports this module before resolving the architecture.
register_span_model()


def validate(config):
    from queue_span_dp4_resume import check_pins
    check_pins(config['pins'])
    if (config['sequence']!=['A','B','A'] or set(config['policies'])!={'A','B'}
            or config['max_model_len']!=4096 or config['physical_devices']!=[4]):
        raise ValueError('fixed A/B/A TP1 population required')
    configs={};records={}
    for key,entry in config['policies'].items():
        root=Path(entry['gate']);c=read(root/'config.json');vl=read(root/'run/vllm/result.json')
        if (entry['adapter_id']!=c['adapter_id'] or checkpoint_identity(c['adapter'])[0]!=entry['adapter_id']
                or c['model']!=config['model'] or c['base_id']!=config['base_id']
                or c['runtime_profile']!=config['runtime_profile'] or not vl['passed']
                or vl['adapter_id']!=entry['adapter_id'] or vl['base_id']!=config['base_id']):
            raise ValueError('accepted immutable policy differs')
        for name in ('config.json','run/vllm/result.json','run/hf/result.json','run/READY.json'):
            if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('complete pinned policy evidence required')
        if c['population']!={'recordings':89,'requests':178}:raise ValueError('full original pilot required')
        configs[key]=c;records[key]={r['id']:r for r in vl['records']}
        if len(records[key])!=178 or len(vl['records'])!=178:raise ValueError('complete distinct reference population required')
    if (configs['A']['inputs']!=configs['B']['inputs'] or records['A'].keys()!=records['B'].keys()
            or configs['A']['adapter_id']==configs['B']['adapter_id']):
        raise ValueError('same inputs and distinct adapters required')
    maximum=max(abs(a-b) for uid in records['A'] for xs,ys in zip(records['A'][uid]['candidate_logprobs'],records['B'][uid]['candidate_logprobs'],strict=True) for a,b in zip(xs,ys,strict=True))
    if maximum<=.04:raise ValueError('reference policies insufficiently distinct to detect stale output')
    return configs,records,maximum


def run(config,out):
    import torch
    import torch_npu
    from vllm import LLM,PoolingParams
    from vllm.lora.request import LoRARequest
    from multimodal_jev.vllm_audio_features import enable_feature_passthrough
    from multimodal_jev.span_runtime_profile import install_rollout_profile
    from multimodal_jev.span_candidate_transport import unpack_candidate_logits
    from check_slurp_native_span_policy import prepared,validate as validate_population
    configs,references,contrast=validate(config)
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4']:raise ValueError('isolated NPU4 required')
    torch.set_num_threads(4);torch.npu.set_device(0);enable_feature_passthrough()
    targets=read(Path(configs['A']['adapter'])/'adapter_config.json')['target_modules']
    if not targets or any(not t.startswith('model.language_model.layers.') or t.rsplit('.',1)[-1] not in ('q_proj','v_proj') for t in targets):raise ValueError('explicit language q/v targets required')
    if targets!=read(Path(configs['B']['adapter'])/'adapter_config.json')['target_modules']:raise ValueError('same LoRA layout required')
    packed=sorted({t.replace('model.language_model.','language_model.model.',1).rsplit('.',1)[0]+'.qkv_proj' for t in targets})
    engine=LLM(model=config['model'],tokenizer=config['model'],runner='pooling',
        hf_overrides={'architectures':['Qwen3ASRForSpanDecisions'],'head_dtype':'model'},
        pooler_config={'task':'token_classify','tok_pooling_type':'ALL','use_activation':False},dtype='bfloat16',
        max_model_len=4096,gpu_memory_utilization=.3,tensor_parallel_size=1,enforce_eager=True,max_num_seqs=1,seed=17,
        worker_extension_cls='multimodal_jev.lora_span_audit.SpanAuditWorkerExtension',
        enable_chunked_prefill=False,enable_prefix_caching=False,enable_mm_embeds=True,
        limit_mm_per_prompt={'audio':1},enable_lora=True,max_loras=1,max_cpu_loras=1,max_lora_rank=8,
        lora_dtype=torch.bfloat16,lora_target_modules=packed)
    reports=[];first={};started=time.monotonic()
    try:
        atomic_json(out/'runtime-profile.json',install_rollout_profile(engine,config['runtime_profile']))
        for phase,key in enumerate(config['sequence']):
            c=configs[key];artifact=validate_population(c);adapter=Path(c['adapter']);slot={'A':1,'B':2}[key]
            lora=LoRARequest('resident-'+c['adapter_id'],slot,str(adapter));rows=[]
            for row,batch,positions,candidates,rid,identity,processor in prepared(c,artifact):
                ref=references[key][row['id']]
                if ref['request_id']!=rid or ref['request_identity']!=identity:raise ValueError('immutable request identity differs')
                ids=batch['input_ids'][0].tolist();audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
                at=[i for i,x in enumerate(ids) if x==audio_id]
                if not at or at!=list(range(at[0],at[-1]+1)):raise ValueError('contiguous audio input required')
                compact=ids[:at[0]]+[audio_id]+ids[at[-1]+1:];frames=int(batch['input_features_mask'].sum())
                data={'input_audio_features':batch['input_features'][0,:,:frames].contiguous(),
                    'audio_feature_lengths':torch.tensor([frames]),'feature_attention_mask':batch['input_features_mask'][:,:frames].contiguous()}
                prompt={'prompt_token_ids':compact,'multi_modal_data':{'audio':data},'multi_modal_uuids':{'audio':[rid]}}
                params=PoolingParams(use_activation=False,extra_kwargs={'span_positions':positions[0].tolist(),
                    'candidate_ids':candidates,'expected_input_ids':ids})
                result=engine.encode([prompt],pooling_params=params,pooling_task='token_classify',use_tqdm=False,lora_request=lora)[0]
                if result.prompt_token_ids!=ids:raise ValueError('expanded token identity differs')
                logp=[p.tolist() for p in unpack_candidate_logits(result.outputs.data,candidates)]
                parity=compare_span_policies(ref['candidate_logprobs'],logp)
                if not parity['passed']:raise ValueError('resident adapter differs from accepted fresh-engine output')
                if phase==0:first[row['id']]=logp
                if phase==2 and logp!=first[row['id']]:raise ValueError('A return probabilities not exact')
                rows.append(dict(id=row['id'],request_id=rid,candidate_logprobs=logp,parity=parity))
                atomic_json(out/'progress.json',dict(phase=phase,policy=key,completed=len(rows),total=178,updated_unix=time.time()))
            worker=engine.llm_engine.collective_rpc('inspect_jev_span_lora')
            if len(worker)!=1 or worker[0]['registered_adapter_ids']!=[slot] or worker[0]['lora_dtype']!='torch.bfloat16':
                raise ValueError('worker did not evict and switch to exact BF16 adapter')
            report=dict(phase=phase,policy=key,adapter_id=c['adapter_id'],slot=slot,worker=worker,records=rows)
            atomic_json(out/f'phase-{phase}.json',report);reports.append(dict(phase=phase,policy=key,requests=len(rows),maximum_logprob_error=max(r['parity']['maximum_logprob_error'] for r in rows)))
        return dict(passed=True,sequence=config['sequence'],reports=reports,reference_policy_contrast=contrast,
            a_return_exact=True,seconds=time.monotonic()-started,
            scope='One resident vLLM TP1 engine, A/B/A LoRA eviction/reload and178 actual audio inputs per phase. No FSDP actor co-residency, optimizer update or full verl trainer claim.')
    finally:engine.llm_engine.engine_core.shutdown(timeout=30.)


if __name__=='__main__':
    # EngineCore spawn needs architecture registration before model loading.
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(read(a.config),a.output));atomic_json(a.output/'terminal.json',{'exit_code':0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
