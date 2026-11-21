"""Cold/warm native KV+prefix execution with complete history restoration.

Finite two-request, four-pass probe; no official target score or release.
"""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from tools.reports.context_batched_request_candidate import prepare_request
from tools.reports.context_cached_history_model import register


def run(c):
    from vllm import LLM,PoolingParams
    from vllm.lora.request import LoRARequest
    register();torch.set_num_threads(2)
    out=Path(c['output']);out.mkdir(exist_ok=False);commit_json(out/'config.json',c)
    bundle=read_execution_bundle(c['snapshot'],expected_policy_id=c['policy_id'],expected_contract=c['contract'],expected_base_id=c['base_id'])
    rows=[];audio=[]
    for entry in c['inputs']:
        p=Path(c['cache'])/(entry['cache_key']+'.pt');raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['cache_sha256']:raise ValueError('input cache changed')
        value=torch.load(p,map_location='cpu',weights_only=True);rows.append(dict(inputs=value['inputs'],positions=value['positions']));audio.append(entry['audio_sha256'])
    owner=SimpleNamespace(bundle=bundle,max_model_len=4096,max_decisions=60)
    prepared=[prepare_request(owner,r,c['audio_token_id'],PoolingParams) for r in rows]
    specs=bundle['contract']['lora'];packed=sorted({n.replace('language_model.','language_model.model.',1).rsplit('.',1)[0]+'.qkv_proj' for n in specs})
    started=time.perf_counter()
    engine=LLM(model=c['model'],tokenizer=bundle['processor'],runner='pooling',
        hf_overrides=dict(architectures=['Qwen3ASRForContextDecisions'],head_dtype='model',jev_context_policy=dict(snapshot=c['snapshot'],policy_id=c['policy_id'],contract=c['contract'],base_id=c['base_id'],profiling_decisions=60)),
        pooler_config=dict(task='token_classify',tok_pooling_type='ALL',use_activation=False),dtype='bfloat16',max_model_len=4096,
        tensor_parallel_size=1,enforce_eager=True,max_num_seqs=4,max_num_batched_tokens=16384,seed=317,
        worker_extension_cls='tools.reports.context_cached_history_model.CachedHistoryWorker',enable_chunked_prefill=False,enable_prefix_caching=True,
        enable_mm_embeds=True,limit_mm_per_prompt={'audio':1},enable_lora=True,max_loras=1,max_cpu_loras=1,max_lora_rank=8,lora_dtype=torch.bfloat16,lora_target_modules=packed,kv_cache_memory_bytes=4*1024**3)
    receipt=engine.llm_engine.collective_rpc('finish_jev_cached_history',args=(c['policy_id'],))
    alignment = None
    if c.get('align_cached_arithmetic', False):
        alignment = engine.llm_engine.collective_rpc('align_jev_cached_history',args=(c['policy_id'],))
    commit_json(out/'startup.json',dict(seconds=time.perf_counter()-started,receipt=receipt,alignment=alignment))
    lora=LoRARequest('context-'+c['policy_id'],1,c['adapter']);reference=None;passes=[]
    for index,order in enumerate(([0,1],[0,1],[1,0],[0,1])):
        prompts=[];params=[]
        for i in order:
            p=prepared[i];prompt=p['prompt'];prompt['multi_modal_uuids']={'audio':[audio[i]]};prompt['cache_salt']=c['policy_id']
            extra=dict(p['params'].extra_kwargs,history_binding=dict(policy_id=c['policy_id'],runtime_id=c['runtime_id'],audio_id=audio[i],audio_start=p['ids'].index(c['audio_token_id'])))
            prompts.append(prompt);params.append(PoolingParams(use_activation=False,skip_reading_prefix_cache=False,extra_kwargs=extra))
        start=time.perf_counter();values=engine.encode(prompts,pooling_params=params,pooling_task='token_classify',lora_request=lora,use_tqdm=False);elapsed=time.perf_counter()-start
        if index==0:
            commit_json(out/'adapter-verification.json',engine.llm_engine.collective_rpc(
                'verify_jev_cached_adapter',args=(c['policy_id'],)))
        canonical=[None]*2
        for slot,v in zip(order,values,strict=True):canonical[slot]=v.outputs.data.detach().cpu()
        if reference is None:reference=canonical
        delta=max((a-b).abs().max().item() for a,b in zip(reference,canonical));stats=engine.llm_engine.collective_rpc('jev_cached_history_stats')
        row=dict(pass_index=index,seconds=elapsed,max_abs_from_cold=delta,exact=all(torch.equal(a,b) for a,b in zip(reference,canonical)),stats=stats,records=[v.tolist() for v in canonical])
        passes.append(row);commit_json(out/f'pass-{index}.json',row)
    commit_json(out/'result.json',dict(passes=passes,formal_admitted=False,scope='Native cold/warm cache only; comparison with frozen aligned runtime still required'))
    commit_json(out/'terminal.json',dict(exit_code=0,formal_admitted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()))
