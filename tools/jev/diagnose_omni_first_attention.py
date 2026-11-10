#!/usr/bin/env python3
"""Frozen layer-0 attention decomposition on an authenticated captured TRAIN row.

Loads only layer-0 norm/attention weights and the padding embedding. No full
model generation, transcripts, supervision, optimizer, weight updates or replay
of the task. Compare reconstructed endpoints before interpreting intermediate
differences as the mechanism observed in the original complete model.
"""
import argparse
import inspect
import json
import math
from pathlib import Path
import time

from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.baseline import atomic_json, file_hash

ROW_SHA='7ff443111a1bb5bfd17acb9f5e043c974eb9e7d761ce086498bc63b41bc1afc5'
IDENTITY_SHA='43c3c69f082ad043a74cfff0602962b11226209f22462f2da6d952a00207ae38'
HF_SHA='a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'
LAYERS_INTENT_SHA='c41837a7fa477879b69e0c29aa8d891cccde2ef60ff73a3195eaf5f40c206d56'
LAYER_CONTAINER='ad90d67b4f273cf171cf6b7fb0455a11f4367734b3a797f4fb0430935b916d28'


def run(args):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    torch.set_num_threads(16);torch.manual_seed(17);begin=time.monotonic()
    read=lambda p:json.loads(Path(p).read_text())
    progress=lambda phase:atomic_json(args.output/'progress.json',dict(phase=phase,heartbeat_unix=time.time()))
    if (file_hash(args.inspection)!=args.inspection_hash or file_hash(args.trace_root/'intent.json')!=LAYERS_INTENT_SHA
            or file_hash(args.trace_root/'run/identity.json')!=IDENTITY_SHA
            or file_hash(args.trace_root/'run/row-00.json')!=ROW_SHA
            or file_hash(inspect.getfile(hf))!=HF_SHA or file_hash(args.asset_audit)!=ASSET_AUDIT_SHA):
        raise ValueError('fixed original row, model source and coordinator inspection required')
    observed=read(args.inspection);intent=read(args.trace_root/'intent.json')
    if (observed['Id']!=LAYER_CONTAINER or observed['Image']!=intent['image']
            or observed['State']['OOMKilled'] or observed['State']['Status'] not in ('running','exited')
            or (observed['State']['Status']=='exited' and observed['State']['ExitCode']!=0)
            or observed['Config']['Cmd']!=intent['argv'][intent['argv'].index(intent['image'])+1:]):
        raise ValueError('original live/successful trace container required')
    for name,pin in intent['source_files'].items():
        if file_hash(args.trace_root/'source'/name)!=pin:raise ValueError('original trace source changed')
    identity=read(args.trace_root/'run/identity.json');report=read(args.trace_root/'run/row-00.json')
    if (identity['preflight_only'] is not False or identity['targets_read'] is not False or identity['weight_updates']!=0
            or report['first_difference']!='layer-00.attention' or not all(x['prior_logits_equal'] for x in report['replay'])):
        raise ValueError('reproduced actual first attention difference required')
    trace=args.trace_root/'run/row-00-trace'
    reference={}
    for name in ('layer-00.input','layer-00.attention'):
        path=trace/(name+'.pt');pin=report['reference'][name]
        if file_hash(path)!=pin['sha256']:raise ValueError('captured activation changed')
        x=torch.load(path,map_location='cpu',weights_only=True)
        if list(x.shape)!=pin['shape'] or str(x.dtype)!=pin['dtype'] or not torch.isfinite(x).all():
            raise ValueError('finite original activation required')
        reference[name]=x
    audit=read(args.asset_audit)
    if audit['model']!=str(args.model):raise ValueError('original local model path required')
    pins={entry['file']:entry['sha256'] for entry in audit['files']}
    for name,pin in pins.items():
        if not name.endswith('.safetensors') and file_hash(args.model/name)!=pin:
            raise ValueError('original model metadata/tokenizer changed')
    index=read(args.model/'model.safetensors.index.json')['weight_map']
    config=Qwen3OmniMoeTextConfig(**read(args.model/'config.json')['thinker_config']['text_config'])
    config._attn_implementation='sdpa'
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    pad=tokenizer.pad_token_id
    if type(pad) is not int or pad<0:raise ValueError('original padding token required')
    norm=hf.Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size,eps=config.rms_norm_eps).to(dtype=torch.bfloat16)
    attention=hf.Qwen3OmniMoeThinkerTextAttention(config,0).to(dtype=torch.bfloat16)
    rotary=hf.Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
    names=['thinker.model.layers.0.input_layernorm.weight',
           *['thinker.model.layers.0.self_attn.'+k for k in attention.state_dict()],
           'thinker.model.embed_tokens.weight']
    selected_shards={index[n] for n in names};progress('verify_selected_shards')
    for name in selected_shards:
        if file_hash(args.model/name)!=pins[name]:raise ValueError('original selected weight shard changed')
    tensors={}
    for shard in selected_shards:
        with safe_open(str(args.model/shard),framework='pt',device='cpu') as handle:
            for name in names:
                if index[name]!=shard:continue
                tensors[name]=(handle.get_slice(name)[pad:pad+1] if name.endswith('embed_tokens.weight') else handle.get_tensor(name))
    norm.load_state_dict({'weight':tensors['thinker.model.layers.0.input_layernorm.weight']},strict=True)
    attention.load_state_dict({k:tensors['thinker.model.layers.0.self_attn.'+k] for k in attention.state_dict()},strict=True)
    norm.eval().requires_grad_(False);attention.eval().requires_grad_(False);rotary.eval().requires_grad_(False)
    valid=reference['layer-00.input'];padding=identity['text_padding_counts'][0]
    hidden=[valid.unsqueeze(0),torch.cat((tensors['thinker.model.embed_tokens.weight'].expand(padding,-1),valid),0).unsqueeze(0)]
    if valid.dtype!=torch.bfloat16 or padding!=154 or valid.shape[1]!=config.hidden_size:
        raise ValueError('original row0 input shape and padding required')
    captures=[];masks=[];outputs=[]
    for condition,states in zip(('audio_padding','text_padding'),hidden,strict=True):
        progress(condition);offset=states.shape[1]-len(valid)
        mask=torch.ones(states.shape[:2],dtype=torch.long);mask[:,:offset]=0
        positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,1)
        saved={};handles=[];original_rotary=hf.apply_rotary_pos_emb
        def capture(name,value,axis=1):
            selected=value.detach().select(0,0)
            selected=selected.narrow(axis-1,offset,len(valid)).clone()
            if not torch.isfinite(selected).all():raise ValueError('nonfinite valid attention stage')
            if name in saved:raise ValueError('duplicate attention stage')
            saved[name]=selected
        def hook(name):
            def observe(module,args,output):capture(name,output)
            return observe
        def rope(q,k,*args,**kw):
            q,k=original_rotary(q,k,*args,**kw);capture('q_rope',q,2);capture('k_rope',k,2);return q,k
        try:
            handles.append(norm.register_forward_hook(hook('input_norm')))
            for name in ('q_proj','q_norm','k_proj','k_norm','v_proj'):
                handles.append(getattr(attention,name).register_forward_hook(hook(name)))
            handles.append(attention.o_proj.register_forward_pre_hook(lambda module,args:capture('attention_core',args[0])))
            handles.append(attention.o_proj.register_forward_hook(hook('o_proj')))
            hf.apply_rotary_pos_emb=rope
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                causal=hf.create_causal_mask(config=config,inputs_embeds=states,attention_mask=mask,
                                            past_key_values=None,position_ids=positions)
                pos_emb=rotary(states,positions.unsqueeze(0).expand(3,-1,-1))
                capture('rotary_cos',pos_emb[0]);capture('rotary_sin',pos_emb[1])
                result=attention(norm(states),position_embeddings=pos_emb,attention_mask=causal)[0]
                outputs.append(result[0,offset:].clone())
            masks.append(dict(condition=condition,mask_is_none=causal is None,
                              shape=None if causal is None else list(causal.shape),
                              dtype=None if causal is None else str(causal.dtype)))
        finally:
            hf.apply_rotary_pos_emb=original_rotary
            for handle in handles:handle.remove()
        captures.append(saved)
    expected=['rotary_cos','rotary_sin','input_norm','q_proj','q_norm','k_proj','k_norm','v_proj',
              'q_rope','k_rope','attention_core','o_proj']
    if list(captures[0])!=expected or list(captures[1])!=expected:
        raise ValueError('matching complete stage coverage required')
    stages=[]
    for name,a in captures[0].items():
        b=captures[1][name]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('stage identity differs')
        delta=(a.float()-b.float()).abs()
        for condition,tensor in [('audio_padding',a),('text_padding',b)]:
            torch.save(tensor,args.output/(condition+'-'+name+'.pt'))
        stages.append(dict(stage=name,shape=list(a.shape),dtype=str(a.dtype),equal=bool(torch.equal(a,b)),
            differing_elements=int((a!=b).sum()),max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()),
            tensors={c:file_hash(args.output/(c+'-'+name+'.pt')) for c in ('audio_padding','text_padding')}))
    delta=(outputs[0].float()-outputs[1].float()).abs()
    previous=next(e for e in report['events'] if e['event']=='layer-00.attention')
    endpoints=dict(unpadded_exact_to_full_trace=bool(torch.equal(outputs[0],reference['layer-00.attention'])),
        unpadded_max_abs_error=float((outputs[0].float()-reference['layer-00.attention'].float()).abs().max()),
        padded_difference_count_matches=int((outputs[0]!=outputs[1]).sum())==previous['differing_elements'],
        padded_difference_max_matches=float(delta.max())==previous['max_abs_error'],
        padded_difference_mean_matches=math.isclose(float(delta.mean()),previous['mean_abs_error'],rel_tol=1e-6,abs_tol=1e-12),
        limitation='Full trace did not persist padded per-layer tensor; only unpadded exact equality and reported paired delta statistics can be checked.')
    reproduced=all(endpoints[k] for k in ('unpadded_exact_to_full_trace','padded_difference_count_matches',
                                        'padded_difference_max_matches','padded_difference_mean_matches'))
    progress('verify_selected_shards_after')
    for name in selected_shards:
        if file_hash(args.model/name)!=pins[name]:raise ValueError('selected model shard changed during diagnostic')
    write_once(args.output/'result.json',dict(status='diagnostic_complete',row=0,valid_tokens=len(valid),padding=padding,
        endpoint_statistics_reproduced=reproduced,endpoints=endpoints,masks=masks,stages=stages,
        first_difference=next((e['stage'] for e in stages if not e['equal']),None),
        source_sha256=file_hash(__file__),hf_source_sha256=HF_SHA,row_report_sha256=ROW_SHA,trace_identity_sha256=IDENTITY_SHA,
        selected_shards={n:pins[n] for n in selected_shards},loaded_tensor_names=names,inspection_sha256=args.inspection_hash,
        elapsed_seconds=time.monotonic()-begin,targets_read=False,weight_updates=0,
        scope='Frozen isolated first-layer attention on captured actual TRAIN row0 inputs. Endpoints checked before interpreting stage differences. No full-model task replay, training, NPU validation or parity acceptance.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('trace-root','model','asset-audit','inspection','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--inspection-hash',required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))
