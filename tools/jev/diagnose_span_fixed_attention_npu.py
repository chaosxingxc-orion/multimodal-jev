#!/usr/bin/env python3
"""Isolated full-model fixed-tile Linear diagnostic; never installs in production."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.baseline import atomic_json, file_hash
    from multimodal_jev.simple_prepared_inputs import prepared
    from multimodal_jev.span_audio_collate import collate_span_audio
    from multimodal_jev.span_device_scope import validate_tp1
    from multimodal_jev.span_runtime_profile import configure_actor_profile
    from multimodal_jev.span_parity import compare_span_policies
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    validate_tp1(config)
    for name, digest in config['pins'].items():
        if file_hash(name) != digest: raise ValueError('changed dependency: '+name)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.npu.set_device(0)
    configure_actor_profile(config['runtime_profile'])
    processor, base = load_runtime(config['model'], 'npu:0', 'bfloat16')
    adapted = PeftModel.from_pretrained(base, config['adapter'], is_trainable=False,
                                      autocast_adapter_dtype=False).eval()
    # Patch only exact Linear leaves: base projections, LoRA A/B, and head.
    # PEFT wrappers keep their original add/scaling/dropout semantics.
    import types
    from multimodal_jev.span_fixed_shape_linear import fixed_shape_linear
    patched = []
    def fixed_forward(module, inputs):
        if inputs.ndim not in (2, 3):
            raise ValueError('diagnostic requires 2D or 3D Linear inputs')
        values = inputs[None] if inputs.ndim == 2 else inputs
        result = fixed_shape_linear(values, module.weight, module.bias,
                                    tile_rows=128, tile_batch=8)
        return result[0] if inputs.ndim == 2 else result
    for name, module in adapted.named_modules():
        if type(module) is torch.nn.Linear:
            if module.weight.dtype != torch.bfloat16:
                raise ValueError('all patched Linear weights must remain BF16: ' + name)
            module.forward = types.MethodType(fixed_forward, module)
            patched.append(dict(name=name, shape=list(module.weight.shape)))
    if not patched or not any('lora_A' in p['name'] for p in patched):
        raise ValueError('expected base and adapter Linear leaves')
    atomic_json(root/'patched-modules.json',dict(modules=patched,
        tile_rows=128,tile_batch=8,production_profile_changed=False))
    core = adapted.get_base_model().model; head = adapted.get_base_model().lm_head
    # Diagnostic only: projections remain batched; attention uses original
    # per-request shapes. This isolates the kernel, not a throughput solution.
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    original_attention = qwen3.eager_attention_forward
    active_lengths = None
    attention_calls = []
    def independent_attention(module, query, key, value, attention_mask,
                              scaling, dropout=0.0, **kwargs):
        if active_lengths is None or len(active_lengths) != query.shape[0]:
            raise ValueError('explicit full-prefill lengths required')
        outputs = []
        for index, length in enumerate(active_lengths):
            mask = None
            if attention_mask is not None:
                mi = index if attention_mask.shape[0] > 1 else 0
                mask = attention_mask[mi:mi+1, :, :length, :length]
            out, _ = original_attention(module, query[index:index+1, :, :length],
                key[index:index+1, :, :length], value[index:index+1, :, :length],
                mask, scaling=scaling, dropout=dropout, **kwargs)
            outputs.append(torch.nn.functional.pad(out, (0,0,0,0,0,query.shape[2]-length)))
        attention_calls.append(dict(batch=query.shape[0],lengths=list(active_lengths)))
        return torch.cat(outputs,0), None
    qwen3.eager_attention_forward = independent_attention
    artifact = json.loads(Path(config['inputs']).read_text())
    rows = sorted(list(prepared(config, artifact, processor=processor)),
                  key=lambda r: (int(r[1]['input_features_mask'].sum()), r[0]['id']))
    rows = [rows[0], rows[len(rows)//2], rows[-2], rows[-1]]
    opts = dict(pad_token_id=processor.tokenizer.pad_token_id, audio_token_id=base.config.audio_token_id,
                audio_chunk_size=base.config.audio_config.n_window*2,
                audio_infer_window=base.config.audio_config.n_window_infer)
    def device(inputs):
        return {k:v.to('npu:0', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
    def probabilities(embeds, mask, positions, candidates, position_ids=None):
        nonlocal active_lengths
        active_lengths = [int(length) for length in mask.sum(-1).cpu().tolist()]
        hidden = core.language_model(inputs_embeds=embeds, attention_mask=mask,
            position_ids=position_ids, use_cache=False, return_dict=True).last_hidden_state
        selected = hidden.gather(1, positions[...,None].expand(-1,-1,hidden.shape[-1]))
        logits = head(selected).float()
        return [logits[:,j,c].log_softmax(-1) for j,c in enumerate(candidates)]
    def embed(inputs, audio):
        emb = core.get_input_embeddings()(inputs['input_ids'])
        mask = core.get_placeholder_mask(inputs['input_ids'], emb, audio)
        return emb.masked_scatter(mask, audio)
    def serial(item):
        inputs = device(item['inputs'])
        audio = core.get_audio_features(inputs['input_features'], inputs['input_features_mask'], return_dict=True).pooler_output
        logp = probabilities(embed(inputs,audio), inputs['attention_mask'], item['positions'].to('npu:0'), item['candidates'])
        return audio, logp
    def wire(logp): return [v[0].cpu().tolist() for v in logp]
    cases=[]
    with torch.inference_mode():
        for case, selected_rows in [('mixed_short_long', [rows[0],rows[-1]]),
                                    ('long_pair',[rows[-2],rows[-1]]),
                                    ('identical_pair',[rows[1],rows[1]])]:
            items=[dict(inputs=r[1],positions=r[2],candidates=r[3]) for r in selected_rows]
            singles=[serial(item) for item in items]
            packed,pos,candidates=collate_span_audio(items, **opts); packed=device(packed);pos=pos.to('npu:0')
            batched_audio=core.get_audio_features(packed['input_features'],packed['input_features_mask'],return_dict=True).pooler_output
            audio_parts=list(batched_audio.split([s[0].shape[0] for s in singles]))
            fixed_audio=torch.cat([s[0] for s in singles],0)
            batched_text=probabilities(embed(packed,fixed_audio),packed['attention_mask'],pos,candidates,packed['position_ids'])
            full_batch=probabilities(embed(packed,batched_audio),packed['attention_mask'],pos,candidates,packed['position_ids'])
            records=[]
            for i,(item,(original_audio,reference)) in enumerate(zip(items,singles)):
                single_inputs=device(item['inputs'])
                audio_only=probabilities(embed(single_inputs,audio_parts[i]),single_inputs['attention_mask'],
                                         item['positions'].to('npu:0'),candidates)
                padded={k:v[i:i+1] for k,v in packed.items()}
                padding_only=probabilities(embed(padded,original_audio),padded['attention_mask'],pos[i:i+1],candidates,padded['position_ids'])
                values=dict(reference=wire(reference),audio_batch_only=wire(audio_only),
                    text_padding_only=wire(padding_only),language_batch_only=wire([v[i:i+1] for v in batched_text]),
                    full_batch=wire([v[i:i+1] for v in full_batch]))
                records.append(dict(id=selected_rows[i][0]['id'],request_id=selected_rows[i][4],
                    frames=int(item['inputs']['input_features_mask'].sum()),
                    audio_max_abs_difference=float((original_audio.float()-audio_parts[i].float()).abs().max()),
                    probabilities=values, comparisons={k:compare_span_policies(values['reference'],v)
                        for k,v in values.items() if k!='reference'}))
            cases.append(dict(case=case,records=records))
            atomic_json(root/(case+'.json'),cases[-1])
    atomic_json(root/'result.json',dict(completed=True,cases=cases,training_promotion=False,
        patched_modules=len(patched), tile_rows=128, tile_batch=8, attention_calls=attention_calls,
        scope='Candidate BF16 fixed Linear plus per-request unpadded language attention for isolation; projections remain batched. No throughput, backward or vLLM admission.'))


if __name__ == '__main__': main()
