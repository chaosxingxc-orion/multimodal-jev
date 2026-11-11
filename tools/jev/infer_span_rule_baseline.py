#!/usr/bin/env python3
"""Immutable BF16 audio rule controls, optional unmerged adapter, no decoding."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def select_rows(release):
    manifest=json.loads((release/'MANIFEST.json').read_text())
    if manifest['schema']!='jev-span-balanced-rule-controls-v1':raise ValueError('balanced rule controls required')
    for name,digest in manifest['files'].items():
        if file_hash(release/name)!=digest:raise ValueError('dataset changed')
    rows={r['id']:r for r in map(json.loads,(release/'learning.jsonl').read_text().splitlines())}
    schedule=json.loads((release/'schedule.json').read_text())
    selected=[];seen=set()
    # Predeclared coverage probe: first scheduled batch per rule pair, every arm.
    for batch in schedule['batches']:
        if batch['rule_pair_id'] in seen:continue
        seen.add(batch['rule_pair_id'])
        for arm in ('final_only','structured','reordered'):
            selected.extend(rows[k] for k in batch['arms'][arm])
    if len(seen)!=9 or len(selected)!=216 or len({r['id'] for r in selected})!=216:
        raise ValueError('nine rule pairs and 216 matched requests required')
    return selected


def run(config,output):
    import torch
    import torch_npu
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import SpanRequest,SpanQuestion,encode_span_request,span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_runtime_profile import configure_actor_profile
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.policy_identity import checkpoint_identity
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci5']:
        raise ValueError('isolated physical NPU5 required')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned asset changed: '+path)
    rows=select_rows(Path(config['release']))
    if [r['id'] for r in rows]!=config['request_ids']:raise ValueError('fixed selection changed')
    torch.set_num_threads(4);torch.npu.set_device(0);torch.manual_seed(17)
    atomic_json(output/'runtime-profile.json',configure_actor_profile(config['runtime_profile']))
    base_id=checkpoint_identity(config['model'])[0]
    if base_id!=config['base_id']:raise ValueError('base identity changed')
    processor,base=load_runtime(config['model'],'npu:0','bfloat16')
    adapter_id=None
    if config.get('adapter'):
        from peft import PeftModel
        from safetensors import safe_open
        from multimodal_jev.bf16_lora import cast_lora_bf16
        adapter=Path(config['adapter']);adapter_id=checkpoint_identity(adapter)[0]
        if adapter_id!=config['adapter_id']:raise ValueError('adapter identity changed')
        with safe_open(str(adapter/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
            if any(saved.get_tensor(k).dtype!=torch.bfloat16 for k in saved.keys()):
                raise ValueError('BF16 adapter storage required')
        adapted=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
        cast_lora_bf16(adapted)
        base=adapted.get_base_model()
    model=NativeAudioSpanModel(base).eval();records=[]
    policy_id=base_id+':'+(adapter_id or 'no-adapter')
    with (output/'records.jsonl').open('x') as handle,torch.inference_mode():
        for index,row in enumerate(rows):
            raw=row['request']
            request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
                for q in raw['questions']),raw['final_id'],tuple(raw['requirements']),raw['version'])
            if file_hash(request.audio)!=row['provenance']['audio_sha256']:raise ValueError('audio changed')
            audio,receipt=load_audio(request.audio,return_receipt=True)
            inputs,positions=encode_span_request(processor,request,audio)
            mappings=span_labels(processor.tokenizer,request.questions);candidates=[m['token_ids'] for m in mappings]
            request_id,identity=bind_span_request(request,receipt,inputs,positions,candidates,policy_id)
            device_inputs={k:v.to('npu:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            _,probabilities=model(device_inputs,positions.to('npu:0'),candidates)
            if any(not torch.isfinite(p).all() for p in probabilities):raise ValueError('nonfinite probabilities')
            record=dict(id=row['id'],request_id=request_id,request_identity=identity,
                candidate_logprobs=[p[0].cpu().tolist() for p in probabilities],
                predictions=[m['values'][int(p[0].argmax())] for m,p in zip(mappings,probabilities,strict=True)])
            handle.write(json.dumps(record)+'\n');handle.flush();records.append(record)
            atomic_json(output/'progress.json',dict(completed=index+1,total=len(rows),updated_unix=time.time()))
    atomic_json(output/'result.json',dict(base_id=base_id,adapter=adapter_id,recordings=len({r['provenance']['recording_id'] for r in rows}),
        requests=len(records),records_sha256=file_hash(output/'records.jsonl'),
        scope='Immutable HF BF16 direct-audio Span probe; learning-only, no vLLM admission or independent generalization score'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(args.config.read_text()),args.output)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
