#!/usr/bin/env python3
"""Real direct-audio multi-span wiring check using the pretrained LM head."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.span_contract import SpanRequest, SpanQuestion, encode_span_request, span_labels
    from multimodal_jev.span_model import NativeAudioSpanModel
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('physical NPU 4 isolation required')
    for path,digest in config['pins'].items():
        if file_hash(path) != digest: raise ValueError('input changed: '+path)
    torch.set_num_threads(4); torch.manual_seed(17)
    processor, base = load_runtime(config['model'],'npu:0','bfloat16')
    model = NativeAudioSpanModel(base).eval()
    tasks = json.loads(Path(config['tasks']).read_text())
    questions = tuple(SpanQuestion(name,DecisionSchema.from_dict(tasks[name]['schema']))
                      for name in ('date','time','timeofday'))
    request = SpanRequest(config['audio'],questions,'timeofday',
                          tuple(tasks['date']['thinking']['requirements']))
    batch, positions = encode_span_request(processor,request,load_audio(config['audio']))
    mapping = span_labels(processor.tokenizer, request.questions)
    atomic_json(output/'encoding.json',{'input_ids':batch['input_ids'].tolist(),
        'positions':positions.tolist(),'mapping':mapping,'request':request.model_input()})
    batch = {k:v.to(device='npu:0',dtype=model.lm_head.weight.dtype if v.is_floating_point() else v.dtype)
             for k,v in batch.items()}
    positions = positions.to('npu:0')
    calls = []
    hook = model.backbone.register_forward_hook(lambda *_: calls.append(1))
    started = time.monotonic()
    try:
        with torch.inference_mode():
            logits, logprobs = model(batch,positions,[m['token_ids'] for m in mapping])
    finally: hook.remove()
    if len(calls) != 1 or logits.shape[:2] != (1,3) or not torch.isfinite(logits).all():
        raise ValueError('single-forward finite three-span contract failed')
    for probs in logprobs:
        torch.testing.assert_close(probs.exp().sum(-1),torch.ones(1,device=probs.device),atol=1e-6,rtol=0)
    return {'status':'passed','backbone_calls':len(calls),'span_positions':positions.tolist(),
            'input_tokens':batch['input_ids'].shape[-1],'audio_frames':int(batch['input_features_mask'].sum()),
            'logits_shape':list(logits.shape),'seconds':time.monotonic()-started,
            'head':'pretrained_native_lm_head','task_finetuned':False,'optimizer_updates':0,
            'candidate_logprobs':[p.tolist() for p in logprobs],
            'encoding_sha256':file_hash(output/'encoding.json'),'input_request':request.model_input(),
            'scope':'Real audio/Qwen3-ASR single forward and native LM-head wiring only. Parallel temporal roles, not a validated CoT dependency or task accuracy.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
