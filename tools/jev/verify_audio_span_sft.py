#!/usr/bin/env python3
"""Reload saved native Span adapter and independently check all learning outputs."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash


def verify(config, output):
    import math
    import torch
    import torch_npu
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_training import load_reviewed_bundles
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('isolated physical NPU 4 required')
    root = Path(config['training_root'])
    for path,digest in config['pins'].items():
        if file_hash(path) != digest:raise ValueError('pinned input changed: '+path)
    result = json.loads((root/'run/result.json').read_text())
    train_config = json.loads((root/'config.json').read_text())
    if json.loads((root/'run/terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('successful training terminal required')
    for name,digest in result['adapter_files'].items():
        if file_hash(root/'run/adapter'/name) != digest:raise ValueError('adapter changed')
    bundles,_ = load_reviewed_bundles(train_config['release'],train_config['manifest_sha256'],train_config['question_order'])
    histories = json.loads((root/'run/history.json').read_text())
    assert len(histories)==result['updates'] and sum(x['supervised_spans'] for x in histories)==result['supervised_spans']
    if train_config.get('sampling_mode')=='joint_uniform_complement_pairs_v1':
        from multimodal_jev.span_balance import balanced_binary_batches
        _,balance=balanced_binary_batches(bundles,train_config['seed'])
        assert balance==json.loads((root/'run/balance.json').read_text())
        assert [h['recordings'] for h in histories]==balance['batch_recording_ids']
    else:
        assert sorted(uid for h in histories for uid in h['recordings'])==sorted(b['id'] for b in bundles)
    assert all(math.isfinite(h['gradient_norm']) and h['gradient_norm']>0 for h in histories)
    assert all(math.isfinite(v) for h in histories for v in h['per_recording_losses'])
    identities = json.loads((root/'run/identity.json').read_text())
    summary = {}
    for name in ('before','after','after-silence'):
        saved = json.loads((root/'run'/f'{name}.json').read_text())
        assert [r['id'] for r in saved['records']]==[b['id'] for b in bundles]
        correct=complete=0
        for b,r in zip(bundles,saved['records']):
            observed=[p==g if active else None for p,g,active in zip(r['predictions'],b['targets'],b['supervised'])]
            assert observed==r['correct']
            correct+=sum(c is True for c in observed)
            complete+=all(c is True for c in observed)
        assert correct==saved['correct'] and complete==saved['complete_correct']
        summary[name]=dict(correct=correct,complete_correct=complete)
    torch.set_num_threads(4)
    torch.manual_seed(train_config['seed'])
    processor,base=load_runtime(train_config['model_path'],'npu:0','bfloat16')
    bf16_lora = train_config.get('lora_execution_dtype') == 'bfloat16'
    adapted=PeftModel.from_pretrained(base,root/'run/adapter',is_trainable=False,
                                     autocast_adapter_dtype=not bf16_lora).eval()
    if bf16_lora:
        from multimodal_jev.bf16_lora import cast_lora_bf16
        cast_lora_bf16(adapted)
    model=NativeAudioSpanModel(adapted.get_base_model()).eval()
    saved=json.loads((root/'run/after.json').read_text())['records']
    maximum=0.
    with torch.inference_mode():
        for i,b in enumerate(bundles):
            audio,receipt=load_audio(b['request'].audio,train_config.get('audio_preprocessing'),return_receipt=True)
            if 'audio_processing' in identities['bundles'][i]:
                assert receipt==identities['bundles'][i]['audio_processing']
            batch,positions=encode_span_request(processor,b['request'],audio)
            assert batch['input_ids'].tolist()==identities['bundles'][i]['input_ids']
            assert positions.tolist()==identities['bundles'][i]['positions']
            mapping=span_labels(processor.tokenizer,b['request'].questions)
            batch={k:v.to('npu:0',dtype=model.lm_head.weight.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
            _,logp=model(batch,positions.to('npu:0'),[m['token_ids'] for m in mapping])
            for j,p in enumerate(logp):
                reference=torch.tensor(saved[i]['candidate_logprobs'][j])
                error=float((p.cpu()-reference).abs().max())
                maximum=max(maximum,error)
                torch.testing.assert_close(p.cpu(),reference,atol=1e-5,rtol=0)
            atomic_json(output/'progress.json',dict(completed=i+1,total=len(bundles),heartbeat_unix=time.time()))
    return dict(status='passed',recordings_reloaded=len(bundles),maximum_logprob_error=maximum,
                tolerance=1e-5,recomputed_metrics=summary,updates=len(histories),
                first_update_mean_loss=sum(histories[0]['per_recording_losses'])/len(histories[0]['per_recording_losses']),
                last_update_mean_loss=sum(histories[-1]['per_recording_losses'])/len(histories[-1]['per_recording_losses']),
                scope='Saved adapter output reproducibility and learning metric arithmetic, not held-out performance or CoT gains.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(args.output/'result.json',verify(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
