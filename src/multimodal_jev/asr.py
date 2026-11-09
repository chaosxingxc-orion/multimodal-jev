"""Qwen3-ASR HF-format inference and frozen feature probes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def load_runtime(model_path, device, dtype):
    import torch
    from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
    config = json.loads((Path(model_path) / 'config.json').read_text())
    if 'thinker_config' in config:
        raise ValueError('Original Qwen ASR format: use the separately pinned official -hf weights')
    processor = Qwen3ASRProcessor.from_pretrained(model_path, local_files_only=True)
    model, loading = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=getattr(torch, dtype),
        attn_implementation='eager', output_loading_info=True)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
        raise RuntimeError(f'non-exact weight load: {loading}')
    return processor, model.to(device).eval()


def prepare(processor, audio_path, device):
    import soundfile as sf
    audio, rate = sf.read(audio_path, dtype='float32')
    if audio.ndim != 1 or rate != 16000:
        raise ValueError('v1 requires mono 16 kHz audio; resampling must be explicit')
    return processor.apply_transcription_request(audio, language='English',
                return_tensors='pt', sampling_rate=rate).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config.get('cpu_threads', 4))
    start = time.monotonic()
    processor, model = load_runtime(config['model_path'], config['device'], config['dtype'])
    inputs = prepare(processor, str(args.audio), config['device'])
    # Cast floating features only, preserve token indices and masks.
    inputs = {k: v.to(dtype=model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=config.get('max_new_tokens', 64), do_sample=False)
        response = processor.decode(generated[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=False)
        clean_response = processor.decode(generated[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        transcription = processor.extract_transcription(clean_response)
        forward = model(**inputs, output_hidden_states=True, use_cache=False)
        positions = torch.arange(inputs['attention_mask'].shape[-1], device=model.device)
        last = (positions * inputs['attention_mask']).max(dim=-1).values
        decoder = forward.hidden_states[-1][torch.arange(len(last), device=model.device), last]
        audio_features = model.get_audio_features(inputs['input_features'],inputs['input_features_mask'],return_dict=True).pooler_output
    receipt = {'status': 'passed', 'config': config,
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        'transcription': transcription, 'raw_response': response,
        'input_shapes': {k: list(v.shape) for k,v in inputs.items()},
        'decoder_feature_shape': list(decoder.shape), 'audio_feature_shape': list(audio_features.shape),
        'finite_decoder_features': bool(torch.isfinite(decoder).all()),
        'finite_audio_features': bool(torch.isfinite(audio_features).all()),
        'elapsed_seconds': time.monotonic()-start,
        'scope': 'single recording inference and frozen feature extraction; not a benchmark reproduction'}
    if not receipt['finite_decoder_features'] or not receipt['finite_audio_features']:
        raise RuntimeError('non-finite features')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(receipt,handle,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
