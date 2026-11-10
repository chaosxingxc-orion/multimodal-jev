"""Frozen official Qwen2.5-Omni Thinker: direct audio to task text."""
import json
from pathlib import Path
from .baseline import file_hash


def task_messages(wave, definition):
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]},
        {'role': 'user', 'content': [{'type': 'audio', 'audio': wave},
                                     {'type': 'text', 'text': definition}]},
    ]


def validate_thinker_loading(loading, index_keys, state_keys):
    """Every Thinker tensor must load exactly; only other official heads excluded."""
    expected = {k.removeprefix('thinker.') for k in index_keys if k.startswith('thinker.')}
    if not expected or expected != set(state_keys):
        raise ValueError('official Thinker tensor names differ from loaded architecture')
    if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('non-exact Thinker tensor load')
    for key in loading.get('unexpected_keys', []):
        if key not in index_keys or not key.startswith(('talker.', 'token2wav.')):
            raise ValueError('unexpected non-output tensor: ' + key)


def load_runtime(model_path, device, dtype):
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    root = Path(model_path)
    config = json.loads((root/'config.json').read_text())
    if config.get('model_type') != 'qwen2_5_omni':
        raise ValueError('official Qwen2.5-Omni architecture required')
    verified = json.loads((root/'VERIFICATION.json').read_text())
    if verified['status'] != 'verified' or not verified['files'] or any(
            r['repository'] != 'Qwen/Qwen2.5-Omni-3B' or
            r['revision'] != 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
            for r in verified['files']):
        raise ValueError('pinned official frozen base required')
    metadata = json.loads((root/'SOURCE-METADATA.json').read_text())
    if {r['file'] for r in verified['files']} != {r['rfilename'] for r in metadata['siblings']}:
        raise ValueError('incomplete official release verification')
    for record in verified['files']:
        if file_hash(root/record['file']) != record['sha256']:
            raise ValueError('official release file changed: ' + record['file'])
    processor = Qwen2_5OmniProcessor.from_pretrained(root, local_files_only=True)
    model, loading = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        root, local_files_only=True, torch_dtype=getattr(torch, dtype),
        attn_implementation='eager', output_loading_info=True)
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    validate_thinker_loading(loading, index, model.state_dict())
    model.requires_grad_(False)
    return processor, model.to(device).eval()
