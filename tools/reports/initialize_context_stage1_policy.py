"""Export and verify a fresh Stage1 policy from a pinned real Qwen-ASR base."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_decision_model import ContextDecisionModel
from multimodal_jev.context_model_profile import from_hf, build_head, validate_against_config
from multimodal_jev.context_policy_snapshot import export_policy, load_policy, lora_contract
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.context_stage_lora import bind_design
from multimodal_jev.context_token_contract import validate_processor_extension
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_request_identity import tensor_identity


def run(c, output):
    from multimodal_jev.model_provider import resolve_provider
    check_paths({'output': output})
    if any(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU initialization must not see NPU devices')
    return resolve_provider(c).initialize_policy(c, output)


def initialize_qwen(c, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter

    if file_hash(c['design']) != c['design_sha256']:
        raise ValueError('Stage1 design changed')
    design = json.loads(Path(c['design']).read_text())
    profile = from_hf(c['model'])
    geometry = validate_against_config(profile, design)
    base_id, files = checkpoint_identity(c['model'])
    if base_id != c.get('base_id',base_id) or files['model.safetensors'] != c['weights_sha256']:
        raise ValueError('official model weights differ')
    processor_root = Path(c['processor'])
    tokens = validate_processor_extension(c['model'], processor_root)
    previous = processor_cache_identity(processor_root.parent)
    if previous != c['cache_processor_sha256']:
        raise ValueError('shared cache processor differs')
    stock = Qwen3ASRProcessor.from_pretrained(c['model'], local_files_only=True)
    processor = Qwen3ASRProcessor.from_pretrained(processor_root, local_files_only=True)
    marker = c['marker']; marker_id = tokens['decision_id']
    if processor.tokenizer.encode(marker, add_special_tokens=False) != [marker_id]:
        raise ValueError('marker does not roundtrip')
    symbols = [processor.tokenizer.encode(x, add_special_tokens=False) for x in c['marker_init_symbols']]
    if len(symbols) != 2 or any(len(x) != 1 for x in symbols) or symbols != [
            stock.tokenizer.encode(x, add_special_tokens=False) for x in c['marker_init_symbols']]:
        raise ValueError('expected two unchanged single-token symbols')
    torch.set_num_threads(c['cpu_workers'])
    _, base = load_runtime(c['model'], 'cpu', 'bfloat16')
    embedding = base.get_input_embeddings().weight
    with torch.no_grad():
        row = embedding[[x[0] for x in symbols]].float().mean(0).to(embedding.dtype)
        embedding[marker_id].copy_(row)
    torch.manual_seed(c['seed'])
    lora = design['stage1']['lora']
    adapted = initialize_adapter(base, dict(lora_rank=lora['rank'], lora_alpha=lora['alpha'],
                                            lora_execution_dtype=lora['execution_dtype'])).eval()
    head = build_head(profile).to(torch.bfloat16).eval()
    model = ContextDecisionModel(adapted.get_base_model().model, head, marker_id).eval()
    if any(p.count_nonzero() for n,p in model.named_parameters() if '.lora_B.' in n):
        raise ValueError('fresh LoRA must have zero B')
    branch = dict(profile.decision_head(), layers=1, position_encoding='inherited_backbone_states_no_extra_branch_encoding',
                  dropout=0., readout='decision_marker_itself',
                  attention_scope='same_request_valid_causal_prefix_including_marker')
    contract = dict(base_id=base_id, runtime_profile=PROFILE, request_version='jev-context-exclusive_event-v1',
        branch=branch, token=dict(id=marker_id,text=marker,row_trainable=False,registration_status='admitted'),
        output=dict(field='prob',activation='sigmoid',normalize_across_candidates=False),
        lora=lora_contract(model.backbone))
    bind_design(design, contract)
    destination = output/'policy'
    policy = export_policy(destination, model, processor_root, contract)
    with torch.no_grad():
        embedding[marker_id].zero_()
    load_policy(destination, model, expected_policy_id=policy,expected_contract=contract,expected_base_id=base_id)
    if not torch.equal(row, embedding[marker_id]) or processor_cache_identity(destination) != previous:
        raise ValueError('restored base-derived marker or processor differs')
    audio_token_id = base.config.audio_token_id
    stock_config = json.loads((Path(c['model'])/'processor_config.json').read_text())
    if audio_token_id != c['audio_token_id'] or stock_config != json.loads((processor_root/'processor_config.json').read_text()):
        raise ValueError('cache acoustic frontend or audio placeholder differs')
    return dict(status='stage1_fresh_cpu_policy_verified', base_id=base_id, base_files=files, policy_id=policy,
        geometry=geometry, marker=tensor_identity(row), processor_sha256=previous, audio_token_id=audio_token_id,
        npu_executed=False, formal_training_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    try:
        result=run(json.loads(a.config.read_text()),a.output)
        commit_json(a.output/'result.json',result)
        commit_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        commit_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
        raise
